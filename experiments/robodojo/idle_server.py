"""One stateful client per server, with request-aware idle shutdown."""
import asyncio
import logging
import time
from client_server.ws.model_server import PolicyServer


class IdlePolicyServer(PolicyServer):
    def __init__(self, model, config, first_timeout=7200, idle_timeout=900):
        super().__init__(model,config)
        self.first_timeout=float(first_timeout); self.idle_timeout=float(idle_timeout)
        if min(self.first_timeout,self.idle_timeout)<=0: raise ValueError('Timeouts must be positive')
        self.last_activity=None; self.active_calls=0; self.connection=None
        self.failure=None

    async def _handle_connection(self, websocket):
        if self.connection is not None:
            await websocket.close(code=1013,reason='One stateful client per Helios server')
            return
        self.connection=websocket
        try: await super()._handle_connection(websocket)
        finally: self.connection=None

    async def _call_model_method(self, method, *args):
        if getattr(method,'__name__','')=='status':
            return await super()._call_model_method(method,*args)
        self.active_calls+=1
        try:
            result=await super()._call_model_method(method,*args)
            self.last_activity=time.monotonic()
            return result
        except Exception as exc:
            self.failure=str(exc)
            raise
        finally: self.active_calls-=1

    async def serve_forever(self):
        await self.start(); ready=time.monotonic()
        logging.info('HELIOS_SERVER_READY first_timeout=%s idle_timeout=%s',self.first_timeout,self.idle_timeout)
        try:
            while True:
                await asyncio.sleep(1)
                if self.failure is not None:
                    raise RuntimeError('Model request failed: '+self.failure)
                origin=ready if self.last_activity is None else self.last_activity
                limit=self.first_timeout if self.last_activity is None else self.idle_timeout
                if not self.active_calls and time.monotonic()-origin>=limit:
                    logging.info('HELIOS_IDLE_EXIT reason=%s', 'first_request_timeout' if self.last_activity is None else 'idle_timeout')
                    return
        finally: await self.stop()
