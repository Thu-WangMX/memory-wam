"""Evaluation scheduling, text fallback, and stateful server lifecycle regressions."""
import asyncio
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from client_server.ws.model_server import PolicyServerConfig, PolicyServer
from experiments.robodojo.idle_server import IdlePolicyServer
from experiments.robodojo.eval_campaign import jobs, aggregate, entries, client_lanes
from experiments.robodojo.eval_text import TextCache
from experiments.robodojo.eval_servers import launch_policy_batches
from fastwam.datasets.robodojo import text_filename, load_context


class EvalTests(unittest.TestCase):
    def test_fourteen_clients_use_distinct_balanced_servers(self):
        slots=[0,0,1,1,2,2,3,3,4,4,5,6,7,7]
        layout=[{'index':i,'gpu':str(i//2),'port':8200+i} for i in range(16)]
        indices=[0,1,2,3,4,5,6,7,8,9,10,12,14,15]
        lanes=client_lanes(slots,layout,indices)
        self.assertEqual(len(lanes),14)
        self.assertEqual([r['port'] for r in lanes],[8200+i for i in indices])
        self.assertTrue(all(layout[r['server_index']]['gpu']==str(r['gpu']) for r in lanes))
        with self.assertRaises(ValueError):client_lanes(slots,layout,[0]*14)
        with self.assertRaises(ValueError):client_lanes([0,0,0],layout)

    def test_parallel_loading_has_per_gpu_batch_barrier(self):
        layout=[{'index':i,'gpu':str(i//2),'port':8200+i} for i in range(16)]
        launched=[]; ready=set(); attempts={}
        def launch(row):
            if row['index']%2:self.assertTrue(set(range(0,16,2))<=ready)
            launched.append(row['index'])
        def probe(row):
            batch=list(range(row['index']%2,16,2))
            self.assertTrue(set(batch)<=set(launched),'Whole batch must be launched before waiting')
            index=row['index'];attempts[index]=attempts.get(index,0)+1
            if index==0 and attempts[index]==1:return False
            ready.add(index);return True
        with patch('experiments.robodojo.eval_servers.time.sleep'):
            launch_policy_batches(layout,launch,probe,lambda:None)
        self.assertEqual(launched,list(range(0,16,2))+list(range(1,16,2)))
        self.assertEqual(ready,set(range(16)))

    def test_loading_timeout_does_not_start_next_batch(self):
        layout=[{'index':i,'gpu':'0','port':8200+i} for i in range(2)]
        launched=[]
        with patch('experiments.robodojo.eval_servers.time.monotonic',side_effect=[0,2]):
            with self.assertRaises(TimeoutError):
                launch_policy_batches(layout,lambda row:launched.append(row['index']),lambda row:False,lambda:None,timeout=1)
        self.assertEqual(launched,[0])

    def test_result_read_retries_partial_write(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'result.json';path.touch()
            with patch.object(Path,'read_text',side_effect=['', '{"details":{"0":{"layout_id":1}},"eval_time":1}']):
                self.assertEqual(entries(path),[('0',{'layout_id':1})])

    def test_protocol_and_task_weighting(self):
        plan=jobs()
        self.assertEqual(len(plan),54)
        self.assertEqual(sum(j['episodes'] for j in plan),840)
        records={j['task']:[{'success':j['dimension']=='Memory','score':int(j['dimension']=='Memory')}
                            for _ in range(j['episodes'])] for j in plan}
        summary=aggregate(records)
        self.assertEqual(summary['complete_base_tasks'],42)
        self.assertAlmostEqual(summary['overall']['score'],100*6/42)
        records['cover_blocks'].pop()
        self.assertIsNone(aggregate(records)['overall'])

    def test_text_cache_padding_and_fastwam_mask(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); source=root/'source'; source.mkdir()
            context=torch.randn(20,4096); mask=torch.ones(20,dtype=torch.bool); mask[-2:]=False
            torch.save({'context':context,'mask':mask},source/text_filename('unseen training instruction'))
            cache=TextCache(source,root/'output')
            cache.ensure('unseen training instruction')
            result,result_mask=load_context(cache.output,'unseen training instruction')
            torch.testing.assert_close(result[:18],context[:18])
            self.assertEqual(result[18:].count_nonzero(),0)
            self.assertTrue(result_mask.all())
            self.assertIsNone(cache.encoder)

    def test_status_is_not_activity_and_observations_are(self):
        class Model:
            def status(self):return {}
            def update_obs(self,obs):return obs
            def fail(self):raise ValueError('bad observation')
        async def run():
            model=Model(); server=IdlePolicyServer(model,PolicyServerConfig(port=0))
            await server._call_model_method(model.status)
            self.assertIsNone(server.last_activity)
            self.assertEqual(await server._call_model_method(model.update_obs,42),42)
            self.assertIsNotNone(server.last_activity)
            self.assertEqual(server.active_calls,0)
            with self.assertRaises(ValueError):await server._call_model_method(model.fail)
            self.assertEqual(server.failure,'bad observation')
            self.assertEqual(server.active_calls,0)
        asyncio.run(run())

    def test_first_request_and_idle_exit_and_active_request(self):
        async def run():
            first=IdlePolicyServer(object(),PolicyServerConfig(host='127.0.0.1',port=0),first_timeout=.05,idle_timeout=.05)
            await asyncio.wait_for(first.serve_forever(),2.5)
            class Model:
                def update_obs(self):time.sleep(1.3)
            model=Model(); server=IdlePolicyServer(model,PolicyServerConfig(host='127.0.0.1',port=0),first_timeout=.05,idle_timeout=.05)
            lifecycle=asyncio.create_task(server.serve_forever())
            await asyncio.sleep(.1)
            request=asyncio.create_task(server._call_model_method(model.update_obs))
            await asyncio.sleep(1.05)
            self.assertFalse(lifecycle.done(),'Active model calls must prevent idle shutdown')
            await request
            await asyncio.wait_for(lifecycle,2.5)
        asyncio.run(run())

    def test_second_connection_rejected(self):
        class WebSocket:
            def __init__(self):self.code=None
            async def close(self,code,reason):self.code=code
        async def run():
            entered=asyncio.Event(); release=asyncio.Event()
            async def handler(instance,ws):entered.set();await release.wait()
            server=IdlePolicyServer(object(),PolicyServerConfig(port=0))
            with patch.object(PolicyServer,'_handle_connection',handler):
                task=asyncio.create_task(server._handle_connection(WebSocket()))
                await entered.wait()
                second=WebSocket(); await server._handle_connection(second)
                self.assertEqual(second.code,1013)
                release.set(); await task
                self.assertIsNone(server.connection)
        asyncio.run(run())


if __name__=='__main__':unittest.main()
