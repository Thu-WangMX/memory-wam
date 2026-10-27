"""Shared CPU T5 fallback for evaluation instructions outside the training set."""
import argparse
import json
import os
from pathlib import Path
import socket
import socketserver
import threading
import torch
from fastwam.datasets.robodojo import PROMPT, text_filename, load_context


def request_context(instruction, socket_path):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(580)
        client.connect(socket_path)
        client.sendall((json.dumps({'instruction': instruction}) + '\n').encode())
        reply = json.loads(client.makefile('rb').readline(1024 * 1024))
    if 'error' in reply:
        raise RuntimeError(reply['error'])
    return load_context(reply['cache_dir'], instruction)


class TextCache:
    def __init__(self, source, output):
        self.source, self.output = Path(source), Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.encoder = self.tokenizer = None

    @torch.no_grad()
    def ensure(self, instruction):
        if not isinstance(instruction, str) or not instruction.strip() or len(instruction) > 10000:
            raise ValueError('Invalid evaluation instruction')
        path = self.output / text_filename(instruction)
        with self.lock:
            if path.exists():
                load_context(self.output, instruction)
                return
            source = self.source / path.name
            if source.exists():
                payload = torch.load(source, map_location='cpu', weights_only=True)
                context, mask = payload['context'], payload['mask'].bool()
                if context.ndim != 2 or context.shape[1] != 4096 or not 0 < len(context) <= 128:
                    raise ValueError('Incompatible existing T5 context')
                if mask.shape != (len(context),):
                    raise ValueError('Incompatible T5 mask')
                padded = context.new_zeros((128,4096)); padded[:len(context)] = context
                padded_mask = torch.zeros(128,dtype=torch.bool); padded_mask[:len(mask)] = mask
                payload = {'context': padded, 'mask': padded_mask}
            else:
                if self.encoder is None:
                    from fastwam.models.wan22.helpers.loader import _load_registered_model, _resolve_configs
                    from fastwam.models.wan22.wan_video_text_encoder import HuggingfaceTokenizer
                    _, text, _, tokenizer = _resolve_configs('Wan-AI/Wan2.2-TI2V-5B', 'Wan-AI/Wan2.1-T2V-1.3B', True)
                    text.download_if_necessary(); tokenizer.download_if_necessary()
                    self.encoder = _load_registered_model(text.path, 'wan_video_text_encoder', torch_dtype=torch.bfloat16, device='cpu').eval().requires_grad_(False)
                    self.tokenizer = HuggingfaceTokenizer(name=tokenizer.path, seq_len=128, clean='whitespace')
                ids, mask = self.tokenizer(PROMPT.format(task=instruction), return_mask=True, add_special_tokens=True)
                context = self.encoder(ids, mask.bool())[0]
                context[~mask[0].bool()] = 0
                payload = {'context':context.cpu(), 'mask':mask[0].bool().cpu()}
                print(json.dumps({'event':'encoded_new_instruction','instruction':instruction}), flush=True)
            if not torch.isfinite(payload['context']).all():
                raise ValueError('Nonfinite T5 context')
            temporary = path.with_suffix(f'.tmp.{os.getpid()}')
            torch.save(payload, temporary); temporary.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--socket',required=True); parser.add_argument('--source',required=True)
    parser.add_argument('--output',required=True); args=parser.parse_args()
    torch.set_num_threads(int(os.getenv('POLICY_TEXT_THREADS','8')))
    cache=TextCache(args.source,args.output)
    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            try:
                data=json.loads(self.rfile.readline(65536))
                cache.ensure(data['instruction']); result={'cache_dir':str(cache.output)}
            except Exception as exc:
                result={'error':str(exc)}
            self.wfile.write((json.dumps(result)+'\n').encode())
    class Server(socketserver.ThreadingUnixStreamServer):
        daemon_threads=True
    if Path(args.socket).exists():
        raise FileExistsError('Use a unique socket for this server group')
    try:
        with Server(args.socket,Handler) as server:
            os.chmod(args.socket,0o600)
            print('TEXT_SERVICE_READY',flush=True); server.serve_forever()
    finally:
        Path(args.socket).unlink(missing_ok=True)


if __name__=='__main__': main()
