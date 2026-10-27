"""Validate exact prefix equivalence and timing using the deployed VAE weights."""
import argparse
import json
import time
from pathlib import Path
import torch
from fastwam.models.wan22.helpers.loader import _load_registered_model
from experiments.robomme.streaming_vae import CausalVAEStream


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--vae', required=True)
    p.add_argument('--output', required=True)
    args=p.parse_args()
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(4*2**30/torch.cuda.get_device_properties(0).total_memory)
    vae=_load_registered_model(args.vae,'wan_video_vae',torch.bfloat16,'cuda:0').eval()
    video=(torch.rand((1,3,129,384,320),generator=torch.Generator().manual_seed(721))*2-1).bfloat16()
    stream=CausalVAEStream(vae)
    rows=[]
    streamed=None
    for i in range(33):
        chunk=video[:,:,:1] if i==0 else video[:,:,1+4*(i-1):1+4*i]
        torch.cuda.synchronize();start=time.perf_counter()
        streamed=stream.append(chunk.cuda())
        torch.cuda.synchronize();incremental=time.perf_counter()-start
        if i in (0,1,4,16,32):
            torch.cuda.synchronize();start=time.perf_counter()
            reference=vae.encode(video[:,:,:1+4*i].cuda(),device='cuda:0',tiled=False)
            torch.cuda.synchronize();full=time.perf_counter()-start
            delta=(streamed.float()-reference.float()).abs()
            row=dict(latents=i+1,bitwise_equal=torch.equal(streamed,reference),
                max_error=delta.max().item(),mean_error=delta.mean().item(),
                full_prefix_seconds=full,incremental_seconds=incremental)
            rows.append(row);print(json.dumps(row),flush=True)
            assert row['bitwise_equal'], row
    # Reusing the same VAE for demo encoding must not contaminate a new phase.
    vae.encode(video[:,:,:17].cuda(),device='cuda:0',tiled=False)
    stream.reset()
    after_demo=stream.append(video[:,:,:1].cuda())
    expected=vae.encode(video[:,:,:1].cuda(),device='cuda:0',tiled=False)
    assert torch.equal(after_demo,expected)
    stream.reset()
    assert not stream.encoded_chunks and all(x is None for x in stream.feature_cache)
    report=dict(prefix_checks=rows,phase_reset_exact=True,
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    print('STREAMING_VAE_CHECK_PASSED',flush=True)


if __name__=='__main__':
    main()
