#!/usr/bin/env python3
"""Small real-data check before a full extraction; uses the same LTM functions."""
import json
import sys
import time
import numpy as np
import torch
import extract_labels as e


def main():
    torch.set_num_threads(2)
    ref, runtime = e.load_reference(e.BASE)
    dataset, metadata = e.load_dataset(ref, e.DATA)
    checkpoint = e.BASE / 'models/LAM_400k.ckpt'
    verified = json.loads((checkpoint.parent/'verified.json').read_text())
    assert verified['sha256'] == e.CHECKPOINT_SHA and checkpoint.stat().st_size == e.CHECKPOINT_BYTES
    identity = ref.CheckpointIdentity(checkpoint,e.CHECKPOINT_SHA,e.CHECKPOINT_BYTES,ref.CHECKPOINT_FORMAT,-1)
    tick = time.monotonic()
    model, encode, preprocess, identity = ref.load_dreamdojo_encoder(runtime, identity, 'cuda:0')
    print('MODEL_READY',round(time.monotonic()-tick,2),'step',identity.step,flush=True)
    source=dataset.episodes[0]
    prepared=e.prepare(ref,source,dataset,metadata[source.episode_index],preprocess)
    head=prepared['lam_frames'][0]
    def run(frames,batch):
        return ref.encode_causal_latents(model,encode,frames,device='cuda:0',pair_batch_size=batch,transition_frame_stride=1)
    warm=run(head[:17],16)
    repeated=run(head[:17],16)
    np.testing.assert_array_equal(warm,repeated)
    smallbatch=run(head[:17],1)
    max_error=float(np.max(np.abs(warm-smallbatch)))
    rmse=float(np.sqrt(np.mean((warm-smallbatch)**2)))
    assert np.isfinite(warm).all() and np.std(warm[1:])>1e-6
    assert rmse<0.02,('unexpected large batch dependence',rmse)
    torch.cuda.synchronize();tick=time.monotonic()
    pair=run(head,128)
    torch.cuda.synchronize();elapsed=time.monotonic()-tick
    result={'status':'passed','checkpoint_step':identity.step,'torch_version':torch.__version__,
            'same_batch_repeat_exact':True,'batch1_vs_batch16_max_abs':max_error,
            'batch1_vs_batch16_rmse':rmse,'pairs':len(head)-1,'encode_seconds':elapsed,
            'pairs_per_second':(len(head)-1)/elapsed,'latent_std':float(pair[1:].std()),
            'peak_cuda_gib':torch.cuda.max_memory_allocated()/1024**3,
            'reference':'LTM unmodified load_dreamdojo_encoder + encode_causal_latents'}
    e.write_json(e.BASE/'validation/encoder.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
