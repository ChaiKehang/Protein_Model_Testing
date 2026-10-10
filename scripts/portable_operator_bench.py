"""Same CPU-generated inputs across devices; shape-derived diagnostic microbenchmarks."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
import torch
import torch.nn.functional as F
from common.backend import Backend, DTYPES
from common.reporting import stats, write_json, environment, sha256


def run(args):
    output=Path(args.output); output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    backend=Backend(args.device)
    shape_path=Path(args.shapes)
    shapes=json.loads(shape_path.read_text())
    linear=next(s for s in shapes if s['type']=='Linear' and s['module'].startswith('transformer.') and len(s['input_shape'])==3)
    shape=linear['input_shape']; out_features,in_features=linear['weight_shape']
    dtype=DTYPES[args.precision]
    generator=torch.Generator().manual_seed(7)
    def sample(size):return torch.randn(size,generator=generator).to(dtype)
    x=sample(shape); weight=sample([out_features,in_features]); bias=sample([out_features])
    attention=next(s for s in shapes if 'n_heads' in s and s['input_shape']==shape)
    heads=attention['n_heads']
    qshape=[shape[0],heads,shape[-2],in_features//heads]
    q,k,v=(sample(qshape) for _ in range(3))
    table=sample([64,in_features]); ids=torch.randint(0,64,shape[:2],generator=generator)
    ln=next((s for s in shapes if s['type']=='LayerNorm'),None)
    ln_shape=ln['input_shape'] if ln else shape
    ln_x=sample(ln_shape)
    normalized=ln['normalized_shape'] if ln else [in_features]
    specs=[('linear',lambda a,w,b:F.linear(a,w,b),[x,weight,bias],2*__import__('math').prod(shape[:-1])*in_features*out_features),
           ('layer_norm',lambda a:F.layer_norm(a,normalized,eps=1e-5),[ln_x],None),
           ('softmax',lambda a:torch.softmax(a,dim=-1),[x],None),
           ('vector_norm',lambda a:torch.linalg.vector_norm(a,dim=-1),[x],None),
           ('topk',lambda a:torch.topk(a,3,dim=-1).values,[x],None),
           ('embedding',lambda t,i:F.embedding(i,t),[table,ids],None),
           ('batch_gemm',lambda a,b:torch.matmul(a,b.transpose(-1,-2)),[q,k],2*shape[0]*heads*shape[-2]**2*qshape[-1]),
           ('sdpa',lambda a,b,c:F.scaled_dot_product_attention(a,b,c),[q,k,v],None)]
    results=[]
    with torch.inference_mode(),backend.context():
        for name,fn,inputs,flops in specs:
            cpu=[a.float() if a.is_floating_point() else a for a in inputs]
            reference=fn(*cpu)
            actual=[a.to(backend.device) for a in inputs]
            times=[]; device_times=[]; events=backend.events()
            for _ in range(5):value=fn(*actual)
            backend.synchronize();backend.reset_peak()
            for _ in range(args.runs):
                backend.synchronize();start=time.perf_counter()
                if events:events[0].record()
                value=fn(*actual)
                if events:events[1].record()
                backend.synchronize();times.append(time.perf_counter()-start)
                if events:device_times.append(events[0].elapsed_time(events[1]))
            observed=value.cpu().float()
            difference=(observed-reference).abs()
            results.append({'operator':name,'input_shapes':[list(a.shape) for a in inputs],
                            'input_strides':[list(a.stride()) for a in inputs],
                            'input_dtypes':[str(a.dtype) for a in inputs],
                            'input_sha256':[hashlib.sha256(a.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() for a in inputs],
                            'latency':stats(times),'device_event_ms':stats(device_times),
                            'memory':backend.memory(),'fp32_reference':{'max_abs_error':float(difference.max()),
                            'mean_abs_error':float(difference.mean()),'rmse':float(difference.square().mean().sqrt()),
                            'finite':bool(torch.isfinite(observed).all()),'scope':'CPU FP32 on already quantized identical inputs; descriptive error, not final task tolerance'},
                            'estimated_flops':flops,'estimated_tflops_per_second':flops/stats(times)['median']/1e12 if flops else None,
                            'measured_dram_bytes':None})
            del actual,value
    report={'device':backend.metadata(),'precision':args.precision,'environment':environment(REPO),
            'source_shapes':str(shape_path.resolve()),'source_shapes_sha256':sha256(shape_path),
            'observed_linear_spec':linear,'observed_layernorm_spec':ln,'observed_attention_spec':attention,'results':results,
            'scope':'synthetic CPU-seeded inputs using observed Linear/LayerNorm shape and attention head dimensions; no real model weights, mask or activation values',
            'limitations':['No DRAM counters, cache/occupancy inference or measured roofline.',
                           'LayerNorm without affine parameters is a simplified diagnostic, not full module replay.',
                           'SDPA unmasked synthetic path is not ESM3 geometry/sequence_id path.',
                           'MUSA operator and event support require actual execution; no CPU fallback.']}
    write_json(output/'microbench.json',report)
    print('OPERATOR_BENCH_DONE',output,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--shapes',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--precision',choices=DTYPES,default='bf16')
    p.add_argument('--runs',type=int,default=30)
    a=p.parse_args()
    if a.runs<1:p.error('runs must be positive')
    run(a)
