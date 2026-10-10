"""Frozen template-free OpenFold inference; formal timings and optional diagnostics."""
import argparse
import contextlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import torch
from common.backend import Backend
from common.reporting import environment, sha256, stats, write_json
from models.openfold_adapter import OpenFoldAdapter, PINNED_COMMIT


def snapshot_sources(root):
    names = ['models/openfold_adapter.py','scripts/openfold_baseline.py','common/backend.py','common/reporting.py']
    for name in names:
        out = root/'source'/name
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes((REPO/name).read_bytes())
    return {name:sha256(REPO/name) for name in names}


def main(args):
    root=Path(args.output).resolve();root.mkdir(parents=True, exist_ok=False)
    report={'status':'running','source_sha256':snapshot_sources(root),'parameters':vars(args),
            'model':'OpenFold model_3_ptm, finetuning_no_templ_ptm_1',
            'precision':'FP32; TF32 disabled; no mixed-precision wrapper',
            'scope':'precomputed MSA; template-free monomer; unrelaxed PDB; includes all default layers and three recycling steps',
            'evaluation_policy':'Reference-only measurement; no optimization applied.'}
    try:
        backend=Backend(args.device);torch.set_num_threads(args.cpu_threads)
        report.update(environment=environment(REPO), device=backend.metadata(),
                      weights={'path':args.weights,'sha256':sha256(args.weights)},
                      input_sha256={'fasta':sha256(args.fasta),
                                    'alignments':{p.name:sha256(p) for p in sorted(Path(args.alignments).iterdir()) if p.is_file()}})
        for name in ['openfold','scipy','biopython','ml-collections','dm-tree']:
            try:report['environment']['versions'][name]=importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:report['environment']['versions'][name]=None
        def measured(name, fn):
            backend.synchronize();start=time.perf_counter();value=fn();backend.synchronize()
            report.setdefault('phases_seconds',{})[name]=time.perf_counter()-start
            print(name,report['phases_seconds'][name],flush=True)
            write_json(root/'report.json',report)
            return value
        adapter=measured('model_load',lambda:OpenFoldAdapter(backend,args.weights,args.chunk_size,args.seed,args.attention_kernel))
        report['openfold_commit']=PINNED_COMMIT
        report['attention_implementation']={'policy':args.attention_kernel,'explicit_torch_hooks':len(adapter.attention_handles),
            'note':'torch selects pinned upstream matmul/softmax Attention and non-inplace IPA branches; not a CPU device fallback. Numerical aggregation order differs from native CUDA branches.'}
        report['config']=adapter.config.to_dict()
        report['features']=measured('prepare',lambda:adapter.prepare(args.fasta,args.alignments,args.features))
        torch.save(adapter.features_cpu,root/'features.pt')
        backend.reset_peak()
        result=measured('first_inference',adapter.run)
        report['first_inference_memory']=backend.memory()
        times=[]
        if not args.numerical_only:
            del result
            for i in range(args.warmup):
                _=measured(f'warmup_{i}',adapter.run)
                del _
            backend.reset_peak()
            for i in range(args.runs):
                if i:
                    del result
                backend.synchronize();start=time.perf_counter();result=adapter.run();backend.synchronize()
                times.append(time.perf_counter()-start)
                print('formal',i,times[-1],flush=True)
        report['latency_seconds']=stats(times);report['samples_seconds']=times
        report['timing_scope']='Warmed full model with prepared device tensors; excludes preprocessing, model load, CPU transfer, PDB and profiler'
        report['memory']=backend.memory()
        report['memory_scope']='Formal full-model calls after warmup; includes prepared inputs and model, one live output; first-call peak reported separately'
        if args.numerical_only:
            report['timing_scope']='One complete first-call numerical reference only; no warmed formal benchmark'
            report['memory_scope']='Full first-call reference with prepared inputs and model'
        positions=result['final_atom_positions'];mask=result['final_atom_mask'].bool()
        n=adapter.features_cpu['aatype'].shape[0]
        if positions.shape != (n,37,3) or not torch.isfinite(positions[mask]).all() or not mask[:,1].all():
            raise ValueError('Nonfinite coordinates, unexpected atom37 shape, or missing predicted CA')
        report['output']={'residues':n,'positions_shape':list(positions.shape),
                          'finite_valid_atoms':True,'valid_ca':int(mask[:,1].sum()),
                          'num_model_passes':int(result['num_recycles']),
                          'plddt_mean':float(result['plddt'].mean()),
                          'plddt_note':'Model confidence, not experimental accuracy'}
        if int(result['num_recycles']) != adapter.config.data.common.max_recycling_iters+1:
            raise ValueError('Unexpected number of recycling passes')
        (root/'prediction_unrelaxed.pdb').write_text(adapter.pdb(result))
        from openfold.utils.tensor_utils import tensor_tree_map
        payload=tensor_tree_map(lambda t:t.detach().cpu(),result)
        torch.save(payload,root/'tensors.pt')
        if args.profile:
            # Separate diagnostic pass; never substitute its timings for formal ones.
            shape_rows={}
            handles=[]
            def hook(name):
                def record(module,inputs,output):
                    tensors=[v for v in inputs if isinstance(v,torch.Tensor)]
                    key=json.dumps([name,[list(v.shape) for v in tensors],[str(v.dtype) for v in tensors]])
                    if key not in shape_rows:shape_rows[key]={'module':name,'type':type(module).__name__,
                        'inputs':[{'shape':list(v.shape),'stride':list(v.stride()),'dtype':str(v.dtype)} for v in tensors],'calls':0}
                    shape_rows[key]['calls']+=1
                return record
            for name,module in adapter.model.named_modules():
                if any(s in type(module).__name__ for s in ['Triangle','Evoformer','OuterProduct','InvariantPoint','StructureModule']):
                    handles.append(module.register_forward_hook(hook(name)))
            activities=[torch.profiler.ProfilerActivity.CPU]
            if backend.device.type=='cuda':activities.append(torch.profiler.ProfilerActivity.CUDA)
            try:
                with torch.profiler.profile(activities=activities, record_shapes=True) as prof:
                    with torch.profiler.record_function('openfold_full_model'):
                        _=adapter.run();backend.synchronize()
                prof.export_chrome_trace(str(root/'torch_trace.json'))
                events=prof.key_averages(group_by_input_shape=True)
                write_json(root/'operators.json',{'operators':[{'name':e.key,'count':e.count,'shapes':e.input_shapes,
                    'self_cpu_us':e.self_cpu_time_total,'self_device_us':e.self_device_time_total} for e in events]})
                write_json(root/'protein_modules.json',list(shape_rows.values()))
            finally:
                for handle in handles:handle.remove()
        report['status']='pass'
        report['artifacts_sha256']={name:sha256(root/name) for name in ['features.pt','tensors.pt','prediction_unrelaxed.pdb']}
    except Exception as exc:
        report.update(status='fail',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
        traceback.print_exc()
    write_json(root/'report.json',report)
    print(report['status'].upper(),root,flush=True)
    return 0 if report['status']=='pass' else 1


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights',required=True);p.add_argument('--fasta',required=True)
    p.add_argument('--alignments',required=True);p.add_argument('--features')
    p.add_argument('--device',default='cuda:0');p.add_argument('--output',required=True)
    p.add_argument('--chunk-size',type=int,default=64);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--attention-kernel',choices=['upstream','torch'],default='upstream')
    p.add_argument('--cpu-threads',type=int,default=8);p.add_argument('--warmup',type=int,default=1)
    p.add_argument('--runs',type=int,default=5);p.add_argument('--profile',action='store_true')
    p.add_argument('--numerical-only',action='store_true',help='One complete prediction for correctness; do not collect warmed timings')
    a=p.parse_args()
    if a.chunk_size<1 or a.runs<1 or a.warmup<0 or a.cpu_threads<1:p.error('Invalid timing or chunk parameters')
    sys.exit(main(a))
