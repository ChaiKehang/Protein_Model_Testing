"""Append-only follow-up: backend preflight, matched benchmarks, native CUDA traces.

Uses main.py for every measurement. A CUDA capture is never substituted for MUSA.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import subprocess
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from common.backend import Backend
from common.reporting import write_json, sha256, environment
from common.telemetry import monitoring
from scripts.run_nsight import command_run, find_tool
from scripts.nsight_analysis import read_systems, compute_targets
from compare_outputs import compare_payloads
import torch


def preflight(device):
    value = {'requested_device': device, 'status': 'unavailable'}
    try:
        backend = Backend(device)
        with backend.context():
            tensor = torch.arange(16, dtype=torch.float32).reshape(4, 4).to(backend.device)
            result = (tensor @ tensor.T).cpu()
            backend.synchronize()
            expected = torch.arange(16, dtype=torch.float32).reshape(4, 4)
            if not torch.equal(result, expected @ expected.T):
                raise RuntimeError('Backend FP32 matmul smoke check failed')
            events = backend.events()
            event_ms = None
            if events:
                events[0].record(); _ = tensor @ tensor.T; events[1].record()
                backend.synchronize(); event_ms = events[0].elapsed_time(events[1])
        value.update(status='pass', actual_device=str(tensor.device), device=backend.metadata(),
                     event_ms=event_ms, memory=backend.memory(),
                     scope='small backend smoke test; does not establish ESM3/MUSA operator compatibility')
    except Exception as exc:
        value['error'] = f'{type(exc).__name__}: {exc}'
    return value


def inventory(profile_dirs):
    families = {'GEMM': ['aten::mm','aten::addmm','aten::linear'], 'BatchGEMM': ['aten::bmm'],
                'Attention': ['scaled_dot_product', 'flash_attention'], 'Softmax': ['softmax'],
                'LayerNorm/RMSNorm': ['layer_norm','rms_norm'], 'Embedding': ['embedding'],
                'Convolution': ['convolution','conv2d'], 'Reduction': ['aten::sum','aten::mean','aten::norm','linalg_vector_norm'],
                'Gather/Scatter': ['gather','scatter','index_select'], 'TopK': ['topk'],
                'Pairwise': [], 'TriangleAttention': [], 'TriangleMultiplication': [],
                '3DGeometry': [], 'Diffusion': [], 'Graph': [], 'Protein-specific': []}
    evidence = {}
    for directory in profile_dirs:
        path = directory / 'operators.json'
        if path.is_file():
            data = json.loads(path.read_text())
            evidence[directory.name] = {'path': str(path), 'sha256': sha256(path),
                                         'names': sorted({r['name'] for r in data['operators'] if r['is_aten_operator']})}
    result = []
    for family, needles in families.items():
        found = {case: [n for n in value['names'] if any(p in n for p in needles)] for case, value in evidence.items()}
        found = {case: names for case, names in found.items() if names}
        result.append({'family': family, 'status': 'observed_ATen' if found else 'not_established_by_ATen_name',
                       'evidence': found, 'note': 'Not observed is not unsupported. Composite protein modules require source/module evidence; OpenFold inventory pending.'})
    geometry = {}
    for directory in profile_dirs:
        path=directory/'operator_shapes.json'
        if path.is_file():
            modules=[s['module'] for s in json.loads(path.read_text()) if '.geom_attn.' in s['module']]
            if modules:geometry[directory.name]=modules
    entry=next(r for r in result if r['family']=='3DGeometry')
    if geometry:
        entry.update(status='observed_model_submodules',evidence=geometry,
                     note='Actual geometry submodule hooks; original model path. Not a MUSA kernel validation.')
    return {'families': result, 'profile_evidence': evidence, 'scope': 'covered profiled cases only; no claimed MUSA kernel execution'}


def main(args):
    root=Path(args.output).resolve();root.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    probe=subprocess.run([sys.executable,__file__,'--preflight-device',args.device],cwd=REPO,env=env,capture_output=True,text=True)
    checks=json.loads(probe.stdout.strip().splitlines()[-1])
    summary={'status':'running','scope':'Reference evaluation only; upstream geometry retained','preflight':checks,'cases':{},'comparisons':{}}
    write_json(root/'audit_summary.json',summary)
    if checks['status']!='pass':raise RuntimeError('Requested backend unavailable; no fallback')
    jobs=[{'name':f'L{l}','task':'forward','sample':f'random_L{l}','profile':l==256} for l in [256,1024,2048]]
    jobs+=[{'name':'mixed_B2','task':'forward','sample':'ubiquitin_human','ids':['insulin_a_chain_human','ubiquitin_human'],'batch':2,'profile':True}]
    jobs+=[{'name':t,'task':t,'sample':'ubiquitin_human','profile':True} for t in ['sequence','structure','inverse']]
    jobs+=[{'name':'ubiquitin_fp32','task':'forward','sample':'ubiquitin_human','precision':'fp32'},
           {'name':'ubiquitin_cpu_fp32','task':'forward','sample':'ubiquitin_human','precision':'fp32','device':'cpu','runs':3}]
    jobs+=[{'name':'crambin_'+t,'task':t,'sample':'crambin_1CRN_A','inputs':str(REPO/'inputs/task_quality_sequences.jsonl'),
            'pdb':args.crambin_pdb} for t in ['forward','sequence','structure','inverse']]
    write_json(root/'jobs.json',jobs)
    with monitoring(args.device,root/'environment'):
        for job in jobs:
            out=root/job['name']
            cmd=[sys.executable,str(REPO/'main.py'),'--task',job['task'],'--sample-id',job['sample'],
                 '--device',job.get('device',args.device),'--precision',job.get('precision',args.precision),
                 '--runs',str(job.get('runs',args.runs)),'--warmup','3','--end-to-end-runs','2',
                 '--manifest',args.manifest,'--pdb',job.get('pdb',args.pdb),'--output',str(out),'--batch-size',str(job.get('batch',1))]
            if job.get('inputs'):cmd+=['--inputs',job['inputs']]
            if job.get('ids'):cmd+=['--sample-ids',*job['ids']]
            if job.get('profile') and args.device.startswith('cuda'):cmd+=['--profile']
            result=command_run(cmd,root,job['name'],env,timeout=1200)
            report=json.loads((out/'report.json').read_text()) if (out/'report.json').exists() else {'status':'failed'}
            summary['cases'][job['name']]={'execution':result,'report':report,'report_path':str(out/'report.json')}
            write_json(root/'audit_summary.json',summary);print(job['name'],report['status'],flush=True)
    if all(summary['cases'][n]['report']['status']=='pass' for n in ['ubiquitin_fp32','ubiquitin_cpu_fp32']):
        summary['comparisons']['cpu_cuda_fp32']=compare_payloads(*[torch.load(root/n/'tensors.pt',map_location='cpu',weights_only=True)
             for n in ['ubiquitin_cpu_fp32','ubiquitin_fp32']],atol=.001,rtol=.001,keys=('logits','embeddings'))
    write_json(root/'operator_inventory.json',inventory([root/j['name'] for j in jobs]))
    summary['status']='pass' if all(v['report']['status']=='pass' for v in summary['cases'].values()) else 'partial'
    write_json(root/'audit_summary.json',summary)
    if args.device.startswith('cuda') and not args.no_nsight:
        captures=[('L256','warm'),('L1024','warm'),('mixed_B2','warm'),('sequence','warm'),('structure','warm'),('inverse','warm'),
                  ('L256','pipeline'),('sequence','pipeline'),('structure','pipeline'),('inverse','pipeline')]
        cases=[]
        for name,scope in captures:
            job=next(j for j in jobs if j['name']==name)
            if summary['cases'][name]['report']['status']!='pass':continue
            case={'name':name+'_'+scope,'task':job['task'],'sample_id':job['sample'],'scope':scope,'device':args.device,
                  'precision':args.precision,'reference':name+'/tensors.pt','compute_top_k':3,'batch_size':job.get('batch',1),'pdb':args.pdb}
            if job.get('ids'):case['sample_ids']=job['ids']
            cases.append(case)
        config={'warmup':3,'systems_calls':3,'compute_launches':1,
                'compute_sections':['SpeedOfLight','Occupancy','LaunchStats','MemoryWorkloadAnalysis','ComputeWorkloadAnalysis'],'cases':cases}
        write_json(root/'nsight_config.json',config)
        cmd=[sys.executable,str(REPO/'scripts/run_nsight.py'),'--suite',str(root),'--output',str(root/'nsight'),
             '--config',str(root/'nsight_config.json'),'--manifest',args.manifest]
        summary['nsight_execution']=command_run(cmd,root,'nsight',env,timeout=1800)
    write_json(root/'audit_summary.json',summary);print(summary['status'].upper(),root,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output');p.add_argument('--device',default='cuda:0')
    p.add_argument('--preflight-device',help=argparse.SUPPRESS);p.add_argument('--precision',choices=['bf16','fp16','fp32'],default='bf16')
    p.add_argument('--runs',type=int,default=20);p.add_argument('--manifest',default=str(REPO.parent/'esm-baseline/model_snapshot.json'))
    p.add_argument('--pdb',default=str(REPO.parent/'esm-baseline/data/1UBQ.pdb'));p.add_argument('--no-nsight',action='store_true')
    p.add_argument('--crambin-pdb',default=str(REPO/'inputs/structures/1CRN.pdb'))
    a=p.parse_args()
    if a.preflight_device:print(json.dumps(preflight(a.preflight_device),ensure_ascii=False));raise SystemExit(0)
    if not a.output or a.runs<1:p.error('New output and positive runs required')
    main(a)
