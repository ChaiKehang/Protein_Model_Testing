"""Portable CUDA OpenFold Systems capture plus representative Compute launches.

Requires a passed reference from this hardware. Counter-policy restrictions stay
explicit; no host-policy mutation or CPU/MUSA substitution is performed.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from common.reporting import write_json,sha256
from common.profiler_permissions import counter_allowed,counter_policy
from scripts.run_nsight import command_run,find_tool
from scripts.nsight_analysis import read_systems,compute_targets,read_compute_csv,verify_compute_launch


def run(a):
    root=Path(a.output).resolve();root.mkdir(parents=True,exist_ok=False)
    reference=Path(a.reference).resolve()
    r=json.loads((reference/'report.json').read_text())
    if r['status']!='pass' or not a.device.startswith('cuda'):raise ValueError('Passed CUDA reference required')
    names=['scripts/run_openfold_nsight.py','scripts/openfold_nsight_workload.py','scripts/nsight_analysis.py',
           'scripts/run_nsight.py','models/openfold_adapter.py','common/backend.py','common/reporting.py','common/profiler_permissions.py']
    sources={}
    for name in names:
        p=root/'source'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((REPO/name).read_bytes())
        sources[name]=sha256(REPO/name)
    summary={'status':'partial','source_sha256':sources,'reference_report_sha256':sha256(reference/'report.json'),
             'counter_policy':counter_policy(),'compute':[],
             'scope':a.scope+' OpenFold native diagnostic; no formal timing claim'}
    env=dict(os.environ,HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    nsys=find_tool(a.nsys,'nsys',['opt/nvidia/nsight-compute/*/host/target-linux-x64/nsys','opt/nvidia/nsight-systems/*/target-linux-x64/nsys'])
    try:ncu=find_tool(a.ncu,'ncu',['usr/local/cuda-*/bin/ncu'])
    except FileNotFoundError:ncu=None
    for label,path in [('nsys',nsys),('ncu',ncu)]:
        if path:
            result=command_run([path,'--version'],root,label+'_version',env)
            summary.setdefault('tools',{})[label]={'path':path,**result}
    def workload(out):
        return [sys.executable,str(REPO/'scripts/openfold_nsight_workload.py'),'--reference',str(reference),
                '--output',str(out),'--device',a.device,'--warmup',str(a.warmup),'--scope',a.scope]
    base=root/('openfold_'+a.scope)
    command=[nsys,'profile','--trace=cuda,nvtx,osrt','--sample=none','--cpuctxsw=none',
             '--capture-range=cudaProfilerApi','--capture-range-end=stop','--force-overwrite=false','--output='+str(base)]
    capture=command_run(command+workload(root/'workload'),root,'capture',env,timeout=1200)
    if capture['returncode']:raise RuntimeError('Native capture failed; preserved log at '+str(root/capture['log']))
    native=base.with_suffix('.nsys-rep');sqlite=base.with_suffix('.sqlite')
    exported=command_run([nsys,'export','--type=sqlite','--force-overwrite=false','--output='+str(sqlite),str(native)],root,'export',env,timeout=300)
    if exported['returncode']:raise RuntimeError('Native export failed; preserved raw report')
    analysis=read_systems(sqlite,capture_name='openfold_capture')
    selection=compute_targets(analysis,top_k=a.top_k)
    systems={'status':'pass','capture':capture,'export':exported,'native_report':str(native),'native_sha256':sha256(native),
             'sqlite_sha256':sha256(sqlite),'analysis':analysis,'compute_targets':selection,'analysis_source_sha256':sources['scripts/nsight_analysis.py']}
    write_json(root/'systems_summary.json',systems)
    denied=False
    for i,target in enumerate(selection):
        item={'selection':target,'status':'pending','collection_attempted':False}
        if not ncu:item.update(status='missing_ncu')
        elif summary['counter_policy']['collection_allowed'] is not True or denied:
            item.update(status='blocked_permission' if summary['counter_policy']['collection_allowed'] is False or denied else 'policy_unverified',
                        selected_metrics={k:None for k in ['compute_percent','memory_percent','dram_percent','occupancy_percent','dram_read_bytes','dram_write_bytes']})
        elif target['status']!='planned':item.update(status='no_representative_launch')
        else:
            stem=f'kernel_{i+1}';item['collection_attempted']=True
            cmd=[ncu,'--nvtx','--nvtx-include',target['nvtx_operation_range']+'/',
                 '--kernel-name-base','demangled','--kernel-name',target['name'],'--rename-kernels','false',
                 '--launch-skip',str(target['filtered_launch_skip']),'--launch-count','1','--clock-control','none',
                 '--cache-control','none','--export',str(root/stem)]
            for section in ['SpeedOfLight','Occupancy','LaunchStats','MemoryWorkloadAnalysis','ComputeWorkloadAnalysis']:
                cmd+=['--section',section]
            result=command_run(cmd+workload(root/(stem+'_workload')),root,stem+'_capture',env,timeout=1200)
            item['capture']=result
            log=(root/result['log']).read_text()
            if 'ERR_NVGPUCTRPERM' in log:
                denied=True;item.update(status='blocked_permission',evidence=str(root/result['log']))
            elif result['returncode'] or not (root/(stem+'.ncu-rep')).exists():item.update(status='failed',reason='No successful native Compute report')
            else:
                report=root/(stem+'.ncu-rep')
                export=command_run([ncu,'--import',str(report),'--page','raw','--csv','--units','base'],root,stem+'_export',env)
                csv=root/(stem+'.csv');shutil.copy2(root/export['log'],csv)
                try:
                    metrics=read_compute_csv(csv);verification=verify_compute_launch(metrics['metrics'],target)
                    item.update(metrics,status='pass' if verification['status']=='pass' else 'launch_unverified',
                                launch_verification=verification,native_report=str(report),native_sha256=sha256(report))
                except Exception as exc:item.update(status='failed',reason=str(exc))
        summary['compute'].append(item)
        write_json(root/'nsight_summary.json',summary)
    summary['status']='complete' if summary['compute'] and all(x['status']=='pass' for x in summary['compute']) else 'partial'
    summary['systems']=str(root/'systems_summary.json')
    write_json(root/'nsight_summary.json',summary)
    print(summary['status'].upper(),root,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference',required=True)
    p.add_argument('--output',required=True);p.add_argument('--device',default='cuda:0')
    p.add_argument('--nsys');p.add_argument('--ncu');p.add_argument('--top-k',type=int,default=3)
    p.add_argument('--scope',choices=['warm','pipeline'],default='warm')
    p.add_argument('--warmup',type=int,default=1);a=p.parse_args()
    if a.top_k<1 or a.warmup<1:p.error('Positive top-k and warmup required')
    run(a)
