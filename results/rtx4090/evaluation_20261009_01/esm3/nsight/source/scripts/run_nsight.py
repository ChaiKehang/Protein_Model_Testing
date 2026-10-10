"""Serial native Nsight captures; append-only directory, explicit partial/blocked outcomes."""
import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import signal
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from common.reporting import write_json, sha256
from scripts.nsight_analysis import read_systems, read_compute_csv, compute_targets, verify_compute_launch


def find_tool(explicit,name,patterns):
    if explicit:
        p=Path(explicit).resolve()
        if not p.is_file():raise FileNotFoundError(p)
        return str(p)
    found=shutil.which(name)
    if found:return found
    for pattern in patterns:
        found=sorted(Path('/').glob(pattern))
        if found:return str(found[-1])
    raise FileNotFoundError(f'{name} not installed; use --{name} to specify executable')


def command_run(command,root,log,env,timeout=300):
    stamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
    original=log; attempt=2
    while (root/f'{log}.log').exists():
        log=f'{original}_attempt{attempt:02d}';attempt+=1
    write_json(root/f'{log}.command.json',{'argv':command,'cwd':str(REPO),'timestamp_utc':stamp,'timeout_seconds':timeout,
                                        'env_overrides':{'HF_HUB_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false'}})
    start=time.monotonic()
    try:
        with (root/f'{log}.log').open('w') as stream:
            process=subprocess.Popen(command,cwd=REPO,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL)
                process.wait();code=124
    except OSError as exc:
        with (root/f'{log}.log').open('a') as stream:stream.write(str(exc)+'\n')
        code=127
    return {'returncode':code,'elapsed_seconds':time.monotonic()-start,'log':f'{log}.log','command':f'{log}.command.json'}


def workload_command(args,case,output,tool,calls):
    if case.get('scope')=='pipeline':calls=1
    cmd=[sys.executable,str(REPO/'scripts/nsight_workload.py'),'--sample-id',case['sample_id'],'--manifest',str(Path(args.manifest).resolve()),
         '--reference',str(Path(args.suite).resolve()/case['reference']),'--tool',tool,'--calls',str(calls),'--warmup',str(args.config_data['warmup']),
         '--output',str(output)]
    for key in ['device','precision','task','scope','attention','batch_size','pdb','inputs']:
        if key in case:cmd.extend(['--'+key.replace('_','-'),str(case[key])])
    if case.get('sample_ids'):cmd.extend(['--sample-ids',*case['sample_ids']])
    return cmd


def run(args):
    from common.profiler_permissions import counter_policy
    suite=Path(args.suite).resolve();root=Path(args.output).resolve()
    if root.exists() and not args.resume:raise FileExistsError('Output exists; use new directory or --resume')
    root.mkdir(parents=True,exist_ok=args.resume)
    config=json.loads(Path(args.config).read_text());args.config_data=config
    nsys=find_tool(args.nsys,'nsys',['opt/nvidia/nsight-compute/*/host/target-linux-x64/nsys','opt/nvidia/nsight-systems/*/target-linux-x64/nsys'])
    try:ncu=find_tool(args.ncu,'ncu',['usr/local/cuda-*/bin/ncu'])
    except FileNotFoundError:ncu=None
    env=dict(os.environ,HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    summary_path=root/'nsight_summary.json'
    previous=json.loads(summary_path.read_text()) if args.resume and summary_path.exists() else {}
    summary={'schema_version':1,'root':str(root),'parent_suite':str(suite),'config':config,'systems':{},'compute':[],
             'tools':{},'parent_hashes':{n:sha256(suite/n) for n in ['summary.json','audit_summary.json','quality_extension.json','metrics.csv','panel_summary.json'] if (suite/n).is_file()},
             'source_sha256':{n:sha256(REPO/n) for n in ['scripts/nsight_workload.py','scripts/nsight_analysis.py','scripts/run_nsight.py','common/profiler_permissions.py',
                     'models/esm3_adapter.py','common/backend.py','common/reporting.py','main.py','compare_outputs.py']},
             'config_sha256':sha256(args.config),
             'limitations':['CPU sampling/context-switch tracing disabled because container perf_event access is unavailable.',
                            'Instrumented profile durations are not official inference benchmarks.',
                            'Nsight GPU activity union is not SM occupancy.']}
    if previous and (previous['config']!=config or previous['parent_hashes']!=summary['parent_hashes']
                     or previous.get('source_sha256')!=summary['source_sha256']):
        raise ValueError('Resume config, parent data or capture source changed; use a new output directory')
    if not previous:
        for name in summary['source_sha256']:
            path=root/'source'/name;path.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(REPO/name,path)
    summary['unit_tests']=command_run([sys.executable,'-m','unittest','discover','-s','tests','-v'],root,'unit_tests',env)
    if summary['unit_tests']['returncode']:
        write_json(summary_path,summary)
        raise RuntimeError('Logic tests failed; do not profile until checks pass')
    for tool,path in [('nsys',nsys),('ncu',ncu)]:
        if not path:
            summary['tools'][tool]={'status':'missing'};continue
        result=subprocess.run([path,'--version'],capture_output=True,text=True)
        summary['tools'][tool]={'path':path,'version':result.stdout.strip(),'returncode':result.returncode}
    summary['environment_check']=command_run([nsys,'status','--environment'],root,'environment_check',env)
    params=Path('/proc/driver/nvidia/params')
    status=Path('/proc/self/status')
    profiling_policy=[line for line in params.read_text().splitlines() if 'ProfilingAdminOnly' in line] if params.exists() else []
    cap_line=next((line for line in status.read_text().splitlines() if line.startswith('CapEff:')),None) if status.exists() else None
    permission_environment={'driver_profiling_policy':profiling_policy,'effective_uid':os.geteuid(),
                            'cap_sys_admin':bool(int(cap_line.split()[1],16)&(1<<21)) if cap_line else None}
    write_json(root/'permission_environment.json',permission_environment)
    summary['permission_environment']=permission_environment
    summary['counter_policy']=counter_policy()
    write_json(root/'config.json',config)
    # Preserve a previously failed probe as evidence; do not repeat identical denied requests.
    probe=root/'L256_original_gemm_capture.log'
    block=previous.get('compute_permission_block')
    if not block and summary['counter_policy']['collection_allowed'] is not True:
        block={'code':'COUNTER_POLICY_RESTRICTED' if summary['counter_policy']['collection_allowed'] is False else 'COUNTER_POLICY_UNVERIFIED',
               'evidence':'permission_environment.json','message':'Read-only capability precheck; no counter collection attempted.'}
    if not block and probe.exists() and 'ERR_NVGPUCTRPERM' in probe.read_text():
        block={'code':'ERR_NVGPUCTRPERM','evidence':probe.name,'message':'GPU counters denied by host/container policy; host settings unchanged.'}
    if block:summary['compute_permission_block']=block
    for case in config['cases']:
        name=case['name'];print(f'NSYS {name}',flush=True)
        report=root/f'{name}.nsys-rep';sqlite=root/f'{name}.sqlite';output=root/f'{name}_workload'
        record={'config':case,'status':'running','native_report':report.name,'sqlite':sqlite.name,'workload':f'{output.name}/workload.json'}
        if report.exists() and args.resume:
            if not (output/'workload.json').is_file():raise RuntimeError(f'Orphan report {report}')
            record['capture']={'resumed_existing':True,'log':f'{name}_capture.log'}
        else:
            cmd=[nsys,'profile','--trace=cuda,nvtx,osrt','--sample=none','--cpuctxsw=none','--capture-range=cudaProfilerApi',
                 '--capture-range-end=stop','--force-overwrite=false',f'--output={root/name}',*workload_command(args,case,output,'nsys',config['systems_calls'])]
            record['capture']=command_run(cmd,root,f'{name}_capture',env)
        if not report.is_file():
            record.update(status='failed',reason='native report missing');summary['systems'][name]=record;write_json(summary_path,summary);continue
        workload=json.loads((output/'workload.json').read_text())
        record['full_tensor_exact']=workload.get('full_tensor_exact',{})
        if workload['status']!='pass':
            record.update(status='failed',reason='workload validation failed');summary['systems'][name]=record;write_json(summary_path,summary);continue
        if not sqlite.exists():
            record['stats_command']=command_run([nsys,'stats','--report','cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum','--format','csv',
                 '--output',str(root/f'{name}_stats'),str(report)],root,f'{name}_stats',env)
        try:
            record['analysis']=read_systems(sqlite);record['status']='pass'
        except Exception as exc:record.update(status='failed',reason=str(exc))
        record['sha256']=sha256(report)
        record['csv_reports']=[p.name for p in sorted(root.glob(f'{name}_stats_*.csv'))]
        summary['systems'][name]=record;write_json(summary_path,summary)
    for case in config['cases']:
        name=case['name'];system=summary['systems'].get(name,{})
        if system.get('status')!='pass':continue
        targets=compute_targets(system['analysis'],case['compute_top_k'])
        for i,selection in enumerate(targets,1):
            kernel=next(k for k in system['analysis']['kernels'] if k['name']==selection['name'])
            stem=f'{name}_kernel{i:02d}'
            target={'case':name,'name':kernel['name'],'systems_kernel_sum_percent':kernel['kernel_sum_percent'],'systems_mean_us':kernel['mean_us'],
                    'selection':selection,'status':'planned'}
            if selection['status']!='planned' or not ncu:
                target.update(status='blocked_missing_tool_or_launch',collection_attempted=False)
                summary['compute'].append(target);continue
            prior=next((c for c in previous.get('compute',[]) if c['case']==name and c['name']==kernel['name'] and c['status']=='pass'),None)
            if prior:summary['compute'].append(prior);continue
            if block:
                target.update(status='blocked_permission',reason=block['code'],evidence=block['evidence'],
                              collection_attempted=False,
                              selected_metrics={k:None for k in ['compute_percent','memory_percent','dram_percent','occupancy_percent']})
                summary['compute'].append(target);write_json(summary_path,summary);continue
            report=root/f'{stem}.ncu-rep';csv_path=root/f'{stem}.csv'
            target['collection_attempted']=True
            print(f'NCU {stem}',flush=True)
            if not report.exists():
                cmd=[ncu,'--nvtx','--nvtx-include',selection['nvtx_operation_range']+'/',
                     '--kernel-name-base','demangled','--kernel-name',kernel['name'],
                     '--rename-kernels','false','--launch-skip',str(selection['filtered_launch_skip']),
                     '--launch-count',str(config['compute_launches']),'--clock-control','none','--cache-control','none','--export',str(root/stem)]
                for section in config['compute_sections']:cmd.extend(['--section',section])
                cmd.extend(workload_command(args,case,root/f'{stem}_workload','ncu',1))
                target['capture']=command_run(cmd,root,f'{stem}_capture',env)
                text=(root/target['capture']['log']).read_text()
                if 'ERR_NVGPUCTRPERM' in text:
                    block={'code':'ERR_NVGPUCTRPERM','evidence':target['capture']['log'],'message':'Host counter permission denied.'}
                    summary['compute_permission_block']=block
                    target.update(status='blocked_permission',reason=block['code'],evidence=block['evidence'])
                    summary['compute'].append(target);write_json(summary_path,summary);continue
            if report.exists():
                if not csv_path.exists():
                    exported=command_run([ncu,'--import',str(report),'--page','raw','--csv','--units','base'],root,f'{stem}_export',env)
                    shutil.copy2(root/exported['log'],csv_path)
                try:
                    target.update(read_compute_csv(csv_path))
                    verification=verify_compute_launch(target['metrics'],selection)
                    target.update(status='pass' if verification['status']=='pass' else 'launch_unverified',
                                  launch_verification=verification,native_report=report.name,csv=csv_path.name)
                except Exception as exc:target.update(status='failed',reason=str(exc))
            else:target.update(status='failed',reason='No native Compute report; see capture log')
            summary['compute'].append(target);write_json(summary_path,summary)
    summary['status']='complete' if bool(summary['systems']) and bool(summary['compute']) and all(s['status']=='pass' for s in summary['systems'].values()) and all(c['status']=='pass' for c in summary['compute']) else 'partial'
    summary['completed_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    after={n:sha256(suite/n) for n in summary['parent_hashes']}
    summary['parent_data_unchanged']=after==summary['parent_hashes']
    if not summary['parent_data_unchanged']:raise RuntimeError('Parent data changed during profiling')
    write_json(summary_path,summary)
    write_json(suite/'nsight_latest.json',{'summary':str(summary_path.relative_to(suite)),'sha256':sha256(summary_path)})
    print(f"Nsight {summary['status']}: {summary_path}",flush=True)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite',required=True);p.add_argument('--output',required=True)
    p.add_argument('--config',default=str(REPO/'configs/nsight_reference.json'))
    p.add_argument('--manifest',default=str(REPO.parent/'esm-baseline/model_snapshot.json'))
    p.add_argument('--nsys');p.add_argument('--ncu');p.add_argument('--resume',action='store_true')
    run(p.parse_args())
