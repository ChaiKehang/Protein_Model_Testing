"""Serial two-environment reference evaluation; no model optimization or GPU overlap."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
REPO=Path(__file__).resolve().parents[1]


def main(a):
    root=Path(a.output).resolve();root.mkdir(parents=True,exist_ok=a.resume)
    state_path=root/'execution.json'
    state=json.loads(state_path.read_text()) if state_path.exists() and a.resume else {'status':'running','stages':{}}
    state['status']='running'
    esm=a.esm_python;of=a.openfold_python;dataset=str(Path(a.dataset).resolve())
    standard_config=root/'standard_nsight_config.json'
    cfg=json.loads((REPO/'configs/nsight_proteingym.json').read_text())
    for case in cfg['cases']:
        case.update(device=a.device,inputs=dataset+'/performance_panel.jsonl')
    standard_config.write_text(json.dumps(cfg,indent=2)+'\n')
    common=[of,str(REPO/'scripts/openfold_baseline.py'),'--weights',a.openfold_weights,
            '--fasta',a.openfold_fasta,'--alignments',a.openfold_alignments,'--device',a.device]
    commands=[('esm3',[esm,'scripts/portable_audit.py','--output',str(root/'esm3'),'--device',a.device,
                     '--manifest',a.esm_manifest,'--pdb',a.esm_pdb,'--crambin-pdb',a.crambin_pdb]),
        ('standard',[esm,'scripts/run_proteingym_panel.py','--dataset',dataset,'--output',str(root/'esm3_standard'),'--device',a.device,'--weights',a.esm_manifest]),
        ('fitness',[esm,'scripts/proteingym_dms.py','--dataset',dataset,'--output',str(root/'esm3_fitness'),'--device',a.device,'--weights',a.esm_manifest]),
        ('batch',[esm,'scripts/check_proteingym_batch.py','--inputs',dataset+'/performance_panel.jsonl','--sample-ids',
                  'pg_AMFR_HUMAN','pg_RPC1_LAMBD','pg_ADRB2_HUMAN','pg_A0A192B1T2_9HIV1','--output',str(root/'esm3_batch_fp32'),'--device',a.device,'--weights',a.esm_manifest]),
        ('standard_nsight',[esm,'scripts/run_nsight.py','--suite',str(root/'esm3_standard'),
                   '--config',str(standard_config),'--output',str(root/'esm3_standard/nsight'),'--manifest',a.esm_manifest]),
        ('openfold',common+['--output',str(root/'openfold'),'--profile']),
        ('openfold_quality',[of,'scripts/openfold_structure_quality.py','--run',str(root/'openfold'),
                   '--reference',str(REPO/'inputs/openfold/6KWC.cif'),'--chain','A']),
        ('openfold_nsight',[of,'scripts/run_openfold_nsight.py','--reference',str(root/'openfold'),'--device',a.device,'--output',str(root/'openfold_nsight')]),
        ('openfold_pipeline',[of,'scripts/run_openfold_nsight.py','--reference',str(root/'openfold'),'--device',a.device,
                              '--scope','pipeline','--output',str(root/'openfold_pipeline')])]
    if a.cpu_reference:
        commands.insert(7,('openfold_cpu_comparison',[of,'scripts/compare_openfold.py','--reference',a.cpu_reference,
                           '--candidate',str(root/'openfold'),'--output',str(root/'openfold_cpu_comparison.json')]))
    commands.insert(5,('operator_diagnostics',[esm,'scripts/portable_operator_bench.py',
                 '--shapes',str(root/'esm3/L256/operator_shapes.json'),'--device',a.device,
                 '--output',str(root/'operator_diagnostics')]))
    if a.wait_existing_esm:
        print('Waiting for existing ESM3 collector to finish, before any GPU launch',flush=True)
        deadline=time.monotonic()+1800
        while True:
            p=root/'esm3/audit_summary.json'
            s=json.loads(p.read_text()) if p.exists() else {}
            if 'nsight_execution' in s:break
            if time.monotonic()>deadline:raise RuntimeError('Existing ESM3 evaluation did not finish')
            time.sleep(2)
        commands=commands[1:];state['stages']['esm3']={'adopted_existing':True,'returncode':s['nsight_execution']['returncode']}
    env=dict(os.environ,HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    for name,cmd in commands:
        if name in state['stages'] and state['stages'][name]['returncode']==0:
            prior=state['stages'][name]
            if not prior.get('adopted_existing') and prior.get('argv')!=cmd:
                raise ValueError('Resume command differs; use a new output directory: '+name)
            continue
        print('START',name,flush=True);started=time.time()
        with (root/(name+'.log')).open('a') as log:
            result=subprocess.run(cmd,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
        state['stages'][name]={'returncode':result.returncode,'argv':cmd,'elapsed_seconds':time.time()-started}
        state_path.write_text(json.dumps(state,indent=2)+'\n');print('FINISH',name,result.returncode,flush=True)
        if result.returncode:raise RuntimeError(name+' failed; inspect preserved log, do not substitute a smaller workload')
    state['status']='complete';state_path.write_text(json.dumps(state,indent=2)+'\n');print('MEASUREMENTS_COMPLETE',root,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--resume',action='store_true')
    p.add_argument('--wait-existing-esm',action='store_true',help=argparse.SUPPRESS);p.add_argument('--device',default='cuda:0')
    p.add_argument('--esm-python',default='/root/autodl-tmp/envs/esm3/bin/python');p.add_argument('--openfold-python',default='/root/autodl-tmp/envs/openfold-cuda/bin/python')
    p.add_argument('--dataset',default=str(REPO/'inputs/proteingym_v1.3_20261009'))
    p.add_argument('--esm-manifest',default=str(REPO/'model_snapshot.json'))
    p.add_argument('--esm-pdb',default=str(REPO/'inputs/structures/1UBQ.pdb'))
    p.add_argument('--crambin-pdb',default=str(REPO/'inputs/structures/1CRN.pdb'))
    p.add_argument('--openfold-weights',default='/root/autodl-tmp/openfold-cache/params/finetuning_no_templ_ptm_1.pt')
    p.add_argument('--openfold-fasta',default='/root/autodl-tmp/references/openfold/examples/monomer/fasta_dir/6kwc.fasta')
    p.add_argument('--openfold-alignments',default='/root/autodl-tmp/references/openfold/examples/monomer/alignments/6KWC_1')
    p.add_argument('--cpu-reference');main(p.parse_args())
