"""Standard ProteinGym target sequences: reference forward and mixed-batch evaluation."""
import argparse
import json
import os
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
import torch
from common.reporting import write_json,sha256
from common.telemetry import monitoring
from scripts.run_nsight import command_run
from compare_outputs import compare_payloads


def run(a):
    root=Path(a.output).resolve();root.mkdir(parents=True,exist_ok=False)
    dataset=Path(a.dataset).resolve();inputs=dataset/'performance_panel.jsonl'
    samples=[json.loads(l) for l in inputs.read_text().splitlines()]
    summary={'status':'running','dataset':str(dataset),'dataset_manifest_sha256':sha256(dataset/'manifest.json'),
             'input_sha256':sha256(inputs),'parameters':vars(a),'cases':{},'batch_consistency':{},
             'scope':'Original model, ten bounded standard targets plus four distinct mixed-batch targets; no population claim',
             'batch_tolerance':{'atol':.1,'rtol':.02,'note':'Prespecified BF16 numerical diagnostic, not final migration gate'}}
    env=dict(os.environ,HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    def case(name,ids):
        out=root/name
        cmd=[sys.executable,str(REPO/'main.py'),'--task','forward','--inputs',str(inputs),'--sample-ids',*ids,
             '--batch-size',str(len(ids)),'--precision','bf16','--device',a.device,'--warmup','3','--runs',str(a.runs),
             '--end-to-end-runs','2','--cpu-threads','4','--manifest',a.weights,'--output',str(out)]
        execution=command_run(cmd,root,name,env,timeout=600)
        report=json.loads((out/'report.json').read_text()) if (out/'report.json').exists() else {'status':'failed'}
        summary['cases'][name]={'report':report,'report_path':str(out/'report.json'),'execution':execution}
        write_json(root/'panel_summary.json',summary);print(name,report['status'],flush=True)
    with monitoring(a.device,root/'environment'):
        for sample in samples:case(sample['id'],[sample['id']])
        ids=[samples[i]['id'] for i in [0,3,4,7]];case('mixed_B4',ids)
    if summary['cases']['mixed_B4']['report']['status']=='pass':
        batched=torch.load(root/'mixed_B4/tensors.pt',map_location='cpu',weights_only=True)
        for i,id in enumerate(ids):
            if summary['cases'][id]['report']['status']!='pass':continue
            single=torch.load(root/id/'tensors.pt',map_location='cpu',weights_only=True);length=single['sequence_ids'].shape[-1]
            row={k:batched[k][i:i+1,:length] for k in ['sequence_ids','valid_mask','logits','embeddings']}
            summary['batch_consistency'][id]=compare_payloads(single,row,atol=.1,rtol=.02,keys=('logits','embeddings'))
    summary['status']='pass' if all(c['report']['status']=='pass' for c in summary['cases'].values()) and len(summary['batch_consistency'])==4 and all(c['status']=='pass' for c in summary['batch_consistency'].values()) else 'partial'
    write_json(root/'panel_summary.json',summary);print(summary['status'].upper(),root,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--runs',type=int,default=20)
    p.add_argument('--weights',default='/root/autodl-tmp/esm-baseline/model_snapshot.json')
    a=p.parse_args()
    if a.runs<3:p.error('At least three formal samples required')
    run(a)
