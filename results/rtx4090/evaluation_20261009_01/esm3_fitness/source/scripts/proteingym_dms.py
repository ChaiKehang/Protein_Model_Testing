"""Three complete ProteinGym assays, scored by sequence-only ESM3 masked marginals.

Scores use cached WT-context predictions and additive log odds for multi mutants.
No label training, cropping, or selection by fitness. Not official leaderboard replication.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
os.environ.setdefault('HF_HUB_OFFLINE','1')
import torch
from esm.sdk.api import ESMProtein,LogitsConfig
from esm.utils.sampling import _BatchedESMProteinTensor
from common.backend import Backend
from common.reporting import environment,write_json,sha256,stats
from common.variant_scoring import AA,parse_mutations,mutated_sequence,score_from_table,ranking_metrics
from models.esm3_adapter import ESM3Adapter


def tables(adapter,backend,base,positions,batch):
    outputs=[];digests=[]
    mask=adapter.model.tokenizers.sequence.mask_token_id
    for start in range(0,len(positions),batch):
        ps=positions[start:start+batch]
        tokens=base.unsqueeze(0).repeat(len(ps),1)
        tokens[torch.arange(len(ps)),torch.tensor(ps)]=mask
        digests.append(hashlib.sha256(tokens.numpy().tobytes()).hexdigest())
        tokens=tokens.to(backend.device)
        adapter.pending_sequence_id=None
        result=adapter.model.logits(_BatchedESMProteinTensor(sequence=tokens),LogitsConfig(sequence=True,return_embeddings=False))
        selected=result.logits.sequence[torch.arange(len(ps),device=backend.device),torch.tensor(ps,device=backend.device)]
        outputs.append(selected.float().cpu())
        del selected,result
    backend.synchronize()
    return torch.cat(outputs),digests


def main(a):
    root=Path(a.output).resolve();root.mkdir(parents=True,exist_ok=False)
    dataset=Path(a.dataset).resolve();manifest=json.loads((dataset/'manifest.json').read_text())
    report={'status':'running','dataset_manifest_sha256':sha256(dataset/'manifest.json'),'parameters':vars(a),'assays':{},
       'scoring':'Sum over substitutions of mutant minus WT logits from a WT sequence masking only that position; equivalent to same-context log-softmax differences',
       'multiple_mutation_limit':'Additive WT masked contexts cannot directly model joint epistasis; observed correlation is assay-specific',
       'scope':'Complete selected assays only; not full ProteinGym, not a verified held-out pretraining test or official ESM3 leaderboard replication'}
    names=['scripts/proteingym_dms.py','common/variant_scoring.py','common/backend.py','common/reporting.py','models/esm3_adapter.py']
    report['source_sha256']={}
    for name in names:
        p=root/'source'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((REPO/name).read_bytes());report['source_sha256'][name]=sha256(REPO/name)
    backend=Backend(a.device);torch.set_num_threads(a.cpu_threads);backend.seed(7)
    adapter=ESM3Adapter(backend,a.precision,a.weights)
    report.update(environment=environment(REPO),device=backend.metadata(),weights=adapter.manifest,model_load_seconds=adapter.load_seconds)
    ids={aa:adapter.model.tokenizers.sequence.convert_tokens_to_ids(aa) for aa in AA}
    if len(set(ids.values()))!=20:raise ValueError('Canonical amino acids must have distinct tokenizer IDs')
    report['amino_token_ids']=ids
    for selected in manifest['quality_assays']:
        name=selected['id'];path=dataset/Path(selected['path']).name;meta=selected['metadata'];seq=meta['target_seq']
        if sha256(path)!=selected['sha256']:raise ValueError('Frozen assay contents changed')
        rows=list(csv.DictReader(path.open()));parsed=[];positions=set();labels=[];binary=[]
        start=time.perf_counter()
        for row in rows:
            changes=parse_mutations(row['mutant'],seq)
            if row.get('mutated_sequence')!=mutated_sequence(seq,changes):raise ValueError('Assay mutated_sequence differs from mutation notation')
            labels.append(float(row['DMS_score']));binary.append(int(row['DMS_score_bin']))
            positions.update(pos for _,pos,_ in changes);parsed.append(changes)
        preparation=time.perf_counter()-start
        positions=sorted(positions)
        base=adapter.model.encode(ESMProtein(sequence=seq)).sequence.detach().cpu()
        if len(base)!=len(seq)+2:raise ValueError('Unexpected tokenizer boundary convention')
        if base[1:-1].tolist()!=[ids[aa] for aa in seq]:raise ValueError('Tokenizer residue mapping mismatch')
        with torch.inference_mode(),backend.context():
            table,hashes=tables(adapter,backend,base,positions,a.batch_size) # full warmup
            probes=sorted({positions[0],positions[len(positions)//2],positions[-1]})
            singles,_=tables(adapter,backend,base,probes,1)
            sampled=table[[positions.index(p) for p in probes]]
            difference=(sampled-singles).abs()
            check={'positions':probes,'atol':.001,'rtol':.00001,'max_abs':float(difference.max()),
                   'mae':float(difference.mean()),'allclose':bool(torch.allclose(sampled,singles,atol=.001,rtol=.00001)),
                   'scope':'Three predetermined contexts validate B1 vs requested batch; all 64 sequence logits at each probe'}
            if not check['allclose']:raise ValueError('Masked batch numerical diagnostic failed')
            backend.reset_peak();times=[]
            for _ in range(a.runs):
                backend.synchronize();t=time.perf_counter();table,again=tables(adapter,backend,base,positions,a.batch_size)
                times.append(time.perf_counter()-t)
                if again!=hashes:raise ValueError('Masked context token inputs changed between measurements')
        if not torch.isfinite(table).all():raise ValueError('Nonfinite masked logits')
        score_start=time.perf_counter()
        predictions=[score_from_table(changes,positions,table,ids) for changes in parsed]
        metrics=ranking_metrics(labels,predictions,binary)
        scoring_seconds=time.perf_counter()-score_start
        out=root/name;out.mkdir()
        torch.save({'positions':positions,'masked_logits':table,'amino_token_ids':ids,'target_sequence':seq,
                    'token_input_sha256_per_batch':hashes},out/'score_tables.pt')
        with (out/'predictions.csv').open('w',newline='') as f:
            writer=csv.writer(f);writer.writerow(['mutant','DMS_score','DMS_score_bin','esm3_masked_marginal_score'])
            writer.writerows((r['mutant'],y,b,p) for r,y,b,p in zip(rows,labels,binary,predictions))
        result={'status':'pass','metadata':meta,'input_sha256':sha256(path),'rows':len(rows),'positions':positions,
            'masked_contexts':len(positions),'target_residues':len(seq),'prepared_assay_seconds':preparation,
            'latency_seconds':stats(times),'samples_seconds':times,
            'timing_scope':'Warmed complete masked-context sweep, including logits gathering/CPU table transfer; excludes CSV parse and variant lookup/ranking/artifacts',
            'throughput_scope':'Masked contexts, not variants; many variants reuse one context prediction',
            'masked_contexts_per_second':len(positions)*len(times)/sum(times),'cpu_lookup_and_metric_seconds':scoring_seconds,
            'batch_check':check,'quality':metrics,'memory':backend.memory(),'precision':adapter.precision_metadata(),
            'artifacts_sha256':{n:sha256(out/n) for n in ['score_tables.pt','predictions.csv']}}
        write_json(out/'report.json',result);report['assays'][name]=result
        write_json(root/'dms_summary.json',report)
        print(name,'rows',len(rows),'positions',len(positions),'Spearman',metrics['spearman'],flush=True)
    report['status']='pass';write_json(root/'dms_summary.json',report);print('PASS',root,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--precision',choices=['fp32','bf16'],default='fp32')
    p.add_argument('--batch-size',type=int,default=8);p.add_argument('--runs',type=int,default=3)
    p.add_argument('--cpu-threads',type=int,default=4)
    p.add_argument('--weights',default='/root/autodl-tmp/esm-baseline/model_snapshot.json')
    a=p.parse_args()
    if min(a.batch_size,a.runs,a.cpu_threads)<1:p.error('Positive batch, run and thread counts required')
    main(a)
