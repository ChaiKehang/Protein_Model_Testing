"""Compare like-for-like run reports; reject mismatched scientific settings."""
import argparse
import csv
import json
from pathlib import Path


def reports(root):
    root=Path(root);out={}
    for folder in ['esm3','esm3_standard']:
        for p in (root/folder).glob('*/report.json'):
            r=json.loads(p.read_text(encoding='utf-8'));c=r['config']
            identity=dict(model='ESM3',task=r['task'],batch=c['batch_size'],lengths=r['residue_lengths'],
                          precision=c['precision'],attention=c['attention'],seed=c['seed'],steps=c['steps'],
                          mask_positions=c['mask_positions'],sequence_sha256=r['sequence_sha256'],
                          reference_pdb_sha256=r.get('reference_pdb_sha256'),weights=r['weights']['revision'],
                          esm_commit=r['environment'].get('esm_commit'),timing_scope=r['timing_scope'],
                          warmup=c['warmup'],formal_runs=r['latency']['count'],cpu_threads=c['cpu_threads'],
                          precision_policy={k:v for k,v in r['precision'].items() if k not in ('observed_first_linear',)},
                          inference_source={k:r['environment']['source_sha256'][k] for k in
                                            ['models/esm3_adapter.py','common/backend.py']})
            out[folder+'/'+p.parent.name]=dict(identity=identity,latency=r['latency'],memory=r['memory'],status=r['status'])
    p=root/'openfold/report.json'
    if p.exists():
        r=json.loads(p.read_text(encoding='utf-8'))
        identity=dict(model='OpenFold',task='structure',batch=1,lengths=[r['output']['residues']],
                      precision=r['precision'],model_config=r['config'],weight_sha256=r['weights']['sha256'],
                      feature_sha256=r['features']['feature_sha256'],openfold_commit=r['openfold_commit'],
                      attention=r['attention_implementation']['policy'],timing_scope=r['timing_scope'],
                      warmup=r['parameters']['warmup'],formal_runs=r['latency_seconds']['count'],
                      cpu_threads=r['parameters']['cpu_threads'],seed=r['parameters']['seed'],
                      inference_source={k:r['source_sha256'][k] for k in ['models/openfold_adapter.py','common/backend.py']})
        out['openfold/6KWC']=dict(identity=identity,latency=r['latency_seconds'],memory=r['memory'],status=r['status'])
    return out


def main(a):
    reference=reports(a.reference);candidate=reports(a.candidate);rows=[]
    if not reference or not candidate:raise ValueError('No measured reports found')
    for name in sorted(set(reference)|set(candidate)):
        row=dict(case=name,status='',mismatch_fields='',reference_median_seconds='',candidate_median_seconds='',
                 latency_speedup='',reference_allocated_peak_gib='',candidate_allocated_peak_gib='')
        r=reference.get(name);c=candidate.get(name)
        if r is None or c is None:row['status']='missing_reference_or_candidate'
        else:
            mismatch=[k for k in set(r['identity'])|set(c['identity']) if r['identity'].get(k)!=c['identity'].get(k)]
            if mismatch:row.update(status='not_comparable',mismatch_fields=';'.join(sorted(mismatch)))
            elif r['status']!='pass' or c['status']!='pass':row['status']='execution_not_pass'
            elif not r['latency']['median'] or not c['latency']['median']:row['status']='no_formal_latency'
            else:
                row.update(status='settings_match_numeric_quality_still_required',
                           reference_median_seconds=r['latency']['median'],candidate_median_seconds=c['latency']['median'],
                           latency_speedup=r['latency']['median']/c['latency']['median'],
                           reference_allocated_peak_gib=r['memory']['allocated_peak_gib'],
                           candidate_allocated_peak_gib=c['memory']['allocated_peak_gib'])
        rows.append(row)
    dest=Path(a.output)
    if dest.exists():raise FileExistsError(dest)
    dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('Wrote',len(rows),'comparisons to',dest)
    print('Settings match does not imply valid cross-device outputs; run complete tensor/structure comparisons separately.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference',required=True)
    p.add_argument('--candidate',required=True);p.add_argument('--output',required=True);main(p.parse_args())
