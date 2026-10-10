"""Compare same-feature OpenFold runs under predeclared exploratory thresholds."""
import argparse
import json
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
import torch
from common.reporting import write_json,sha256
from scripts.openfold_structure_quality import kabsch_rmsd


def compare(before,after,contract):
    reports=[json.loads((p/'report.json').read_text()) for p in [before,after]]
    for r in reports:
        if r['status']!='pass':raise ValueError('Both runs must pass')
    for field in ['config','weights','features']:
        if field=='weights':
            if reports[0][field]['sha256']!=reports[1][field]['sha256']:raise ValueError('Weight contents differ')
        elif field=='features':
            if reports[0][field]['feature_sha256']!=reports[1][field]['feature_sha256']:raise ValueError('Input feature contents differ')
        elif reports[0][field]!=reports[1][field]:raise ValueError('Model configuration differs')
    if reports[0]['openfold_commit']!=reports[1]['openfold_commit']:raise ValueError('OpenFold commits differ')
    a,b=[torch.load(p/'tensors.pt',map_location='cpu',weights_only=True) for p in [before,after]]
    if not torch.equal(a['final_atom_mask'],b['final_atom_mask']):raise ValueError('Atom masks differ')
    metrics={}
    for key in ['final_atom_positions','plddt','distogram_logits','predicted_aligned_error']:
        if key not in a or key not in b:raise ValueError(f'Missing output field: {key}')
        x,y=a[key].float(),b[key].float()
        if x.shape!=y.shape:raise ValueError(f'Shapes differ: {key}')
        if key=='final_atom_positions':x,y=x[a['final_atom_mask'].bool()],y[a['final_atom_mask'].bool()]
        if not torch.isfinite(x).all() or not torch.isfinite(y).all():raise ValueError('Nonfinite valid outputs')
        diff=(x-y).abs()
        metrics[key]={'mae':float(diff.mean()),'max_abs':float(diff.max()),'relative_l2':float(torch.linalg.vector_norm(x-y)/torch.linalg.vector_norm(x).clamp(min=1e-20))}
    ca=a['final_atom_mask'][:,1].bool()
    rmsd=kabsch_rmsd(a['final_atom_positions'][ca,1].numpy(),b['final_atom_positions'][ca,1].numpy())
    threshold=contract['comparison_diagnostic_thresholds']
    passed=rmsd<=threshold['ca_aligned_rmsd_angstrom'] and metrics['plddt']['max_abs']<=threshold['plddt_max_abs']
    return {'status':'pass' if passed else 'fail','scope':threshold['note'],
        'attention_implementations':{'reference':reports[0]['parameters'].get('attention_kernel','upstream'),
                                    'candidate':reports[1]['parameters'].get('attention_kernel','upstream')},
        'implementation_note':'Model config/weights/features match; attention kernel policy may differ explicitly. This is numerical validation, not an identical-kernel replay.',
        'reference':str(before),'candidate':str(after),'reference_report_sha256':sha256(before/'report.json'),
        'candidate_report_sha256':sha256(after/'report.json'),'feature_sha256':reports[0]['features']['feature_sha256'],
        'thresholds':threshold,'ca_aligned_rmsd_angstrom':rmsd,'metrics':metrics,
        'exact_atom_mask':True,'same_config_weights_features_commit':True,
        'comparison_source_sha256':sha256(__file__)}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference',required=True)
    p.add_argument('--candidate',required=True);p.add_argument('--output',required=True)
    p.add_argument('--contract',default=str(REPO/'configs/openfold_reference.json'));a=p.parse_args()
    out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    result=compare(Path(a.reference),Path(a.candidate),json.loads(Path(a.contract).read_text()))
    write_json(out,result);print(json.dumps(result,indent=2))
    sys.exit(0 if result['status']=='pass' else 1)
