"""Compare predicted CA geometry with experimental mmCIF after sequence mapping."""
import argparse
import json
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[1]
import numpy as np
sys.path.insert(0,str(REPO))
from common.reporting import sha256,write_json


def kabsch_rmsd(predicted,reference):
    p,q=np.asarray(predicted,dtype=np.float64),np.asarray(reference,dtype=np.float64)
    if p.shape != q.shape or p.ndim!=2 or p.shape[1]!=3 or p.shape[0]<3:
        raise ValueError('At least three matching 3D points required')
    if not np.isfinite(p).all() or not np.isfinite(q).all():raise ValueError('Nonfinite coordinates')
    p,q=p-p.mean(0),q-q.mean(0)
    u,_,vt=np.linalg.svd(p.T@q)
    correction=np.eye(3);correction[-1,-1]=np.linalg.det(u@vt)
    return float(np.sqrt(np.mean(np.sum((p@(u@correction@vt)-q)**2,axis=-1))))


def ca_lddt(predicted,reference,cutoff=15.):
    p,q=np.asarray(predicted),np.asarray(reference)
    if p.shape!=q.shape or p.ndim!=2 or p.shape[1]!=3 or len(p)<3:
        raise ValueError('At least three matching 3D points required')
    if not np.isfinite(p).all() or not np.isfinite(q).all():raise ValueError('Nonfinite coordinates')
    dp=np.linalg.norm(p[:,None,:]-p[None,:,:],axis=-1)
    dq=np.linalg.norm(q[:,None,:]-q[None,:,:],axis=-1)
    mask=(dq<cutoff)&~np.eye(len(p),dtype=bool)
    if not mask.any():raise ValueError('No reference CA pairs within cutoff')
    errors=np.abs(dp-dq)[mask]
    return float(np.mean(np.stack([errors<t for t in [.5,1.,2.,4.]]))),int(mask.sum())


def main(a):
    import torch
    from Bio.Align import PairwiseAligner
    from openfold.data import mmcif_parsing
    from openfold.np import residue_constants
    from openfold.utils.loss import lddt_ca
    run=Path(a.run).resolve()
    prediction=torch.load(run/'tensors.pt',map_location='cpu',weights_only=True)
    features=torch.load(run/'features.pt',map_location='cpu',weights_only=True)
    residues=features['aatype'][...,-1]
    seq=''.join(residue_constants.restypes_with_x[i] for i in residues.tolist())
    parsed=mmcif_parsing.parse(file_id=Path(a.reference).stem,mmcif_string=Path(a.reference).read_text())
    if parsed.mmcif_object is None:raise ValueError(f'mmCIF parse failed: {parsed.errors}')
    reference=parsed.mmcif_object
    aligner=PairwiseAligner(mode='global',match_score=2,mismatch_score=-1,open_gap_score=-4,extend_gap_score=-.5)
    alignments=[]
    for chain,sequence in reference.chain_to_seqres.items():
        alignment=aligner.align(seq,sequence)[0]
        alignments.append((alignment.score,chain,sequence,alignment))
    if a.chain:alignments=[x for x in alignments if x[1]==a.chain]
    if not alignments:raise ValueError('No requested chain')
    _,chain,reference_seq,alignment=max(alignments,key=lambda x:x[0])
    positions,mask=mmcif_parsing.get_atom_coords(reference,chain)
    indexpairs=[(int(i),int(j)) for i,j in alignment.indices.T if i>=0 and j>=0 and seq[i]==reference_seq[j]]
    valid=[(i,j) for i,j in indexpairs if mask[j,1] and prediction['final_atom_mask'][i,1]
           and np.isfinite(positions[j,1]).all() and torch.isfinite(prediction['final_atom_positions'][i,1]).all()]
    ii,jj=zip(*valid)
    pred=prediction['final_atom_positions'][list(ii)].float()
    true=torch.tensor(positions[list(jj)],dtype=pred.dtype)
    sharedmask=torch.tensor(mask[list(jj)])*prediction['final_atom_mask'][list(ii)]
    lddt_value,pair_count=ca_lddt(pred[:,1].numpy(),true[:,1].numpy())
    official=float(lddt_ca(pred,true,sharedmask,per_residue=False))
    if abs(official-lddt_value)>1e-6:raise ValueError('CA lDDT implementation differs from pinned OpenFold metric')
    result={'scope':'One repository-provided target; sequence-matched finite experimental CA only; no dataset-level validation claim',
        'reference':{'path':a.reference,'sha256':sha256(a.reference),'chain':chain,
                     'resolution':reference.header.get('resolution'),'sequence':reference_seq},
        'prediction_sequence':seq,'prediction_residues':len(seq),
        'alignment':{'mode':'global','match':2,'mismatch':-1,'open_gap':-4,'extend_gap':-.5,
                     'equal_residue_pairs':len(indexpairs),'valid_ca_pairs':len(valid),
                     'prediction_ca_coverage':len(valid)/len(seq),'index_pairs':valid,
                     'mismatch_or_gap_prediction_positions':len(seq)-len(indexpairs)},
        'metrics':{'ca_rmsd_angstrom':kabsch_rmsd(pred[:,1].numpy(),true[:,1].numpy()),
                   'ca_lddt':lddt_value,'official_openfold_ca_lddt':official,'directed_neighbor_pairs':pair_count},
        'metric_definition':'Kabsch RMSD removes rotation/translation; CA lDDT uses reference distances <15 A and error thresholds .5/1/2/4 A, excluding diagonal',
        'acceptance':'Exploratory accuracy measurement; no post-hoc quality threshold or cross-device acceptance decision',
        'metric_source_sha256':sha256(__file__),'tensors_sha256':sha256(run/'tensors.pt'),
        'features_sha256':sha256(run/'features.pt')}
    target=run/'structure_quality.json'
    if target.exists():raise FileExistsError(target)
    write_json(target,result);print(json.dumps(result['metrics'],indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True)
    p.add_argument('--reference',required=True);p.add_argument('--chain')
    main(p.parse_args())
