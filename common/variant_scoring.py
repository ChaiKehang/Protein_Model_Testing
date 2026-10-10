"""Validated masked-marginal variant scoring, independent of GPU backend."""
import math
import re
import numpy as np
AA='ACDEFGHIKLMNPQRSTVWY'
MUTATION=re.compile(r'^([ACDEFGHIKLMNPQRSTVWY])([1-9][0-9]*)([ACDEFGHIKLMNPQRSTVWY])$')


def parse_mutations(mutant,sequence):
    if not mutant or not sequence or not set(sequence)<=set(AA):raise ValueError('Canonical sequence and mutation notation required')
    result=[];seen=set()
    for text in mutant.split(':'):
        match=MUTATION.fullmatch(text)
        if not match:raise ValueError('Invalid substitution: '+text)
        wild,pos,target=match.group(1),int(match.group(2)),match.group(3)
        if pos>len(sequence) or sequence[pos-1]!=wild:raise ValueError('Mutation WT/position differs from target sequence: '+text)
        if pos in seen:raise ValueError('Duplicate mutation position')
        seen.add(pos);result.append((wild,pos,target))
    return result


def mutated_sequence(sequence,changes):
    letters=list(sequence)
    for _,pos,target in changes:letters[pos-1]=target
    return ''.join(letters)


def score_from_table(changes,positions,logits,token_ids):
    index={p:i for i,p in enumerate(positions)}
    value=0.
    for wild,pos,target in changes:
        if pos not in index:raise ValueError('Missing masked context position')
        row=logits[index[pos]]
        # Same-position log_softmax normalization cancels in this difference.
        value+=float(row[token_ids[target]])-float(row[token_ids[wild]])
    if not math.isfinite(value):raise ValueError('Nonfinite variant prediction')
    return value


def ranking_metrics(labels,predictions,binary):
    from scipy.stats import spearmanr,rankdata
    y,p,b=np.asarray(labels,dtype=float),np.asarray(predictions,dtype=float),np.asarray(binary,dtype=int)
    if y.ndim!=1 or p.shape!=y.shape or b.shape!=y.shape or len(y)<3:raise ValueError('Matched one-dimensional assay arrays required')
    if not np.isfinite(y).all() or not np.isfinite(p).all():raise ValueError('Nonfinite assay values')
    if not set(b.tolist())<={0,1}:raise ValueError('Binary labels must be zero/one')
    rho=float(spearmanr(y,p).statistic) if len(np.unique(y))>1 and len(np.unique(p))>1 else None
    npos=int(b.sum());nneg=len(b)-npos
    auc=float((rankdata(p,method='average')[b==1].sum()-npos*(npos+1)/2)/(npos*nneg)) if npos and nneg else None
    return {'spearman':rho,'auc':auc,'rows':len(y),'positive_labels':npos,'negative_labels':nneg,
            'direction':'Higher experimental DMS_score and predicted score mean higher fitness',
            'null_reason':'Constant ranks or missing class where corresponding metric is null',
            'aggregation':'This assay only; no full ProteinGym protein/function-category aggregation'}
