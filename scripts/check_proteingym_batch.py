"""Separate FP32 diagnostics for length padding, batch grouping and pad isolation.

This is not a performance benchmark and does not alter failed BF16 acceptance.
"""
import argparse
import json
import os
from pathlib import Path
import sys
os.environ.setdefault('HF_HUB_OFFLINE','1')
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
import torch
from esm.sdk.api import LogitsConfig
from esm.utils.sampling import _BatchedESMProteinTensor
from esm.utils.constants import esm3 as C
from common.backend import Backend
from common.reporting import write_json,sha256
from compare_outputs import compare_payloads
from main import load_samples
from models.esm3_adapter import ESM3Adapter


def run(a):
    root=Path(a.output).resolve();root.mkdir(parents=True,exist_ok=False)
    report={'status':'running','parameters':vars(a),'checks':{},'source_sha256':{},
            'scope':'FP32 numerical diagnosis only. BF16 panel and its frozen thresholds remain unchanged.',
            'thresholds':{'logits':{'atol':.001,'rtol':.00001},'embeddings':{'atol':.01,'rtol':.00001},
                          'padding_value_isolation':{'atol':0,'rtol':0}}}
    for name in ['scripts/check_proteingym_batch.py','models/esm3_adapter.py','common/backend.py','compare_outputs.py']:
        p=root/'source'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((REPO/name).read_bytes());report['source_sha256'][name]=sha256(p)
    write_json(root/'batch_diagnostic.json',report)
    torch.set_num_threads(4);backend=Backend(a.device);backend.seed(7)
    adapter=ESM3Adapter(backend,'fp32',a.weights)
    samples=load_samples(a.inputs,a.sample_ids,len(a.sample_ids))
    def payload(tokens,mask,ids):
        adapter.pending_sequence_id=ids
        result=adapter.model.logits(_BatchedESMProteinTensor(sequence=tokens),LogitsConfig(sequence=True,return_embeddings=True))
        return {'sequence_ids':tokens.cpu(),'valid_mask':mask.cpu(),
                'logits':result.logits.sequence.cpu(),'embeddings':result.embeddings.cpu()}
    def compare(ref,candidate):
        return {key:compare_payloads(ref,candidate,keys=(key,),**report['thresholds'][key]) for key in ['logits','embeddings']}
    with torch.inference_mode(),backend.context():
        prepared=adapter.prepare('forward',samples)
        full=payload(prepared['tokens'],prepared['valid_mask'],(prepared['tokens']==C.SEQUENCE_PAD_TOKEN).long())
        torch.save(full,root/'batch_fp32.pt')
        for i,s in enumerate(samples):
            length=len(s['sequence'])+2
            singleprep=adapter.prepare('forward',[s])
            single=payload(singleprep['tokens'],singleprep['valid_mask'],None)
            padtokens=prepared['tokens'][i:i+1];padmask=prepared['valid_mask'][i:i+1]
            seqid=(padtokens==C.SEQUENCE_PAD_TOKEN).long()
            padded=payload(padtokens,padmask,seqid)
            row={k:v[i:i+1,:length] for k,v in full.items()}
            cut={k:v[:,:length] for k,v in padded.items()}
            altered=padtokens.clone();altered[seqid.bool()]=adapter.model.tokenizers.sequence.convert_tokens_to_ids('W')
            adversarial=payload(altered,padmask,seqid)
            # Inputs differ only at ignored padding; compare matching token metadata
            # explicitly after validating unchanged valid tokens and attention IDs.
            if not torch.equal(altered[padmask],padtokens[padmask]):raise ValueError('Valid input changed')
            adversarial['sequence_ids']=padded['sequence_ids']
            checks={'unpadded_vs_padded_B1':compare(single,cut),'unpadded_B1_vs_B4':compare(single,row),
                    'padded_B1_vs_B4':compare(cut,row),
                    'padding_value_isolation':compare_payloads(padded,adversarial,atol=0,rtol=0,keys=('logits','embeddings'))}
            report['checks'][s['id']]=checks
            torch.save({'single':single,'padded':padded,'adversarial_logits':adversarial['logits'],
                        'adversarial_embeddings':adversarial['embeddings'],'altered_input_tokens':altered.cpu(),
                        'attention_sequence_id':seqid.cpu()},root/(s['id']+'.pt'))
            write_json(root/'batch_diagnostic.json',report)
            print(s['id'],{k: (v.get('status') or {t:c['status'] for t,c in v.items()}) for k,v in checks.items()},flush=True)
    report['status']='pass' if all(c['status']=='pass' for checks in report['checks'].values() for name,v in checks.items()
                                  for c in ([v] if 'status' in v else v.values())) else 'partial'
    report.update(weights_revision=adapter.manifest['revision'],input_sha256=sha256(a.inputs),device=backend.metadata(),
                  precision=adapter.precision_metadata(),lengths=[len(s['sequence']) for s in samples])
    write_json(root/'batch_diagnostic.json',report);print(report['status'].upper(),root,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inputs',required=True);p.add_argument('--output',required=True)
    p.add_argument('--sample-ids',nargs='+',required=True);p.add_argument('--device',default='cuda:0')
    p.add_argument('--weights',default='/root/autodl-tmp/esm-baseline/model_snapshot.json');run(p.parse_args())
