"""Run in the isolated OpenFold environment, not the ESM dependency environment."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import unittest
import torch
from openfold.model.primitives import Attention
from models.openfold_adapter import torch_attention_hook, torch_ipa_hook

class AttentionPortTests(unittest.TestCase):
    def test_mask_bias_and_positional_flag_keep_official_torch_semantics(self):
        torch.manual_seed(12)
        module=Attention(c_q=16,c_k=16,c_v=16,c_hidden=4,no_heads=4).eval()
        with torch.no_grad():module.linear_o.weight.normal_()
        q=torch.randn(2,7,16);k=torch.randn(2,9,16)
        mask=torch.zeros(2,1,1,9);mask[...,7:]=-1e5
        bias=torch.randn(2,4,7,9)
        with torch.inference_mode():reference=module(q,k,[mask,bias],False)
        self.assertGreater(float(reference.abs().max()),.01)
        h=module.register_forward_pre_hook(torch_attention_hook,with_kwargs=True)
        try:
            with torch.inference_mode():
                # CPU would reject True without the explicit policy hook.
                positional=module(q,k,[mask,bias],True)
                keyword=module(q_x=q,kv_x=k,biases=[mask,bias],use_memory_efficient_kernel=True)
            self.assertTrue(torch.equal(reference,positional))
            self.assertTrue(torch.equal(reference,keyword))
            k2=k.clone();k2[:,7:]=1e3
            with torch.inference_mode():hidden_changed=module(q,k2,[mask,bias],True)
            self.assertTrue(torch.equal(reference,hidden_changed))
        finally:h.remove()

class IPAPortTests(unittest.TestCase):
    def test_explicit_non_inplace_ipa_preserves_official_cpu_path(self):
        from openfold.model.structure_module import InvariantPointAttention
        from openfold.utils.rigid_utils import Rigid
        torch.manual_seed(5)
        module=InvariantPointAttention(16,8,4,2,2,2).eval()
        with torch.no_grad():module.linear_out.weight.normal_()
        s=torch.randn(2,7,16);z=torch.randn(2,7,7,8)
        r=Rigid.identity((2,7),dtype=s.dtype,device=s.device,requires_grad=False)
        mask=torch.ones(2,7);mask[:,-2:]=0
        with torch.inference_mode():reference=module(s,z,r,mask,inplace_safe=False)
        self.assertGreater(float(reference.abs().max()),.01)
        h=module.register_forward_pre_hook(torch_ipa_hook,with_kwargs=True)
        try:
            with torch.inference_mode():
                positional=module(s,z,r,mask,True)
                keyword=module(s=s,z=z,r=r,mask=mask,inplace_safe=True)
            self.assertTrue(torch.equal(reference,positional))
            self.assertTrue(torch.equal(reference,keyword))
        finally:h.remove()

if __name__=='__main__':unittest.main()
