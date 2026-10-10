"""OpenFold monomer with public weights and precomputed alignments.

No relaxation, template search, synthetic weights, or CPU fallback. Keep feature
preprocessing on CPU (as in upstream); run AlphaFold on the requested backend.
"""
from pathlib import Path
import hashlib
import json
import subprocess
import torch

PINNED_COMMIT = 'be2ec1841f16c966c65ae0e7599ebbadc725757d'


def torch_attention_hook(module, args, kwargs):
    """Explicitly select upstream's matmul/softmax branch, keeping biases/masks.

    The flag is argument four of pinned Attention.forward. No device transfer,
    weights change, or replacement attention formula is introduced.
    """
    kwargs = dict(kwargs)
    if len(args) >= 4:
        args = (*args[:3], False, *args[4:])
        kwargs.pop('use_memory_efficient_kernel', None)
    else:
        kwargs['use_memory_efficient_kernel'] = False
    return args, kwargs


def torch_ipa_hook(module, args, kwargs):
    """Select official non-inplace IPA, which uses torch softmax on CPU.

    Upstream inplace_safe=True invokes the CUDA softmax extension directly and
    also uses matmul for point aggregation; False is its official tensor path.
    The different aggregation order must pass the full structural comparison.
    """
    kwargs = dict(kwargs)
    if len(args) >= 5:
        args = (*args[:4], False, *args[5:])
        kwargs.pop('inplace_safe', None)
    else:
        kwargs['inplace_safe'] = False
    return args, kwargs


def tensor_digest(features):
    h = hashlib.sha256()
    for name, value in sorted(features.items()):
        value = value.detach().cpu().contiguous()
        h.update(json.dumps([name, list(value.shape), str(value.dtype)]).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


class OpenFoldAdapter:
    def __init__(self, backend, weights, chunk_size=64, seed=42, attention_kernel='upstream'):
        from openfold.config import model_config
        from openfold.model.model import AlphaFold
        from openfold.utils.import_weights import import_openfold_weights_
        import openfold
        self.source = Path(openfold.__file__).resolve().parents[1]
        commit = subprocess.check_output(['git','-C',str(self.source),'rev-parse','HEAD'], text=True).strip()
        if commit != PINNED_COMMIT:
            raise ValueError(f'OpenFold commit differs from frozen contract: {commit}')
        self.backend, self.seed = backend, seed
        if attention_kernel not in {'upstream','torch'}:
            raise ValueError('Unknown explicit attention implementation')
        self.attention_kernel = attention_kernel
        self.config = model_config('model_3_ptm', precision='fp32')
        # Explicit benchmarking policy: fixed chunk, no per-device autotuning.
        # All model layers, MSA limits, and three recycling steps remain upstream.
        self.config.globals.chunk_size = chunk_size
        self.config.model.extra_msa.extra_msa_stack.tune_chunk_size = False
        self.config.model.evoformer_stack.tune_chunk_size = False
        self.config.model.template.template_pair_stack.tune_chunk_size = False
        backend.seed(seed)
        if backend.device.type == 'cuda':
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        elif backend.device.type == 'musa':
            raise RuntimeError('OpenFold custom extension and backend paths require an actual MUSA port; no fallback is allowed')
        self.model = AlphaFold(self.config).eval()
        state = torch.load(weights, map_location='cpu', weights_only=True)
        if 'ema' in state:
            state = state['ema']['params']
        import_openfold_weights_(self.model, state)
        self.attention_handles = []
        if attention_kernel == 'torch':
            from openfold.model.primitives import Attention
            from openfold.model.structure_module import InvariantPointAttention
            for module in self.model.modules():
                if isinstance(module, Attention):
                    self.attention_handles.append(module.register_forward_pre_hook(torch_attention_hook, with_kwargs=True))
                elif isinstance(module, InvariantPointAttention):
                    self.attention_handles.append(module.register_forward_pre_hook(torch_ipa_hook, with_kwargs=True))
        self.model.to(backend.device)
        backend.synchronize()

    def prepare(self, fasta, alignments, feature_file=None):
        if feature_file:
            features = torch.load(feature_file, map_location='cpu', weights_only=True)
        else:
            from openfold.data.data_pipeline import DataPipeline
            from openfold.data.feature_pipeline import FeaturePipeline
            self.backend.seed(self.seed)
            raw = DataPipeline(template_featurizer=None).process_fasta(str(fasta), str(alignments))
            features = FeaturePipeline(self.config.data).process_features(raw, mode='predict')
        if not features or not all(isinstance(t, torch.Tensor) for t in features.values()):
            raise ValueError('Expected a nonempty feature tensor dictionary')
        self.features_cpu = {k: v.detach().cpu() for k, v in features.items()}
        self.feature_sha256 = tensor_digest(self.features_cpu)
        self.features = {k: v.to(self.backend.device) for k, v in self.features_cpu.items()}
        self.backend.synchronize()
        return {'feature_sha256': self.feature_sha256,
                'shapes': {k: list(v.shape) for k,v in features.items()},
                'dtypes': {k: str(v.dtype) for k,v in features.items()}}

    @torch.inference_mode()
    def run(self):
        return self.model(self.features)

    def pdb(self, result):
        from openfold.np import protein
        out = {k: v.detach().cpu().numpy() for k,v in result.items() if isinstance(v,torch.Tensor)}
        features = {k: v[..., -1].numpy() for k,v in self.features_cpu.items()}
        b_factors = out['plddt'][:,None] * out['final_atom_mask']
        prediction = protein.from_prediction(features, out, b_factors=b_factors,
                                             remove_leading_feature_dimension=False)
        return protein.to_pdb(prediction)
