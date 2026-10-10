"""Pinned ESM 3.2.3 adapter with explicit weights, precision and task checks."""
import importlib.metadata
import json
from pathlib import Path
import time
import types
import torch
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, ESMProteinError, GenerationConfig, LogitsConfig
from esm.utils.sampling import _BatchedESMProteinTensor
from esm.utils.constants import esm3 as C
import esm.pretrained as pretrained
from common.backend import DTYPES
from common.reporting import sha256, write_json


def ca_rmsd(predicted, reference):
    """Kabsch CA RMSD in angstroms; sequences must already be matched."""
    a, b = predicted.double(), reference.double()
    valid = torch.isfinite(a).all(-1) & torch.isfinite(b).all(-1)
    if int(valid.sum()) < 3:
        return None, int(valid.sum())
    a, b = a[valid], b[valid]
    a, b = a - a.mean(0), b - b.mean(0)
    u, _, vh = torch.linalg.svd(a.T @ b)
    correction = torch.eye(3, dtype=a.dtype)
    correction[-1, -1] = torch.det(u @ vh)
    aligned = a @ (u @ correction @ vh)
    return float((aligned - b).square().sum(-1).mean().sqrt()), int(valid.sum())


class ESM3Adapter:
    def __init__(self, backend, precision, manifest_path, attention="auto", verify_weights=False):
        if importlib.metadata.version("esm") != "3.2.3":
            raise RuntimeError("Adapter verified for esm==3.2.3; review before upgrading")
        if backend.device.type == "cpu" and precision != "fp32":
            raise ValueError("CPU reference currently supports fp32 only")
        if precision == "bf16" and backend.device.type == "cuda":
            with backend.context():
                if not torch.cuda.is_bf16_supported():
                    raise RuntimeError("Selected GPU does not support BF16")
        self.backend, self.precision = backend, precision
        self.manifest_path = Path(manifest_path).resolve()
        self.manifest = json.loads(self.manifest_path.read_text())
        root = Path(self.manifest["snapshot_dir"]).resolve()
        required = [root / "data/weights" / name for name in self.manifest["weights_sha256"]]
        required.append(root / "data/uniref90_and_mgnify90_residue_annotations_gt_1k_proteins.csv")
        for path in required:
            if not path.is_file():
                raise FileNotFoundError(f"Missing snapshot resource: {path}")
        self.weight_verification = "manifest_hashes_recorded_not_rehashed_this_process"
        if verify_weights:
            for filename, expected in self.manifest["weights_sha256"].items():
                if sha256(root / "data/weights" / filename) != expected:
                    raise RuntimeError(f"Checkpoint SHA-256 mismatch: {filename}")
            self.weight_verification = "all_manifest_weights_rehashed_and_matched"
        C.data_root = lambda model_name: root
        pretrained.data_root = C.data_root
        torch.set_float32_matmul_precision("highest")
        if backend.device.type == 'cuda':
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        elif backend.device.type == 'musa':
            mudnn = getattr(torch.backends, 'mudnn', None)
            if mudnn is not None and hasattr(mudnn, 'allow_tf32'):
                mudnn.allow_tf32 = False
        if attention == "math" and backend.device.type != "cuda":
            raise ValueError("Forced math SDPA is CUDA-specific")
        self.attention = attention
        self.pending_sequence_id = None
        self.linear_observations = {}
        start = time.perf_counter()
        self.model = ESM3.from_pretrained("esm3-sm-open-v1", device=torch.device("cpu"))
        self.model.to(device=backend.device, dtype=DTYPES[precision]).eval()
        upstream_forward = self.model.forward

        def controlled_forward(model, *args, **kwargs):
            if self.pending_sequence_id is not None:
                kwargs["sequence_id"] = self.pending_sequence_id
            # Control evaluation precision without changing upstream computation.
            with torch.autocast(device_type=backend.device.type, enabled=precision != "fp32", dtype=DTYPES[precision]):
                if self.attention == "math":
                    with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
                        return upstream_forward(*args, **kwargs)
                return upstream_forward(*args, **kwargs)

        self.model.forward = types.MethodType(controlled_forward, self.model)
        name, first_linear = next((n, m) for n, m in self.model.named_modules() if isinstance(m, torch.nn.Linear))
        def observe(module, inputs, output):
            self.linear_observations.update(module=name, input_dtype=str(inputs[0].dtype), output_dtype=str(output.dtype))
        first_linear.register_forward_hook(observe)
        backend.synchronize()
        self.load_seconds = time.perf_counter() - start
        self.load_memory = backend.memory()

    def precision_metadata(self):
        aux = {}
        for name in ["_structure_encoder", "_structure_decoder", "_function_decoder"]:
            module = getattr(self.model, name, None)
            if module is not None:
                aux[name] = str(next(module.parameters()).dtype)
        return {"weights": str(next(self.model.parameters()).dtype),
                "main_compute_policy": "FP32 autocast disabled" if self.precision == "fp32" else f"{self.precision} mixed precision",
                "sdk_inner_override": True, "observed_first_linear": dict(self.linear_observations),
                "auxiliary_parameter_dtypes": aux,
                "tf32_matmul": torch.backends.cuda.matmul.allow_tf32 if self.backend.device.type == 'cuda' else None,
                "tf32_cudnn": torch.backends.cudnn.allow_tf32 if self.backend.device.type == 'cuda' else None,
                "tf32_mudnn": getattr(getattr(torch.backends, 'mudnn', None), 'allow_tf32', None) if self.backend.device.type == 'musa' else None,
                "attention_requested": self.attention,
                "note": "FP32 computes the same checkpoint in FP32; cannot recover unavailable training precision. Auxiliary modules retain SDK policies."}

    def prepare(self, task, samples, pdb_path=None, mask_positions=(4, 5, 13), steps=None):
        self.pending_sequence_id = None
        if task == "forward":
            sequences = [s["sequence"] for s in samples]
            rows = [self.model.encode(ESMProtein(sequence=s)).sequence for s in sequences]
            token_lengths = [len(r) for r in rows]
            padded = torch.nn.utils.rnn.pad_sequence(rows, batch_first=True, padding_value=C.SEQUENCE_PAD_TOKEN)
            valid = torch.zeros_like(padded, dtype=torch.bool)
            for i, seq in enumerate(sequences):
                valid[i, 1:len(seq) + 1] = True
            seq_ids = (padded == C.SEQUENCE_PAD_TOKEN).long() if len(set(token_lengths)) > 1 else None
            tokens = _BatchedESMProteinTensor(sequence=padded)
            config = LogitsConfig(sequence=True, return_embeddings=True)
            def operation():
                self.pending_sequence_id = seq_ids
                return self.model.logits(tokens, config)
            return {"task": task, "samples": samples, "operation": operation,
                    "tokens": padded, "valid_mask": valid, "padding_attention_mask": seq_ids is not None}
        if len(samples) != 1:
            raise ValueError("Generation verification currently requires batch_size=1")
        sequence = samples[0]["sequence"]
        if task == "structure":
            prompt = ESMProtein(sequence=sequence)
            track, nsteps = "structure", steps or 8
        else:
            positions = list(mask_positions)
            if not positions or any(p < 1 or p > len(sequence) for p in positions) or len(set(positions)) != len(positions):
                raise ValueError("Mask positions must be unique one-based sequence positions")
            letters = list(sequence)
            for p in positions:
                letters[p - 1] = "_"
            prompt = ESMProtein(sequence="".join(letters))
            track, nsteps = "sequence", steps or len(positions)
        ref = None
        if pdb_path and task in {"structure", "inverse"}:
            ref = ESMProtein.from_pdb(Path(pdb_path), chain_id="A")
            if ref.sequence != sequence:
                raise ValueError("PDB chain A sequence differs from selected input")
            if task == "inverse":
                prompt.coordinates = ref.coordinates
        if task == "inverse" and ref is None:
            raise ValueError("Inverse task requires matching PDB")
        config = GenerationConfig(track=track, num_steps=nsteps, schedule="linear", strategy="entropy",
                                  temperature=0.7, temperature_annealing=False, condition_on_coordinates_only=False)
        extra = {}
        if task == "inverse":
            encoded = self.model.encode(prompt)
            extra["input_structure_ids"] = encoded.structure.detach().cpu()
        def operation():
            self.pending_sequence_id = None
            return self.model.generate(prompt.copy(), config)
        return {"task": task, "samples": samples, "operation": operation, "prompt": prompt,
                "config": config, "reference": ref, "reference_pdb": str(pdb_path) if ref else None,
                "mask_positions": list(mask_positions), "extra": extra}

    def artifacts(self, prepared, result, directory):
        task = prepared["task"]
        if task == "forward":
            logits, embeddings = result.logits.sequence, result.embeddings
            shape = (len(prepared["samples"]), prepared["tokens"].shape[1], 64)
            if tuple(logits.shape) != shape or not bool(torch.isfinite(logits).all() and torch.isfinite(embeddings).all()):
                raise RuntimeError("Invalid shape or nonfinite forward output")
            payload = {"logits": logits.detach().cpu(), "embeddings": embeddings.detach().cpu(),
                       "sequence_ids": prepared["tokens"].detach().cpu(), "valid_mask": prepared["valid_mask"].cpu()}
            quality = {"logits_shape": list(logits.shape), "embeddings_shape": list(embeddings.shape),
                       "finite": True, "padding_attention_mask": prepared["padding_attention_mask"],
                       "valid_residues": int(payload["valid_mask"].sum())}
            return payload, quality
        if isinstance(result, ESMProteinError):
            raise RuntimeError(f"ESM3: {result.error_code}: {result.error_msg}")
        if not isinstance(result, ESMProtein):
            raise TypeError(f"Unexpected SDK result: {type(result)}")
        sequence = prepared["samples"][0]["sequence"]
        if not result.sequence or "_" in result.sequence or len(result.sequence) != len(sequence):
            raise RuntimeError("Invalid generated sequence")
        payload = dict(prepared["extra"])
        quality = {"reference_sequence": sequence, "generated_sequence": result.sequence,
                   "coordinate_condition": prepared["prompt"].coordinates is not None}
        if task == "structure":
            if result.sequence != sequence or result.coordinates is None:
                raise RuntimeError("Structure generation did not retain sequence or return coordinates")
            coords = result.coordinates.detach().cpu()
            valid = torch.isfinite(coords[:, 1, :]).all(-1)
            if not bool(valid.all()):
                raise RuntimeError("All expected CA coordinates must be finite")
            payload["coordinates"] = coords
            quality.update(coordinates_shape=list(coords.shape), finite_ca_count=int(valid.sum()),
                           finite_ca_fraction=float(valid.float().mean()))
            if prepared["reference"] is not None:
                rmsd, matched = ca_rmsd(coords[:, 1, :], prepared["reference"].coordinates[:, 1, :])
                quality.update(ca_rmsd_angstrom=rmsd, rmsd_matched_residues=matched,
                               rmsd_scope=f"{Path(prepared['reference_pdb']).stem} A; one-to-one sequence mapping; Kabsch CA alignment")
            for name in ["plddt", "ptm"]:
                value = getattr(result, name)
                if value is not None:
                    payload[name] = value.detach().cpu()
                    finite = value[torch.isfinite(value)]
                    quality[f"{name}_mean"] = float(finite.mean()) if finite.numel() else None
            result.to_pdb(directory / "candidate.pdb")
        else:
            positions = prepared["mask_positions"]
            if any(result.sequence[i] != sequence[i] for i in range(len(sequence)) if i + 1 not in positions):
                raise RuntimeError("Known sequence positions changed")
            letters = [{"position": p, "reference": sequence[p - 1], "generated": result.sequence[p - 1]} for p in positions]
            quality.update(known_positions_retained=True, masked_positions=letters,
                           masked_recovery_fraction=sum(x["reference"] == x["generated"] for x in letters) / len(letters))
        write_json(directory / "generated.json", quality)
        return payload, quality
