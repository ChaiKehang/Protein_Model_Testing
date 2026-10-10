"""Compare saved tensors using explicit masks and tolerances."""
import argparse
import json
from pathlib import Path
import torch
from common.reporting import write_json


def normalize(payload):
    result = dict(payload)
    if "logits" not in result and "sequence_logits" in result:
        result["logits"] = result["sequence_logits"]
    if "valid_mask" not in result and "sequence_ids" in result:
        ids = result["sequence_ids"]
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
        result["sequence_ids"] = ids
        result["valid_mask"] = ~((ids == 0) | (ids == 1) | (ids == 2) | (ids == 31))
    return result


def tensor_metrics(reference, candidate, mask, atol, rtol):
    if reference.shape != candidate.shape:
        raise ValueError(f"Shape differs: {tuple(reference.shape)} vs {tuple(candidate.shape)}")
    if reference.ndim != 3 or mask.shape != reference.shape[:2] or mask.dtype != torch.bool:
        raise ValueError("Expected [B,T,D] tensors and matching boolean [B,T] valid_mask")
    if not bool(mask.any()):
        raise ValueError("No valid residue positions")
    a, b = reference.float()[mask], candidate.float()[mask]
    if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
        raise ValueError("NaN/Inf at valid residue positions")
    delta = a - b
    norm_a, norm_b = torch.linalg.vector_norm(a), torch.linalg.vector_norm(b)
    cosine = None
    if norm_a > 0 and norm_b > 0:
        cosine = float(((a.flatten() @ b.flatten()) / (norm_a * norm_b)).clamp(-1, 1))
    elif norm_a == 0 and norm_b == 0:
        cosine = 1.0
    relative_l2 = float(torch.linalg.vector_norm(delta) / norm_a) if norm_a > 0 else (0.0 if norm_b == 0 else None)
    close = delta.abs() <= atol + rtol * a.abs()
    return {"shape": list(reference.shape), "valid_positions": int(mask.sum()),
            "mae": float(delta.abs().mean()), "max_abs": float(delta.abs().max()),
            "rmse": float(delta.square().mean().sqrt()), "relative_l2": relative_l2,
            "cosine": cosine, "close_fraction": float(close.float().mean()),
            "all_close": bool(close.all()),
            "top1_agreement": float((a.argmax(-1) == b.argmax(-1)).float().mean())}


def compare_payloads(reference, candidate, *, atol=1e-4, rtol=1e-4, keys=("logits",), min_top1=0.0):
    if atol < 0 or rtol < 0 or not 0 <= min_top1 <= 1:
        raise ValueError("Invalid tolerance or top-1 threshold")
    a, b = normalize(reference), normalize(candidate)
    if "valid_mask" not in a or "valid_mask" not in b:
        raise ValueError("Both artifacts must contain valid_mask or sequence_ids")
    mask = a["valid_mask"].cpu()
    if not torch.equal(mask, b["valid_mask"].cpu()):
        raise ValueError("Valid residue masks differ")
    if "sequence_ids" in a and "sequence_ids" in b and not torch.equal(a["sequence_ids"].cpu(), b["sequence_ids"].cpu()):
        raise ValueError("Token inputs differ; same-input comparison is invalid")
    metrics = {}
    for key in keys:
        if key not in a or key not in b:
            raise ValueError(f"Missing tensor: {key}")
        metrics[key] = tensor_metrics(a[key].cpu(), b[key].cpu(), mask, atol, rtol)
    passed = all(m["all_close"] for m in metrics.values())
    if "logits" in metrics:
        passed = passed and metrics["logits"]["top1_agreement"] >= min_top1
    return {"status": "pass" if passed else "fail", "atol": atol, "rtol": rtol,
            "min_top1": min_top1, "metrics": metrics,
            "scope": "valid residues only; BOS/EOS/padding/chainbreak excluded"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--min-top1", type=float, default=0.0)
    parser.add_argument("--keys", nargs="+", choices=["logits", "embeddings"], default=["logits"])
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error("Output exists; choose a new path")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = compare_payloads(torch.load(args.reference, map_location="cpu", weights_only=True),
                                  torch.load(args.candidate, map_location="cpu", weights_only=True),
                                  atol=args.atol, rtol=args.rtol, keys=args.keys, min_top1=args.min_top1)
    except (ValueError, OSError, RuntimeError) as exc:
        result = {"status": "error", "error": str(exc)}
    result.update(reference=str(Path(args.reference).resolve()), candidate=str(Path(args.candidate).resolve()))
    write_json(output, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
