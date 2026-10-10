"""Microbenchmarks derived from profiled ESM shapes, with synthetic fixed inputs."""
import json
import math
import torch
import torch.nn.functional as F
from common.reporting import write_json, stats


def run_microbench(backend, directory):
    shapes = json.loads((directory / "operator_shapes.json").read_text())
    linear = next(s for s in shapes if s["type"] == "Linear" and len(s["input_shape"]) == 3
                  and s["weight_shape"][1] == 1536 and s["weight_shape"][0] > 1536)
    norm = next(s for s in shapes if s["type"] == "LayerNorm" and s["normalized_shape"] == [1536])
    attention = next(s for s in shapes if "n_heads" in s and len(s["input_shape"]) == 3)
    device = backend.device
    torch.manual_seed(17)
    b, t, k = linear["input_shape"]
    n = linear["weight_shape"][0]
    x = torch.randn(b*t, k, device=device, dtype=torch.bfloat16)
    w = torch.randn(n, k, device=device, dtype=torch.bfloat16) / math.sqrt(k)
    nx = torch.randn(*norm["input_shape"], device=device, dtype=torch.bfloat16)
    b, t, d = attention["input_shape"]
    h = attention["n_heads"]
    q, key, value = [torch.randn(b, h, t, d//h, device=device, dtype=torch.bfloat16) for _ in range(3)]
    def sdpa_math():
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
            return F.scaled_dot_product_attention(q, key, value)
    def sdpa_fp32():
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
            return F.scaled_dot_product_attention(q.float(), key.float(), value.float())
    operations = [
        ("GEMM", linear, lambda: x @ w.T, lambda: x.float() @ w.float().T),
        ("LayerNorm", norm, lambda: F.layer_norm(nx, (1536,)), lambda: F.layer_norm(nx.float(), (1536,))),
        ("SDPA_auto", attention, lambda: F.scaled_dot_product_attention(q, key, value), sdpa_fp32),
        ("SDPA_math", attention, sdpa_math, sdpa_fp32),
    ]
    results = []
    with torch.inference_mode(), backend.context():
        for name, source, operation, reference in operations:
            for _ in range(10):
                operation()
            backend.synchronize()
            events = backend.events()
            durations = []
            for _ in range(50):
                events[0].record()
                out = operation()
                events[1].record()
                events[1].synchronize()
                durations.append(events[0].elapsed_time(events[1]))
            ref = reference()
            delta = out.float() - ref
            results.append({"operator": name, "source_spec": source, "warmup": 10, "runs": 50,
                            "gpu_event_ms": durations, "gpu_latency_ms": stats(durations),
                            "finite": bool(torch.isfinite(out).all()), "mae_vs_fp32_same_quantized_inputs": float(delta.abs().mean()),
                            "relative_l2_vs_fp32": float(torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(ref))})
    write_json(directory / "microbench.json", {"scope": "synthetic inputs matching actual profiled shapes, BF16, model still resident; not whole-model speedup",
                                              "results": results})
