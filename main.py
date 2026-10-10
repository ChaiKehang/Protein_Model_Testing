"""Offline, reproducible ESM3 inference measurement; one configuration per process."""
import argparse
import datetime
import json
import os
from pathlib import Path
import time
import traceback
import uuid

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch
from common.backend import Backend, DTYPES
from common.reporting import environment, sha256, stats, write_json
from models.esm3_adapter import ESM3Adapter

REPO = Path(__file__).resolve().parent


def load_samples(path, ids, batch_size):
    samples = {}
    for line in Path(path).read_text().splitlines():
        sample = json.loads(line)
        if sample["id"] in samples:
            raise ValueError(f"Duplicate sample ID: {sample['id']}")
        if not sample["sequence"] or any(c not in "ACDEFGHIKLMNPQRSTVWY" for c in sample["sequence"]):
            raise ValueError(f"Invalid sequence: {sample['id']}")
        samples[sample["id"]] = sample
    selected = [samples[id] for id in ids]
    if len(selected) == 1:
        selected *= batch_size
    elif len(selected) != batch_size:
        raise ValueError("Number of explicit sample IDs must equal batch_size")
    return selected


def timed_call(backend, operation, events=None):
    backend.synchronize()
    t0 = time.perf_counter()
    if events:
        events[0].record()
    result = operation()
    if events:
        events[1].record()
    backend.synchronize()
    elapsed = time.perf_counter() - t0
    return result, elapsed, events[0].elapsed_time(events[1]) if events else None


def run(args):
    output = Path(args.output).resolve() if args.output else REPO / "outputs" / (
        datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "run_id": output.name, "status": "running", "task": args.task,
              "config": vars(args), "output_directory": str(output)}
    try:
        torch.set_num_threads(args.cpu_threads)
        backend = Backend(args.device)
        backend.seed(args.seed)
        samples = load_samples(args.inputs, args.sample_ids or [args.proteinID], args.batch_size)
        if args.task != "forward" and args.batch_size != 1:
            raise ValueError("Generation task verification currently uses batch_size=1")
        report.update(environment=environment(REPO), device=backend.metadata(),
                      sample_ids=[s["id"] for s in samples], residue_lengths=[len(s["sequence"]) for s in samples],
                      input_sequences=[s["sequence"] for s in samples],
                      input_sha256=sha256(args.inputs),
                      sequence_sha256=[__import__('hashlib').sha256(s["sequence"].encode()).hexdigest() for s in samples],
                      seed_rule="reset before every warmup/measured call; reset excluded from timing")
        backend.reset_peak()
        adapter = ESM3Adapter(backend, args.precision, args.manifest, args.attention, args.verify_weights)
        report.update(model_load_seconds=adapter.load_seconds, load_memory=adapter.load_memory,
                      weights={"repo_id": adapter.manifest["repo_id"], "revision": adapter.manifest["revision"],
                               "manifest_sha256": sha256(args.manifest), "sha256": adapter.manifest["weights_sha256"],
                               "verification": adapter.weight_verification})
        if args.task in {"structure", "inverse"}:
            report["reference_pdb_sha256"] = sha256(args.pdb)
        start = time.perf_counter()
        with torch.inference_mode():
            prepared = adapter.prepare(args.task, samples, args.pdb, args.mask_positions, args.steps)
        backend.synchronize()
        report["input_prepare_seconds"] = time.perf_counter() - start
        with torch.inference_mode(), backend.context():
            events = backend.events()
            backend.seed(args.seed)
            cold, first_seconds, first_gpu_ms = timed_call(backend, prepared["operation"], events)
            del cold
            report.update(first_operation_seconds=first_seconds, first_operation_gpu_ms=first_gpu_ms)
            for _ in range(args.warmup):
                backend.seed(args.seed)
                result = prepared["operation"]()
                del result
            backend.synchronize()
            backend.reset_peak()
            times, gpu_times = [], []
            for _ in range(args.runs):
                backend.seed(args.seed)
                result, seconds, milliseconds = timed_call(backend, prepared["operation"], events)
                times.append(seconds)
                if milliseconds is not None:
                    gpu_times.append(milliseconds)
                # Release all but the final result, so previous outputs don't inflate the next peak.
                if len(times) != args.runs:
                    del result
            report.update(latency_seconds=times, latency=stats(times), gpu_event_ms=gpu_times,
                          gpu_latency_ms=stats(gpu_times), memory=backend.memory(),
                          timing_scope="warmed_logits_only" if args.task == "forward" else "warmed_generate_including_SDK_encode_decode",
                          timing_exclusions=["model loading", "input prepare", "seed reset", "artifact serialization", "profiler"],
                          throughput={"proteins_per_second": len(samples) * len(times) / sum(times),
                                      "residues_per_second": sum(map(lambda s: len(s['sequence']), samples)) * len(times) / sum(times),
                                      "scope": "actual measured calls; excludes padding; no overlapping requests"})
            payload, quality = adapter.artifacts(prepared, result, output)
            report["quality"] = quality
            report["precision"] = adapter.precision_metadata()
            report["output_dtypes"] = {key: str(t.dtype) for key, t in payload.items() if isinstance(t, torch.Tensor)}
            torch.save(payload, output / "tensors.pt")
            del result
            # Distinct hot end-to-end scope: input prepare -> delivered CPU output / PDB.
            end_to_end = []
            for i in range(args.end_to_end_runs):
                destination = output / f"end_to_end_{i}"
                destination.mkdir()
                backend.seed(args.seed)
                backend.synchronize()
                start = time.perf_counter()
                request = adapter.prepare(args.task, samples, args.pdb, args.mask_positions, args.steps)
                value = request["operation"]()
                tensors, checks = adapter.artifacts(request, value, destination)
                backend.synchronize()
                end_to_end.append(time.perf_counter() - start)
                del request, value, tensors
            report.update(end_to_end_seconds=end_to_end, end_to_end=stats(end_to_end),
                          end_to_end_scope="warm input prepare + task + validation + CPU tensors; structure adds PDB export; excludes model load and .pt serialization")
            if args.profile:
                from scripts.profile_model import collect_profile
                collect_profile(adapter, prepared["operation"], output, backend)
                report["profile_artifacts"] = ["trace.json", "operators.json", "operator_shapes.json"]
            if args.microbench:
                from scripts.operator_bench import run_microbench
                if not (output / "operator_shapes.json").exists():
                    raise ValueError("--microbench requires --profile")
                run_microbench(backend, output)
                report["microbench_artifact"] = "microbench.json"
        report["status"] = "pass"
        write_json(output / "report.json", report)
        print(f"PASS {args.task} {args.precision} B={args.batch_size} {report['sample_ids']}: "
              f"median={report['latency']['median']*1000:.3f}ms allocated={report['memory']['allocated_peak_gib']}GiB; {output}", flush=True)
        return 0
    except Exception as exc:
        report.update(status="oom" if isinstance(exc, torch.cuda.OutOfMemoryError) else "fail",
                      error={"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
        if "backend" in locals():
            try:
                report["failure_memory"] = backend.memory()
            except Exception:
                pass
        write_json(output / "report.json", report)
        print(report["error"]["traceback"], flush=True)
        return 1


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=["forward", "sequence", "structure", "inverse"], default="forward")
    p.add_argument("--proteinID", "--sample-id", default="ubiquitin_human")
    p.add_argument("--sample-ids", nargs="+")
    p.add_argument("--inputs", default=str(REPO / "inputs/test_sequences.jsonl"))
    p.add_argument("--manifest", default=os.environ.get("ESM_MANIFEST", str(REPO.parent / "esm-baseline/model_snapshot.json")))
    p.add_argument("--pdb", default=str(REPO.parent / "esm-baseline/data/1UBQ.pdb"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--precision", choices=DTYPES, default="bf16")
    p.add_argument("--attention", choices=["auto", "math"], default="auto")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--runs", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--end-to-end-runs", type=int, default=3)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--mask-positions", type=int, nargs="+", default=[4, 5, 13])
    p.add_argument("--steps", type=int)
    p.add_argument("--output")
    p.add_argument("--verify-weights", action="store_true")
    p.add_argument("--profile", action="store_true")
    p.add_argument("--microbench", action="store_true")
    return p


if __name__ == "__main__":
    p = parser()
    args = p.parse_args()
    if min(args.runs, args.batch_size, args.cpu_threads) < 1 or min(args.warmup, args.end_to_end_runs) < 0 or (args.steps is not None and args.steps < 1):
        p.error("runs/batch/cpu-threads/steps must be positive; warmup/end-to-end-runs nonnegative")
    if args.microbench and not args.profile:
        p.error("--microbench requires --profile")
    raise SystemExit(run(args))
