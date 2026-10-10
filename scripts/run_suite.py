"""Run the CUDA reference serially with independent processes and durable artifacts."""
import argparse
import csv
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from common.reporting import write_json, sha256
from compare_outputs import compare_payloads
from common.telemetry import start_monitor, stop_monitor


def compare_case(root, name, ref_path, test_path, *, atol, rtol, keys=("logits", "embeddings"), row=None, observational=False):
    try:
        ref = torch.load(ref_path, map_location="cpu", weights_only=True)
        test = torch.load(test_path, map_location="cpu", weights_only=True)
        if row is not None:
            size = ref["logits"].shape[1]
            test = {key: value[row:row+1, :size] if isinstance(value, torch.Tensor) and value.ndim >= 2 else value for key, value in test.items()}
        value = compare_payloads(ref, test, atol=atol, rtol=rtol, keys=keys)
        if observational:
            value["criterion_status"] = value.pop("status")
            value["status"] = "observed"
            value["note"] = "Diagnostic numerical comparison; provisional tolerance, not final migration acceptance."
    except Exception as exc:
        value = {"status": "error", "error": str(exc)}
    value.update(name=name, reference=str(ref_path), candidate=str(test_path))
    write_json(root / "comparisons" / f"{name}.json", value)
    return value


def run_suite(args):
    config = json.loads(Path(args.config).read_text())
    root = Path(args.output).resolve() if args.output else REPO / "outputs" / (
        "cuda_reference_" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:6])
    root.mkdir(parents=True, exist_ok=False)
    (root / "cases").mkdir(); (root / "comparisons").mkdir(); (root / "environment").mkdir()
    write_json(root / "suite_config.json", config)
    print(f"SUITE_OUTPUT={root}", flush=True)
    unit = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], cwd=REPO, capture_output=True, text=True)
    (root / "unit_tests.log").write_text(unit.stdout + unit.stderr)
    if unit.returncode:
        raise RuntimeError("Unit checks failed; see unit_tests.log")
    for command, filename in [(["nvidia-smi"], "nvidia-smi.txt"), (["git", "diff"], "changes.patch"),
                              ([sys.executable, "-m", "pip", "freeze"], "pip-freeze.txt")]:
        completed = subprocess.run(command, cwd=REPO, capture_output=True, text=True)
        (root / "environment" / filename).write_text(completed.stdout)
    samples = [json.loads(line) for line in (REPO / "inputs/test_sequences.jsonl").read_text().splitlines()]
    jobs = []
    for sample in samples:
        id = sample["id"]
        jobs.append({"name": f"forward_{id}_b1_bf16", "task": "forward", "id": id,
                     "profile": id == "random_L256", "microbench": id == "random_L256",
                     "verify_weights": id == "ubiquitin_human"})
    for i in range(2, config["independent_ubiquitin_runs"] + 1):
        jobs.append({"name": f"forward_ubiquitin_repeat{i}", "task": "forward", "id": "ubiquitin_human"})
    for precision in ["fp32", "fp16"]:
        jobs.append({"name": f"forward_ubiquitin_{precision}", "task": "forward", "id": "ubiquitin_human", "precision": precision})
    for length in config["batch_lengths"]:
        for batch in config["batch_sizes"]:
            jobs.append({"name": f"forward_random_L{length}_b{batch}_bf16", "task": "forward", "id": f"random_L{length}", "batch": batch})
    jobs.append({"name": "forward_mixed_lengths_b2", "task": "forward", "id": "ubiquitin_human", "batch": 2,
                 "sample_ids": ["insulin_a_chain_human", "ubiquitin_human"]})
    for task in ["sequence", "structure", "inverse"]:
        jobs.append({"name": f"{task}_ubiquitin", "task": task, "id": "ubiquitin_human", "profile": task in {"structure", "inverse"}})
    jobs.append({"name": "forward_random_L256_math", "task": "forward", "id": "random_L256", "attention": "math", "profile": True})
    write_json(root / "jobs.json", jobs)
    env = dict(os.environ, HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    monitor, monitor_handles = start_monitor(config['device'], root/'environment')
    reports = {}
    suite_start = time.perf_counter()
    try:
        for i, job in enumerate(jobs, 1):
            directory = root / "cases" / job["name"]
            generation = job["task"] != "forward"
            command = [sys.executable, str(REPO / "main.py"), "--task", job["task"], "--proteinID", job["id"],
                       "--device", config["device"], "--precision", job.get("precision", config["precision"]),
                       "--batch-size", str(job.get("batch", 1)), "--runs", str(config["generation_runs"] if generation else config["runs"]),
                       "--warmup", str(config["generation_warmup"] if generation else config["warmup"]),
                       "--end-to-end-runs", str(config["end_to_end_runs"]), "--seed", str(config["seed"]),
                       "--manifest", str(Path(args.manifest).resolve()), "--pdb", str(Path(args.pdb).resolve()),
                       "--attention", job.get("attention", "auto"), "--output", str(directory)]
            if job.get("sample_ids"):
                command.extend(["--sample-ids", *job["sample_ids"]])
            for flag in ["profile", "microbench", "verify_weights"]:
                if job.get(flag):
                    command.append("--" + flag.replace("_", "-"))
            with (root / "cases" / f"{job['name']}.log").open("w") as log:
                completed = subprocess.run(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
            if (directory / "report.json").exists():
                report = json.loads((directory / "report.json").read_text())
            else:
                report = {"status": "fail", "error": {"message": f"Process exit {completed.returncode} before report creation"}}
            reports[job["name"]] = report
            write_json(root / "progress.json", {"completed": i, "total": len(jobs), "case": job["name"], "status": report["status"]})
            metric = f"{report['latency']['median']*1000:.2f} ms" if "latency" in report else report.get("error", {}).get("message", "failed")
            print(f"[{i}/{len(jobs)}] {job['name']}: {report['status']} {metric}", flush=True)
    finally:
        stop_monitor(monitor, monitor_handles)
    comparisons = []
    def tensor_path(name):
        return root / "cases" / name / "tensors.pt"
    bf16 = "forward_ubiquitin_human_b1_bf16"
    if reports[bf16]["status"] == "pass":
        if Path(args.baseline).is_file():
            comparisons.append(compare_case(root, "legacy_baseline_regression", Path(args.baseline), tensor_path(bf16), atol=0, rtol=0, keys=("logits",)))
        for i in range(2, config["independent_ubiquitin_runs"] + 1):
            name = f"forward_ubiquitin_repeat{i}"
            if reports[name]["status"] == "pass":
                comparisons.append(compare_case(root, f"independent_replay_{i}", tensor_path(bf16), tensor_path(name), atol=0, rtol=0))
    tolerance = config["observational_comparison"]
    fp32 = "forward_ubiquitin_fp32"
    if reports[fp32]["status"] == "pass":
        for name in [bf16, "forward_ubiquitin_fp16"]:
            if reports[name]["status"] == "pass":
                comparisons.append(compare_case(root, f"precision_{reports[name]['config']['precision']}_vs_fp32", tensor_path(fp32), tensor_path(name), atol=tolerance["atol"], rtol=tolerance["rtol"], observational=True))
    for length in config["batch_lengths"]:
        ref = f"forward_random_L{length}_b1_bf16"
        for batch in config["batch_sizes"]:
            name = f"forward_random_L{length}_b{batch}_bf16"
            if reports[name]["status"] == "pass" and reports[ref]["status"] == "pass":
                # Every actual row is checked against its single-sample reference.
                for row in range(batch):
                    comparisons.append(compare_case(root, f"batch_L{length}_b{batch}_row{row}", tensor_path(ref), tensor_path(name), atol=tolerance["atol"], rtol=tolerance["rtol"], row=row, observational=True))
    mixed = "forward_mixed_lengths_b2"
    if reports[mixed]["status"] == "pass":
        for row, name in enumerate(["forward_insulin_a_chain_human_b1_bf16", bf16]):
            comparisons.append(compare_case(root, f"padding_row{row}_vs_single", tensor_path(name), tensor_path(mixed), atol=tolerance["atol"], rtol=tolerance["rtol"], row=row, observational=True))
    auto = "forward_random_L256_b1_bf16"; math_case = "forward_random_L256_math"
    if reports[auto]["status"] == reports[math_case]["status"] == "pass":
        comparisons.append(compare_case(root, "attention_auto_vs_math", tensor_path(auto), tensor_path(math_case), atol=tolerance["atol"], rtol=tolerance["rtol"], observational=True))
    required_names = [f"independent_replay_{i}" for i in range(2, config["independent_ubiquitin_runs"] + 1)]
    if Path(args.baseline).is_file():
        required_names.append("legacy_baseline_regression")
    strict_results = {c["name"]: c["status"] for c in comparisons if c["status"] != "observed"}
    summary = {"schema_version": 1, "suite_root": str(root), "suite_seconds": time.perf_counter() - suite_start,
               "config": config, "unit_tests": {"passed": True, "log": "unit_tests.log"}, "cases": reports,
               "comparisons": comparisons, "case_pass": sum(r["status"] == "pass" for r in reports.values()),
               "case_total": len(jobs), "capabilities": {"real_musa": False, "openfold": False, "multi_gpu": False},
               "required_regression_names": required_names,
               "legacy_baseline_available": Path(args.baseline).is_file(),
               "required_regression_pass": all(strict_results.get(name) == "pass" for name in required_names)}
    write_json(root / "summary.json", summary)
    with (root / "metrics.csv").open("w", newline="") as handle:
        writer = csv.writer(handle);writer.writerow(["case", "status", "task", "precision", "lengths", "batch", "median_ms", "p90_ms", "proteins_s", "allocated_gib", "reserved_gib"])
        for name, report in reports.items():
            if report["status"] == "pass":
                writer.writerow([name, report["status"], report["task"], report["config"]["precision"], report["residue_lengths"], report["config"]["batch_size"], report["latency"]["median"]*1000, report["latency"]["p90"]*1000, report["throughput"]["proteins_per_second"], report["memory"]["allocated_peak_gib"], report["memory"]["reserved_peak_gib"]])
            else:
                writer.writerow([name, report["status"]])
    print(f"Completed {summary['case_pass']}/{len(jobs)}; strict regressions={summary['required_regression_pass']}; {root}", flush=True)
    return root


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(REPO / "configs/cuda_reference.json"))
    p.add_argument("--manifest", default=str(REPO.parent / "esm-baseline/model_snapshot.json"))
    p.add_argument("--pdb", default=str(REPO.parent / "esm-baseline/data/1UBQ.pdb"))
    p.add_argument("--baseline", default=str(REPO.parent / "esm-baseline/outputs/forward_check_20261006/tensors.pt"))
    p.add_argument("--output")
    args = p.parse_args()
    root = run_suite(args)
    print(root)
    if not json.loads((root / "summary.json").read_text())["required_regression_pass"]:
        raise SystemExit(1)
