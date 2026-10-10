"""Add a second real protein to all four tasks, preserving the original suite."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from common.reporting import write_json, sha256


def run(root, pdb):
    source = json.loads((root / "summary.json").read_text())
    config = source["config"]
    output = root / "quality_extension"
    output.mkdir(exist_ok=False)
    reports = {}
    env = dict(os.environ, HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    inputs = REPO / "inputs/task_quality_sequences.jsonl"
    for task in ["forward", "sequence", "structure", "inverse"]:
        directory = output / task
        command = [sys.executable, str(REPO / "main.py"), "--task", task,
                   "--proteinID", "crambin_1CRN_A", "--inputs", str(inputs), "--pdb", str(pdb),
                   "--device", config["device"], "--precision", config["precision"], "--seed", str(config["seed"]),
                   "--runs", str(config["runs"] if task == "forward" else config["generation_runs"]),
                   "--warmup", str(config["warmup"] if task == "forward" else config["generation_warmup"]),
                   "--end-to-end-runs", str(config["end_to_end_runs"]), "--output", str(directory)]
        with (output / f"{task}.log").open("w") as log:
            subprocess.run(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
        report = json.loads((directory / "report.json").read_text())
        reports[task] = report
        print(f"Crambin {task}: {report['status']} " + (f"{report['latency']['median']*1000:.2f} ms" if 'latency' in report else report['error']['message']), flush=True)
        write_json(output / "progress.json", {"completed": len(reports), "total": 4})
    write_json(root / "quality_extension.json", {"schema_version": 1, "cases": reports,
               "case_pass": sum(r["status"] == "pass" for r in reports.values()), "case_total": 4,
               "reference": {"url": "https://files.rcsb.org/download/1CRN.pdb", "chain": "A", "sha256": sha256(pdb)},
               "limits": "Two real proteins across four tasks including original ubiquitin; same seed for measured generation; still not population accuracy or diverse-design evaluation"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--pdb", default=str(REPO.parent / "esm-baseline/data/1CRN.pdb"))
    args = parser.parse_args()
    run(Path(args.suite).resolve(), Path(args.pdb).resolve())
