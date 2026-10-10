import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import importlib.metadata
import datetime
import platform
import os


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def stats(values):
    if not values:
        return None
    ordered = sorted(values)
    position = 0.9 * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    p90 = ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return {"count": len(values), "mean": mean, "median": statistics.median(values),
            "p90": p90, "stdev": std, "min": min(values), "max": max(values),
            "cv_percent": 100 * std / mean if mean else None}


def environment(repo):
    def git(*args):
        result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    versions = {}
    for name in ["torch", "esm", "numpy", "transformers", "huggingface-hub", "biotite", "torch_musa"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    source_hashes = {str(p.relative_to(repo)): sha256(p) for folder in ["common", "models", "scripts", "tests"]
                     for p in (repo / folder).glob("*.py")}
    for filename in ["main.py", "compare_outputs.py"]:
        if (repo / filename).exists():
            source_hashes[filename] = sha256(repo / filename)
    # OpenFold has an independent environment; reporting must not require ESM.
    try:
        import esm
    except ModuleNotFoundError as exc:
        if exc.name != "esm":
            raise
        esm_source, esm_commit = None, None
    else:
        esm_source = str(esm.__file__)
        esm_repo = Path(esm_source).resolve().parents[1]
        esm_result = subprocess.run(["git", "-C", str(esm_repo), "rev-parse", "HEAD"], capture_output=True, text=True)
        esm_commit = esm_result.stdout.strip() if esm_result.returncode == 0 else None
    return {"timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "host": {"platform": platform.platform(), "machine": platform.machine(),
                     "cpu_count": os.cpu_count(), "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None},
            "python": sys.version, "interpreter": sys.executable,
            "versions": versions, "repo_commit": git("rev-parse", "HEAD"),
            "repo_dirty": bool(git("status", "--porcelain", "--untracked-files=normal")),
            "source_sha256": source_hashes, "esm_source": esm_source,
            "esm_commit": esm_commit}
