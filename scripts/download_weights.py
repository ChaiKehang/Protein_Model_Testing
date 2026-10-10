"""Download an immutable official ESM3 checkpoint snapshot; no inference."""
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

REPO_ID = "biohub/esm3-sm-open-v1"
REQUIRED = (
    "esm3_sm_open_v1.pth",
    "esm3_structure_encoder_v0.pth",
    "esm3_structure_decoder_v0.pth",
    "esm3_function_decoder_v0.pth",
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    manifest_path = Path("model_snapshot.json")
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous["repo_id"] != REPO_ID:
            raise ValueError("Existing manifest belongs to a different model")
        revision = previous["revision"]
    else:
        revision = HfApi().model_info(REPO_ID).sha
    assert revision
    root = Path(snapshot_download(
        repo_id=REPO_ID, revision=revision, allow_patterns=["data/*"],
    )).resolve()
    hashes = {}
    for filename in REQUIRED:
        path = root / "data" / "weights" / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        hashes[filename] = sha256(path)
    report = {
        "repo_id": REPO_ID, "revision": revision,
        "snapshot_dir": str(root), "weights_sha256": hashes,
    }
    manifest_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("Weight snapshot ready. This did not run the model.")


if __name__ == "__main__":
    main()
