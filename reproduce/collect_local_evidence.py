"""Copy shareable experiment evidence; hash bulk artifacts without copying them."""

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path


CAMPAIGNS = ("calibration-v1", "calibration-v2", "extended-v1")
RECOVERY = ("cuda-resume-v1", "cuda-resume-deterministic-v1", "cuda-resume-v2")
SMALL = {".json", ".jsonl", ".yaml", ".txt", ".log"}
AUDIT_OUTPUTS = {
    "codex_probes_after.txt", "e2e_cli_smoke.txt", "mutation_rereview_results.txt",
    "mutation_results.txt", "mutation_round3_results.txt", "mutation_round4_results.txt",
    "repro_after.txt", "repro_before.txt", "repro_rereview_after.txt",
    "repro_rereview_before.txt", "repro_round3_after.txt", "repro_round3_before.txt",
    "repro_round4_after.txt", "repro_round4_before.txt",
}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")


def sanitize(text, roots):
    for prefix, placeholder in roots:
        if Path(prefix).is_absolute():
            text = re.sub(re.escape(Path(prefix).as_uri()), lambda _: placeholder,
                          text, flags=re.IGNORECASE)
        escaped = prefix.replace("\\", "\\\\")
        text = re.sub(re.escape(escaped), lambda _: placeholder, text, flags=re.IGNORECASE)
        text = re.sub(re.escape(prefix), lambda _: placeholder, text, flags=re.IGNORECASE)
    # Temporary paths in old console probes are useful only as categories.
    text = re.sub(r"(?i)\$LOCAL_USER(?:\\\\|\\)AppData(?:\\\\|\\)Local(?:\\\\|\\)Temp",
                  "$TMP", text)
    return text


def copy_text(src, dest, roots):
    clean = sanitize(src.read_text(encoding="utf-8-sig", errors="strict"), roots)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(clean, encoding="utf-8", newline="\n")


def validate_checkpoint(path, file_hashes):
    meta = json.loads((path / "checkpoint.json").read_text(encoding="utf-8"))
    actual = {name: file_hashes[name] for name in (
        "model.safetensors", "optimizer.pt", "rng.pt", "scheduler.pt")}
    if "scaler.pt" in file_hashes:
        actual["scaler.pt"] = file_hashes["scaler.pt"]
    declared = meta.get("files_sha256")
    if declared is not None and declared != actual:
        raise ValueError(f"Checksum mismatch: {path}")
    return {"step": meta["step"], "tokens_seen": meta["tokens_seen"],
            "declared_manifest_matched": True if declared is not None else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--implementation", type=Path, required=True)
    ap.add_argument("--nas-root", type=Path, required=True)
    ap.add_argument("--research", type=Path, required=True)
    ap.add_argument("--skip-evaluations", action="store_true",
                    help="collect before new validation output is available")
    args = ap.parse_args()
    data, impl, out = (p.resolve() for p in (args.data, args.implementation, args.research))
    roots = [(str(impl / ".venv-cuda" / "Scripts" / "python.exe"), "$CUDA_PYTHON"),
             (str(data), "$REALDATA_ROOT"), (str(impl), "$IMPLEMENTATION_ROOT"),
             (str(args.nas_root), "$NAS_ROOT"), (str(Path.home()), "$LOCAL_USER")]
    copies = out / "experiments/data/archive"
    for campaign in CAMPAIGNS + RECOVERY:
        for fh in sorted((data / campaign).rglob("*")):
            if not fh.is_file() or fh.suffix not in SMALL:
                continue
            if "checkpoints" in fh.relative_to(data).parts and fh.name != "latest.json":
                continue  # checkpoint JSON is retained separately with the full file manifest
            if args.skip_evaluations and fh.name.startswith("evals-step-"):
                continue
            copy_text(fh, copies / fh.relative_to(data), roots)
    for name in ("extended-v1.stdout.log", "extended-v1.stderr.log", "launch-extended-v1.py"):
        copy_text(data / name, copies / "extended-v1" / name, roots)
    for src in sorted((data / "specs").glob("*.json")):
        copy_text(src, out / "datasets/data/raw-specifications" / src.name, roots)
    for name in sorted(AUDIT_OUTPUTS):
        src = impl / "post_audit" / name
        if src.is_file():
            copy_text(src, copies / "cpu-audit" / name, roots)
    for src in sorted((impl / "experiments/DML-0001").iterdir()):
        if src.is_file():
            # Files are already published by the implementation repo. Preserve original bytes.
            dest = copies / "DML-0001" / src.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(src.read_bytes())

    files = []
    checkpoints = []
    for campaign in CAMPAIGNS + RECOVERY:
        dirs = sorted((data / campaign).rglob("checkpoint.json"))
        for cj in dirs:
            folder = cj.parent
            inventory = {fh.name: digest(fh) for fh in sorted(folder.iterdir()) if fh.is_file()}
            record = validate_checkpoint(folder, inventory)
            relative = folder.relative_to(data).as_posix()
            checkpoints.append({"path": relative, **record})
            copy_dest = copies / "checkpoint-metadata" / cj.relative_to(data)
            copy_text(cj, copy_dest, roots)
            for name, h in inventory.items():
                artifact = {"path": f"{relative}/{name}", "sha256": h,
                            "bytes": (folder / name).stat().st_size,
                            "category": "operator-local NVMe checkpoint",
                            "published": name == "checkpoint.json"}
                if name == "checkpoint.json":
                    artifact.update(publication="path-sanitized",
                                    public_file=copy_dest.relative_to(out).as_posix(),
                                    public_sha256=digest(copy_dest))
                files.append(artifact)
    for path in sorted((data / "packed").glob("*/meta.json")):
        meta = json.loads(path.read_text(encoding="utf-8"))
        declared = meta["packed_sha256"]
        for key, name in meta["files"].items():
            fh = Path(name)
            h = digest(fh)
            if h != declared[key]:
                raise ValueError(f"Packed dataset checksum mismatch: {fh}")
            if key.endswith("_bin"):
                split = key.removesuffix("_bin")
                if fh.stat().st_size != 2 * meta["tokens"][split]:
                    raise ValueError(f"Packed file wrong size: {fh}")
            files.append({"path": fh.relative_to(data).as_posix(), "sha256": h,
                          "bytes": fh.stat().st_size,
                          "category": "operator-local NVMe prepared dataset", "published": False})
        tok = Path(meta["tokenizer"]["path"])
        h = digest(tok)
        if h != meta["tokenizer"]["sha256"]:
            raise ValueError(f"Tokenizer hash mismatch: {tok}")
        files.append({"path": tok.relative_to(data).as_posix(), "sha256": h,
                      "bytes": tok.stat().st_size,
                      "category": "operator-local NVMe tokenizer", "published": False})
        for name in ("meta.json", "source-manifest.json"):
            copy_text(path.parent / name, out / "datasets/data/published-packing" / path.parent.name / name,
                      roots)
    auxiliary = sorted((data / "packed").glob("*/*.ends"))
    auxiliary += sorted((data / "packed").glob("*.staging/documents.sqlite"))
    auxiliary += sorted((data / "packed").glob("*.staging/tokenizer-sample.txt"))
    for path in auxiliary:
        files.append({"path": path.relative_to(data).as_posix(), "sha256": digest(path),
                      "bytes": path.stat().st_size,
                      "category": "operator-local NVMe preparation intermediate", "published": False})
    for src in sorted((impl / "experiments/DML-0001").iterdir()):
        if src.is_file():
            files.append({"path": f"DML-0001/{src.name}", "sha256": digest(src),
                          "bytes": src.stat().st_size, "category": "historical, Git tracked",
                          "published": True})
    dest = out / "provenance/artifact-hashes/2026-11-24-local.json"
    dump(dest, {"captured_utc": datetime.now(timezone.utc).isoformat(),
                "path_base": "$REALDATA_ROOT for current artifacts; historical files under implementation experiments/",
                "access": "Bulk files require project-operator access or independent reconstruction from pinned inputs.",
                "record_count": len(files), "checkpoint_count": len(checkpoints),
                "checkpoints": checkpoints, "files": files})
    print(f"Collected shareable logs and hashed {len(files)} files in {len(checkpoints)} checkpoint directories.")


if __name__ == "__main__":
    main()
