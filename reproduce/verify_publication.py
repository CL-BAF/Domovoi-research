"""Validate the Domovoi publication from its public machine-readable evidence."""

import ast
import hashlib
import math
import re
import statistics
from pathlib import Path
from urllib.parse import unquote

from publication_data import ARCHIVE, ROOT, differences, load_campaign, read, records


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check_links_and_public_paths():
    count = 0
    patterns = (r"(?i)[a-z]:[/\\]+users[/\\]", r"(?i)file:///[a-z]:",
                r"(?:\\\\|//)10\.\d+\.\d+\.\d+", r"(?i)\bgithub_pat_[a-z0-9_]{30,}\b",
                r"(?i)\bgh[pousr]_[a-z0-9]{30,}\b", r"\bsk-[a-zA-Z0-9_-]{30,}\b")
    for path in ROOT.rglob("*"):
        rel = path.relative_to(ROOT)
        if any(part in (".git", "__pycache__", ".venv", ".pytest_cache") for part in rel.parts):
            continue
        if not path.is_file() or path.suffix not in (".md", ".py", ".json", ".jsonl", ".txt",
                                                      ".csv", ".yaml", ".log", ".svg"):
            continue
        assert path.stat().st_size < 2 * 2**20, f"Oversized research file {rel}"
        content = path.read_text(encoding="utf-8-sig")
        if path.name != Path(__file__).name:
            assert not any(re.search(p, content) for p in patterns), f"Private path or token: {rel}"
        if path.suffix == ".py":
            ast.parse(content, filename=str(rel))
        if path.suffix != ".md":
            continue
        in_code = False
        for line in content.splitlines():
            if line.lstrip().startswith('```'):
                in_code = not in_code
            elif not in_code:
                for match in re.finditer(r"\]\(([^()\s]+)\)", line):
                    link = unquote(match[1].split("#", 1)[0])
                    if not link or re.match(r"^[a-z][a-z+.-]*:", link, flags=re.I):
                        continue
                    assert (path.parent / link).resolve().exists(), f"{rel}: missing {link}"
                    count += 1
    return count


def check_artifact_hashes():
    manifest = read(ROOT / "provenance/artifact-hashes/2026-11-24-local.json")
    assert len(manifest["files"]) == manifest["record_count"]
    assert len(manifest["checkpoints"]) == manifest["checkpoint_count"]
    publish_checked = 0
    for entry in manifest["files"]:
        assert re.fullmatch("[0-9a-f]{64}", entry["sha256"])
        if "public_file" in entry:
            assert sha(ROOT / entry["public_file"]) == entry["public_sha256"]
            publish_checked += 1
        elif entry["path"].startswith("DML-0001/"):
            assert sha(ARCHIVE / entry["path"]) == entry["sha256"]
            publish_checked += 1
    extended = [c for c in manifest["checkpoints"] if c["path"].startswith("extended-v1/")]
    assert len(extended) == 48 and all(c["declared_manifest_matched"] for c in extended)
    assert len([c for c in extended if c["tokens_seen"] == 100007936]) == 3
    return manifest["record_count"], publish_checked


def check_inputs():
    inventory = read(ROOT / "datasets/data/NAS-INVENTORY-2026-11-24.json")
    expected = {"TinyStories": (2, 1943728838), "WikiText-103": (4, 315466397),
                "FineWeb-Edu": (14, 28518193415)}
    for name, item in inventory.items():
        assert not (item["temporary"] or item["missing"])
        files = list(item["files"].values())
        assert (len(files), sum(f["bytes"] for f in files)) == expected[name]
        assert all(f["hash_matches_etag"] and f["sha256"] == f["upstream_etag"] for f in files)
    packed = read(ROOT / "datasets/data/PACKED-MANIFEST-2026-11-24.json")
    local = read(ROOT / "provenance/artifact-hashes/2026-11-24-local.json")
    hashes = {f["path"]: f["sha256"] for f in local["files"]}
    counts = {"tinystories-v1": (110052057, 4805339, 2018159),
              "wikitext103-v1": (152441047, 322861, 368001),
              "fineweb-edu-preview-v1": (5044464, 371484, 413911)}
    for name, old in packed.items():
        archived = read(ROOT / "datasets/data/published-packing" / name / "meta.json")
        assert archived["tokens"] == old["tokens"]
        assert archived["packed_sha256"] == old["packed_sha256"]
        assert tuple(archived["tokens"][s] for s in ("train", "val", "test")) == counts[name]
        for key, h in archived["packed_sha256"].items():
            raw_name = archived["files"][key].replace("\\", "/").rsplit("/", 1)[-1]
            assert hashes[f"packed/{name}/{raw_name}"] == h
        assert hashes[f"packed/{name}/tokenizer.json"] == archived["tokenizer"]["sha256"]


def check_comparisons():
    first, deterministic, extended = (load_campaign(n)
                                      for n in ("calibration-v1", "calibration-v2", "extended-v1"))
    assert all(run["summary"]["tokens_trained"] == 3276800 for run in first.values())
    published = read(ROOT / "experiments/data/TINYSTORIES-CALIBRATION-v2.json")["campaign"]
    for name, run in deterministic.items():
        expect = published[name]
        assert run["step_rate"] == expect["median_synchronized_tokens_per_s"]
        assert run["eval"][200]["nll"] == expect["full_validation"]["nll"]
        assert run["summary"] == expect["summary"]
        assert run["meta"]["git_commit"] == "d914684dc730cd32132aeff389d6291c447ed7e4"
    assert all(run["summary"]["tokens_trained"] == 100007936 for run in extended.values())
    assert all(run["meta"]["git_commit"].startswith("d914684") for run in extended.values())
    assert all(run["meta"]["git_commit"].startswith("f5fac7a") for run in first.values())
    a = differences(deterministic, 200)
    b = differences(extended, 6104)
    assert math.isclose(a["b08_minus_r02x4_peak_mib"], 406.33)
    assert math.isclose(b["fraction_of_b08_gain_over_b02_recovered"], .28582, abs_tol=.00005)
    historic = records(ARCHIVE / "DML-0001/metrics.jsonl")
    assert len(historic) == 52 and historic[-1]["step"] == 520
    assert statistics.median(row["tok_per_s"] for row in historic) == 17402.05
    resume = read(ARCHIVE / "cuda-resume-v2/verification.json")
    assert resume["max_final_weight_delta"] == 0 and resume["weights_and_adamw_restore_exact"]
    assert resume["reconciled_steps"] == [1, 2, 3, 4] and resume["final_sampler_position"] == 128
    return {"calibration_200": a, "extended_6104": b}


if __name__ == "__main__":
    import json

    check_inputs()
    stored, published = check_artifact_hashes()
    print("Verified", stored, "manifest records and", published, "published artifact checksums.")
    print(json.dumps(check_comparisons(), indent=2))
    print("Verified", check_links_and_public_paths(), "relative links; private path scan passed.")
