"""Hydrate archived JSON/YAML paths without overwriting a local config."""

import argparse
import json
from pathlib import Path


def transform(value, paths):
    if isinstance(value, dict):
        return {k: transform(v, paths) for k, v in value.items()}
    if isinstance(value, list):
        return [transform(v, paths) for v in value]
    if isinstance(value, str):
        for name, root in paths.items():
            if value == name or value.startswith((name + "\\", name + "/")):
                tail = value[len(name):].replace("\\", "/").lstrip("/")
                return (root / tail).as_posix()
    return value


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", type=Path)
    ap.add_argument("output", type=Path)
    for name in ("nas", "realdata", "implementation"):
        ap.add_argument(f"--{name}-root", type=Path, required=True)
    args = ap.parse_args()
    roots = {f"${name.upper()}_ROOT": getattr(args, name + "_root").resolve()
             for name in ("nas", "realdata", "implementation")}
    text = args.source.read_text(encoding="utf-8-sig")
    if args.source.suffix in (".yaml", ".yml"):
        import yaml
        data = transform(yaml.safe_load(text), roots)
        output = yaml.safe_dump(data, sort_keys=False)
    else:
        output = json.dumps(transform(json.loads(text), roots), indent=2) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as file:
        file.write(output)


if __name__ == "__main__":
    main()
