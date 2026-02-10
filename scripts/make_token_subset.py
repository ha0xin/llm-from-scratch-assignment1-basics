#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description="Create prefix subset of uint16 token binary")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--meta-out", default="")
    args = parser.parse_args()

    src = np.memmap(args.input, dtype=np.uint16, mode="r")
    n = min(len(src), args.max_tokens)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as wf:
        np.asarray(src[:n], dtype=np.uint16).tofile(wf)

    meta = {
        "input": str(Path(args.input).resolve()),
        "output": str(out_path.resolve()),
        "input_tokens": int(len(src)),
        "output_tokens": int(n),
    }
    meta_out = args.meta_out if args.meta_out else str(out_path.with_suffix(".meta.json"))
    with open(meta_out, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(json.dumps(meta))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
