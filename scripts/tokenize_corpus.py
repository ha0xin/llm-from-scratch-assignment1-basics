#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from cs336_basics.tokenizer import Tokenizer


def load_vocab(path: str | os.PathLike) -> dict[int, bytes]:
    vocab: dict[int, bytes] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            tok_id = int(obj["id"])
            vocab[tok_id] = bytes.fromhex(obj["bytes_hex"])
    return vocab


def load_merges(path: str | os.PathLike) -> list[tuple[bytes, bytes]]:
    merges: list[tuple[bytes, bytes]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            merges.append((bytes.fromhex(obj["a_hex"]), bytes.fromhex(obj["b_hex"])))
    return merges


def iter_docs(path: str | os.PathLike, delimiter: str = "<|endoftext|>", chunk_chars: int = 4 << 20):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        buf = ""
        while True:
            chunk = f.read(chunk_chars)
            if not chunk:
                break
            buf += chunk
            parts = buf.split(delimiter)
            for part in parts[:-1]:
                txt = part.strip()
                if txt:
                    yield txt
            buf = parts[-1]
        tail = buf.strip()
        if tail:
            yield tail


def tokenize_streaming(
    tokenizer: Tokenizer,
    input_path: str | os.PathLike,
    output_path: str | os.PathLike,
    delimiter: str,
    flush_every_tokens: int = 1_000_000,
) -> dict[str, float]:
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    eos_tokens = tokenizer.encode(delimiter)
    if len(eos_tokens) != 1:
        raise ValueError(
            f"Expected delimiter '{delimiter}' to map to one token, got {len(eos_tokens)}"
        )

    start = time.perf_counter()
    total_tokens = 0
    total_docs = 0
    total_bytes = 0
    buffer: list[int] = []

    with open(out_path, "wb") as wf:
        for doc in iter_docs(input_path, delimiter=delimiter):
            total_docs += 1
            total_bytes += len(doc.encode("utf-8"))
            buffer.extend(tokenizer.encode(doc))
            buffer.extend(eos_tokens)

            if len(buffer) >= flush_every_tokens:
                arr = np.asarray(buffer, dtype=np.uint16)
                arr.tofile(wf)
                total_tokens += int(arr.size)
                buffer.clear()

        if buffer:
            arr = np.asarray(buffer, dtype=np.uint16)
            arr.tofile(wf)
            total_tokens += int(arr.size)
            buffer.clear()

    elapsed = time.perf_counter() - start
    return {
        "elapsed_sec": elapsed,
        "total_tokens": float(total_tokens),
        "total_docs": float(total_docs),
        "bytes_processed": float(total_bytes),
        "tokens_per_sec": float(total_tokens / elapsed if elapsed > 0 else 0.0),
        "bytes_per_sec": float(total_bytes / elapsed if elapsed > 0 else 0.0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Streaming tokenization to uint16 binary")
    parser.add_argument("--input", required=True, help="Input text corpus path")
    parser.add_argument("--vocab", required=True, help="Tokenizer vocab jsonl path")
    parser.add_argument("--merges", required=True, help="Tokenizer merges jsonl path")
    parser.add_argument("--output", required=True, help="Output .bin path (uint16)")
    parser.add_argument("--delimiter", default="<|endoftext|>")
    parser.add_argument("--flush-every-tokens", type=int, default=1_000_000)
    parser.add_argument("--meta-out", default="", help="Optional meta json output")
    args = parser.parse_args()

    vocab = load_vocab(args.vocab)
    merges = load_merges(args.merges)
    tokenizer = Tokenizer(vocab=vocab, merges=merges, special_tokens=[args.delimiter])

    stats = tokenize_streaming(
        tokenizer=tokenizer,
        input_path=args.input,
        output_path=args.output,
        delimiter=args.delimiter,
        flush_every_tokens=args.flush_every_tokens,
    )

    meta = {
        "input": os.path.abspath(args.input),
        "vocab": os.path.abspath(args.vocab),
        "merges": os.path.abspath(args.merges),
        "output": os.path.abspath(args.output),
        "delimiter": args.delimiter,
        **stats,
    }
    meta_out = args.meta_out if args.meta_out else str(Path(args.output).with_suffix(".meta.json"))
    with open(meta_out, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(json.dumps(meta, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
