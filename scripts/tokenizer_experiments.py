#!/usr/bin/env python3
import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np

from cs336_basics.tokenizer import Tokenizer


def _load_vocab(path: str) -> dict[int, bytes]:
    vocab: dict[int, bytes] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            tok_id = int(obj["id"])
            vocab[tok_id] = bytes.fromhex(obj["bytes_hex"])
    return vocab


def _load_merges(path: str) -> list[tuple[bytes, bytes]]:
    merges: list[tuple[bytes, bytes]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            a = bytes.fromhex(obj["a_hex"])
            b = bytes.fromhex(obj["b_hex"])
            merges.append((a, b))
    return merges


def _iter_docs(path: str, delimiter: str = "<|endoftext|>", chunk_size: int = 4 << 20):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        buf = ""
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            buf += chunk
            parts = buf.split(delimiter)
            for part in parts[:-1]:
                doc = part.strip()
                if doc:
                    yield doc
            buf = parts[-1]
        tail = buf.strip()
        if tail:
            yield tail


def _reservoir_sample(iterable, k: int, seed: int = 0):
    rng = random.Random(seed)
    sample = []
    n = 0
    for n, item in enumerate(iterable, 1):
        if n <= k:
            sample.append(item)
        else:
            j = rng.randrange(n)
            if j < k:
                sample[j] = item
    return sample, n


def _compression_ratio(docs: list[str], tokenizer: Tokenizer) -> float:
    total_bytes = 0
    total_tokens = 0
    for doc in docs:
        total_bytes += len(doc.encode("utf-8"))
        total_tokens += len(tokenizer.encode(doc))
    return total_bytes / max(total_tokens, 1)


def _read_text_chunks(path: str, max_bytes: int, chunk_size: int = 1 << 20):
    chunks = []
    total = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        while total < max_bytes:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk.encode("utf-8"))
    return chunks, total


def _measure_throughput(tokenizer: Tokenizer, chunks: list[str], total_bytes: int):
    start = time.perf_counter()
    token_count = 0
    for _tok in tokenizer.encode_iterable(chunks):
        token_count += 1
    elapsed = time.perf_counter() - start
    bps = total_bytes / elapsed if elapsed > 0 else float("inf")
    return {
        "bytes": total_bytes,
        "tokens": token_count,
        "seconds": elapsed,
        "bytes_per_sec": bps,
    }


def _tokenize_to_uint16(
    tokenizer: Tokenizer,
    input_path: str,
    output_path: str,
    delimiter: str = "<|endoftext|>",
):
    # Two-pass: count tokens, then write uint16 array.
    def _token_iter():
        for doc in _iter_docs(input_path, delimiter=delimiter):
            yield from tokenizer.encode(doc)
            # Append delimiter between docs
            for _ in tokenizer.encode(delimiter):
                yield _

    count = 0
    for _ in _token_iter():
        count += 1

    arr = np.memmap(output_path, dtype=np.uint16, mode="w+", shape=(count,))
    i = 0
    for tok in _token_iter():
        arr[i] = tok
        i += 1
    arr.flush()
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description="Tokenizer experiments (2.7)")
    parser.add_argument("--tinystories-vocab", default="artifacts/bpe/tinystories/vocab.jsonl")
    parser.add_argument("--tinystories-merges", default="artifacts/bpe/tinystories/merges.jsonl")
    parser.add_argument("--owt-vocab", default="artifacts/bpe/owt/vocab.jsonl")
    parser.add_argument("--owt-merges", default="artifacts/bpe/owt/merges.jsonl")
    parser.add_argument("--tinystories-data", default="/data/share/hw1-data/TinyStoriesV2-GPT4-train.txt")
    parser.add_argument("--owt-data", default="/data/share/hw1-data/owt_train.txt")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--throughput-bytes", type=int, default=50_000_000)
    parser.add_argument("--out-dir", default="artifacts/experiments")
    parser.add_argument("--tokenize-datasets", action="store_true")
    parser.add_argument("--tinystories-valid", default="/data/share/hw1-data/TinyStoriesV2-GPT4-valid.txt")
    parser.add_argument("--owt-valid", default="/data/share/hw1-data/owt_valid.txt")
    parser.add_argument("--tokenized-out-dir", default="artifacts/tokenized")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ts_vocab = _load_vocab(args.tinystories_vocab)
    ts_merges = _load_merges(args.tinystories_merges)
    owt_vocab = _load_vocab(args.owt_vocab)
    owt_merges = _load_merges(args.owt_merges)

    special = ["<|endoftext|>"]
    ts_tok = Tokenizer(ts_vocab, ts_merges, special_tokens=special)
    owt_tok = Tokenizer(owt_vocab, owt_merges, special_tokens=special)

    ts_docs, ts_total = _reservoir_sample(
        _iter_docs(args.tinystories_data), args.samples, seed=args.seed
    )
    owt_docs, owt_total = _reservoir_sample(
        _iter_docs(args.owt_data), args.samples, seed=args.seed
    )

    ts_ratio = _compression_ratio(ts_docs, ts_tok)
    owt_ratio = _compression_ratio(owt_docs, owt_tok)
    owt_with_ts_ratio = _compression_ratio(owt_docs, ts_tok)

    chunks, total_bytes = _read_text_chunks(args.owt_data, args.throughput_bytes)
    throughput = _measure_throughput(owt_tok, chunks, total_bytes)

    pile_bytes = int(825 * (1024**3))
    pile_seconds = pile_bytes / throughput["bytes_per_sec"]

    results = {
        "samples": args.samples,
        "seed": args.seed,
        "tinystories_total_docs": ts_total,
        "owt_total_docs": owt_total,
        "compression_ratio_ts": ts_ratio,
        "compression_ratio_owt": owt_ratio,
        "compression_ratio_owt_with_ts": owt_with_ts_ratio,
        "throughput": throughput,
        "pile_seconds_est": pile_seconds,
        "pile_days_est": pile_seconds / 86400.0,
        "tokenize_datasets": bool(args.tokenize_datasets),
    }

    with open(out_dir / "tokenizer_experiments.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    if args.tokenize_datasets:
        tok_out = Path(args.tokenized_out_dir)
        tok_out.mkdir(parents=True, exist_ok=True)

        ts_train_out = tok_out / "tinystories_train_uint16.dat"
        ts_valid_out = tok_out / "tinystories_valid_uint16.dat"
        owt_train_out = tok_out / "owt_train_uint16.dat"
        owt_valid_out = tok_out / "owt_valid_uint16.dat"

        results["tinystories_train_tokens"] = _tokenize_to_uint16(
            ts_tok, args.tinystories_data, str(ts_train_out)
        )
        results["tinystories_valid_tokens"] = _tokenize_to_uint16(
            ts_tok, args.tinystories_valid, str(ts_valid_out)
        )
        results["owt_train_tokens"] = _tokenize_to_uint16(
            owt_tok, args.owt_data, str(owt_train_out)
        )
        results["owt_valid_tokens"] = _tokenize_to_uint16(
            owt_tok, args.owt_valid, str(owt_valid_out)
        )

        with open(out_dir / "tokenizer_experiments.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    md = out_dir / "tokenizer_experiments.md"
    with open(md, "w", encoding="utf-8") as f:
        f.write("# Tokenizer Experiments\n\n")
        f.write(
            f"(a) Compression ratio (bytes/token) on 10 sampled docs: "
            f"TinyStories tokenizer on TinyStories = {ts_ratio:.3f}; "
            f"OWT tokenizer on OWT = {owt_ratio:.3f}.\n\n"
        )
        f.write(
            f"(b) OWT sample tokenized with TinyStories tokenizer: "
            f"{owt_with_ts_ratio:.3f} bytes/token, which is lower than the OWT "
            f"tokenizer (more tokens per byte), as expected for a smaller, "
            f"less matched vocab.\n\n"
        )
        f.write(
            f"(c) Throughput (tokenizer only, in-memory): "
            f"{throughput['bytes_per_sec']:.2f} bytes/sec on "
            f"{throughput['bytes']/1e6:.1f} MB sample. "
            f"Estimated time to tokenize 825GB (Pile) ≈ "
            f"{pile_seconds/3600.0:.1f} hours (~{pile_seconds/86400.0:.1f} days).\n\n"
        )
        f.write(
            "(d) uint16 is appropriate because vocab sizes are <= 32K, so all "
            "token IDs fit in 16 bits (0..65535), which halves storage vs int32.\n"
        )

        if args.tokenize_datasets:
            f.write("\nTokenized datasets saved as uint16 memmaps in ")
            f.write(str(args.tokenized_out_dir))
            f.write(".\n")
        else:
            f.write(
                "\nDataset tokenization not run in this script invocation. "
                "Use --tokenize-datasets to produce uint16 files.\n"
            )

    print(f"Wrote results: {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
