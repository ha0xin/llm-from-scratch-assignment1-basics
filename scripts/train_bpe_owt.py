#!/usr/bin/env python3
import argparse
import json
import os
import time
import resource
from pathlib import Path

from cs336_basics.bpe import train_bpe


def _bytes_preview(b: bytes, max_len: int = 40) -> str:
    if not b:
        return ""
    try:
        s = b.decode("utf-8")
    except Exception:
        s = b.decode("utf-8", errors="replace")
    if len(s) > max_len:
        return s[:max_len] + "..."
    return s


def _serialize_vocab(vocab: dict[int, bytes], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        for idx in sorted(vocab.keys()):
            b = vocab[idx]
            rec = {
                "id": idx,
                "bytes_hex": b.hex(),
                "utf8": _bytes_preview(b),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _serialize_merges(merges: list[tuple[bytes, bytes]], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        for i, (a, b) in enumerate(merges):
            rec = {
                "rank": i,
                "a_hex": a.hex(),
                "b_hex": b.hex(),
                "a_utf8": _bytes_preview(a),
                "b_utf8": _bytes_preview(b),
                "merged_utf8": _bytes_preview(a + b),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _max_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Train BPE tokenizer on OpenWebText")
    parser.add_argument(
        "--input",
        default="/data/share/hw1-data/owt_train.txt",
        help="Path to OpenWebText training text",
    )
    parser.add_argument("--vocab-size", type=int, default=32_000)
    parser.add_argument(
        "--special-token",
        action="append",
        default=["<|endoftext|>"],
        help="Special token(s) to add; can be repeated",
    )
    parser.add_argument(
        "--out-dir",
        default="artifacts/bpe/owt",
        help="Output directory",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=1000,
        help="Print progress every N merges (0 disables)",
    )

    args = parser.parse_args()
    # Deduplicate while preserving order (avoid duplicates from defaults + CLI)
    seen = set()
    special_tokens = []
    for tok in args.special_token:
        if tok not in seen:
            special_tokens.append(tok)
            seen.add(tok)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[train_bpe] start input={args.input} vocab_size={args.vocab_size} "
        f"special_tokens={special_tokens} log_every={args.log_every}"
    )

    t0 = time.time()
    vocab, merges = train_bpe(
        input_path=args.input,
        vocab_size=args.vocab_size,
        special_tokens=special_tokens,
        log_every=args.log_every,
    )
    elapsed = time.time() - t0
    max_rss_mb = _max_rss_mb()

    _serialize_vocab(vocab, out_dir / "vocab.jsonl")
    _serialize_merges(merges, out_dir / "merges.jsonl")

    meta = {
        "input": os.path.abspath(args.input),
        "vocab_size": args.vocab_size,
        "special_tokens": special_tokens,
        "elapsed_sec": elapsed,
        "max_rss_mb": max_rss_mb,
    }
    with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("Saved:", out_dir / "vocab.jsonl")
    print("Saved:", out_dir / "merges.jsonl")
    print("Saved:", out_dir / "meta.json")
    print(f"Elapsed: {elapsed/60.0:.2f} min | Max RSS: {max_rss_mb:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
