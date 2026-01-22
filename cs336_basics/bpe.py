import os
from typing import BinaryIO
from collections import Counter
from multiprocessing import Pool, cpu_count

import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Compute byte offsets that split a file into chunks suitable for independent
    processing (e.g., parallel pre-tokenization).

    The returned boundaries are byte indices into the file. The first boundary
    is always 0, and the last boundary is always the file size. For each interior
    boundary, the function starts from an initial uniform split point and scans
    forward until it finds the next occurrence of `split_special_token`. The
    boundary is then set to the start of that special token, ensuring that no
    chunk begins in the middle of a special token or document boundary.

    As a result, except for the first chunk, all chunks begin at the start of
    `split_special_token` (when such a token is found). If boundaries overlap,
    fewer than `desired_num_chunks` boundaries may be returned.

    Args:
        file: An open binary file handle.
        desired_num_chunks: The target number of chunks to split the file into.
        split_special_token: A byte string representing a special token (e.g.,
            b"<|endoftext|>") that must not be split across chunks.

    Returns:
        A sorted list of unique byte offsets. Consecutive offsets define
        half-open intervals [start, end) corresponding to file chunks.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def _process_chunk(args: tuple) -> Counter:
    """Process a single chunk and return pre-token counts.

    Args:
        args: Tuple of (input_path, start, end, special_tokens_pattern)

    Returns:
        Counter mapping pre-token (as tuple of bytes) to count
    """
    input_path, start, end, special_tokens_pattern = args

    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    # Split by special tokens to prevent merging across document boundaries
    if special_tokens_pattern:
        parts = re.split(special_tokens_pattern, chunk)
    else:
        parts = [chunk]

    # Pre-tokenize each part and count
    counts: Counter[tuple[bytes, ...]] = Counter()
    for part in parts:
        for match in re.finditer(PAT, part):
            token = match.group()
            # Convert to tuple of single bytes
            token_bytes = tuple(bytes([b]) for b in token.encode("utf-8"))
            counts[token_bytes] += 1

    return counts


def _pretokenize_parallel(
    input_path: str | os.PathLike,
    special_tokens: list[str],
    num_processes: int | None = None,
) -> Counter:
    """Parallel pre-tokenization of input file.

    Args:
        input_path: Path to input file
        special_tokens: List of special tokens
        num_processes: Number of processes to use (default: cpu_count)

    Returns:
        Counter mapping pre-token (as tuple of bytes) to count
    """
    if num_processes is None:
        num_processes = cpu_count()

    # Build special tokens pattern for splitting
    special_tokens_pattern = "|".join(re.escape(t) for t in special_tokens)
    split_token = special_tokens[0].encode("utf-8") # ???

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, split_token)

    tasks = [
        (str(input_path), start, end, special_tokens_pattern)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]

    # Process chunks in parallel
    with Pool(num_processes) as pool:
        results = pool.map(_process_chunk, tasks)

    # Merge all counts
    total_counts: Counter[tuple[bytes, ...]] = Counter()
    for counts in results:
        total_counts.update(counts)

    return total_counts


def _get_pair_counts(pre_token_counts: Counter) -> Counter:
    """Count all adjacent pairs across all pre-tokens.

    Args:
        pre_token_counts: Counter mapping pre-token to count

    Returns:
        Counter mapping (token1, token2) pair to count
    """
    pair_counts: Counter[tuple[bytes, bytes]] = Counter()
    for token_seq, count in pre_token_counts.items():
        for i in range(len(token_seq) - 1):
            pair = (token_seq[i], token_seq[i + 1])
            pair_counts[pair] += count
    return pair_counts


def _merge_pair(
    pre_token_counts: Counter,
    pair: tuple[bytes, bytes],
    pair_counts: Counter,
) -> Counter:
    """Merge all occurrences of a pair in pre-token counts.

    Also incrementally updates pair_counts.

    Args:
        pre_token_counts: Counter mapping pre-token to count
        pair: The pair to merge
        pair_counts: Counter of pair frequencies (modified in place)

    Returns:
        New Counter with the pair merged
    """
    new_counts: Counter[tuple[bytes, ...]] = Counter()
    a, b = pair
    merged = a + b

    for token_seq, count in pre_token_counts.items():
        # 懒构造：只有真的发生合并才创建 new_seq
        new_seq = None  # None 表示“目前还没发生合并”
        i = 0
        n = len(token_seq)

        while i < n:
            if i + 1 < n and token_seq[i] == a and token_seq[i + 1] == b:
                # 第一次命中复制前缀
                if new_seq is None:
                    new_seq = list(token_seq[:i])
                new_seq.append(merged)
                i += 2
            else:
                if new_seq is not None:
                    new_seq.append(token_seq[i])
                i += 1

        # 没发生合并：序列不变，pair_counts 也不变
        if new_seq is None:
            new_counts[token_seq] += count
            continue

        new_seq_tuple = tuple(new_seq)
        new_counts[new_seq_tuple] += count

        # 发生合并：撤销所有旧序列 pair 贡献的 pair_counts
        for j in range(len(token_seq) - 1):
            old_pair = (token_seq[j], token_seq[j + 1])
            pair_counts[old_pair] -= count

        # 加回新序列贡献的 pair_counts，显然这里可以进一步优化
        for j in range(len(new_seq) - 1):
            new_pair = (new_seq[j], new_seq[j + 1])
            pair_counts[new_pair] += count

    # 清理掉 <= 0 的项，实际上不会有负数，只会是 0
    for p in list(pair_counts.keys()):
        if pair_counts[p] <= 0:
            print("Removing pair with non-positive count:", p, pair_counts[p])
            del pair_counts[p]

    return new_counts


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a BPE tokenizer on the given input corpus.

    Args:
        input_path: Path to the training corpus
        vocab_size: Total vocabulary size (including special tokens)
        special_tokens: List of special tokens to add to vocabulary

    Returns:
        vocab: Mapping from token ID to token bytes
        merges: List of merge operations in order
    """
    vocab: dict[int, bytes] = {}
    idx = 0

    for token in special_tokens:
        vocab[idx] = token.encode("utf-8")
        idx += 1

    for i in range(256):
        vocab[idx] = bytes([i])
        idx += 1

    # Pre-tokenize the corpus in parallel
    pre_token_counts = _pretokenize_parallel(input_path, special_tokens)

    # Build initial pair counts
    pair_counts = _get_pair_counts(pre_token_counts)

    # Compute BPE merges
    merges: list[tuple[bytes, bytes]] = []

    while idx < vocab_size:
        if not pair_counts:
            break

        # Find the most frequent pair (break ties by lexicographic order)
        best_pair = max(pair_counts.keys(), key=lambda p: (pair_counts[p], p))

        pre_token_counts = _merge_pair(pre_token_counts, best_pair, pair_counts)

        # Add to merges and vocabulary
        merges.append(best_pair)
        vocab[idx] = best_pair[0] + best_pair[1]
        idx += 1

    return vocab, merges
