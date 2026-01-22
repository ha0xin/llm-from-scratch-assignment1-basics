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
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
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
    if special_tokens:
        special_tokens_pattern = "|".join(re.escape(t) for t in special_tokens)
        split_token = special_tokens[0].encode("utf-8")
    else:
        special_tokens_pattern = None
        split_token = b"\n"

    # Find chunk boundaries
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, split_token)

    # Prepare tasks
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
) -> Counter:
    """Merge all occurrences of a pair in pre-token counts.

    Args:
        pre_token_counts: Counter mapping pre-token to count
        pair: The pair to merge

    Returns:
        New Counter with the pair merged
    """
    new_counts: Counter[tuple[bytes, ...]] = Counter()
    merged = pair[0] + pair[1]  # Concatenate bytes

    for token_seq, count in pre_token_counts.items():
        # Find and merge all occurrences of the pair
        new_seq = []
        i = 0
        while i < len(token_seq):
            if i < len(token_seq) - 1 and token_seq[i] == pair[0] and token_seq[i + 1] == pair[1]:
                new_seq.append(merged)
                i += 2
            else:
                new_seq.append(token_seq[i])
                i += 1
        new_counts[tuple(new_seq)] += count

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
    # Initialize vocabulary with special tokens first, then 256 byte values
    vocab: dict[int, bytes] = {}
    idx = 0

    # Add special tokens first
    for token in special_tokens:
        vocab[idx] = token.encode("utf-8")
        idx += 1

    # Add 256 byte values
    for i in range(256):
        vocab[idx] = bytes([i])
        idx += 1

    # Pre-tokenize the corpus in parallel
    pre_token_counts = _pretokenize_parallel(input_path, special_tokens)

    # Compute BPE merges
    merges: list[tuple[bytes, bytes]] = []
    num_merges = vocab_size - len(vocab)

    for _ in range(num_merges):
        # Count all pairs
        pair_counts = _get_pair_counts(pre_token_counts)

        if not pair_counts:
            break

        # Find the most frequent pair (break ties by lexicographic order)
        best_pair = max(pair_counts.keys(), key=lambda p: (pair_counts[p], p))

        # Merge the pair
        pre_token_counts = _merge_pair(pre_token_counts, best_pair)

        # Add to merges and vocabulary
        merges.append(best_pair)
        vocab[idx] = best_pair[0] + best_pair[1]
        idx += 1

    return vocab, merges
