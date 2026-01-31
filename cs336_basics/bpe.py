import os
from typing import BinaryIO
from collections import Counter, defaultdict
import heapq
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

    with open(input_path, "rb") as f:
        if special_tokens:
            split_token = special_tokens[0].encode("utf-8")
            boundaries = find_chunk_boundaries(f, num_processes, split_token)
        else:
            f.seek(0, os.SEEK_END)
            file_size = f.tell()
            f.seek(0)
            chunk_size = file_size // num_processes
            boundaries = [i * chunk_size for i in range(num_processes + 1)]
            boundaries[-1] = file_size

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


class _Seq:
    def __init__(self, symbols: list[bytes], weight: int):
        self.sym = symbols
        self.weight = weight
        n = len(symbols)
        self.prev = [-1] * n
        self.next = [-1] * n
        for i in range(n):
            if i > 0:
                self.prev[i] = i - 1
            if i < n - 1:
                self.next[i] = i + 1
        self.alive = [True] * n


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    log_every: int = 0,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a BPE tokenizer on the given input corpus.

    Args:
        input_path: Path to the training corpus
        vocab_size: Total vocabulary size (including special tokens)
        special_tokens: List of special tokens to add to vocabulary
        log_every: Print progress every N merges (0 disables logging)

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

    # Build sequence objects
    sequences: list[_Seq] = []
    for token_seq, count in pre_token_counts.items():
        sequences.append(_Seq(list(token_seq), count))

    pair_counts: dict[tuple[bytes, bytes], int] = {}
    pair_occurrences: dict[tuple[bytes, bytes], set[tuple[int, int]]] = defaultdict(set)

    for seq_id, seq in enumerate(sequences):
        if len(seq.sym) < 2:
            continue
        i = 0
        while i != -1 and seq.next[i] != -1:
            j = seq.next[i]
            pair = (seq.sym[i], seq.sym[j])
            pair_counts[pair] = pair_counts.get(pair, 0) + seq.weight
            pair_occurrences[pair].add((seq_id, i))
            i = j

    # Buckets by count for correct lexicographic tie-breaking
    count_to_pairs: dict[int, set[tuple[bytes, bytes]]] = defaultdict(set)
    count_heap: list[int] = []
    for pair, count in pair_counts.items():
        count_to_pairs[count].add(pair)
    for count in count_to_pairs.keys():
        heapq.heappush(count_heap, -count)

    merges: list[tuple[bytes, bytes]] = []

    def _update_pair(pair: tuple[bytes, bytes], delta: int) -> None:
        old_count = pair_counts.get(pair, 0)
        if old_count > 0:
            bucket = count_to_pairs.get(old_count)
            if bucket is not None:
                bucket.discard(pair)
                if not bucket:
                    count_to_pairs.pop(old_count, None)

        new_count = old_count + delta
        if new_count > 0:
            pair_counts[pair] = new_count
            count_to_pairs[new_count].add(pair)
            heapq.heappush(count_heap, -new_count)
        else:
            pair_counts.pop(pair, None)

    total_merges_target = max(0, vocab_size - (len(special_tokens) + 256))
    start_time = None
    if log_every:
        import time
        start_time = time.time()

    while idx < vocab_size and count_heap:
        # Find best valid count bucket, then pick lexicographically greatest pair
        best_pair = None
        best_count = 0
        while count_heap:
            count = -heapq.heappop(count_heap)
            bucket = count_to_pairs.get(count)
            if not bucket:
                continue
            best_pair = max(bucket)
            best_count = count
            break

        if best_pair is None:
            break

        a, b = best_pair
        merged_sym = a + b
        merges.append(best_pair)
        vocab[idx] = merged_sym
        idx += 1

        # Always log the first few merges for sanity.
        if len(merges) <= 10:
            print(
                f"[train_bpe] merge#{len(merges)} pair={best_pair} "
                f"count={best_count}"
            )

        if log_every and (len(merges) % log_every == 0):
            import time
            now = time.time()
            elapsed = now - (start_time or now)
            done = len(merges)
            if done > 0 and total_merges_target > 0 and elapsed > 0:
                rate = done / elapsed
                remaining = total_merges_target - done
                eta = remaining / rate if rate > 0 else float("inf")
                print(
                    f"[train_bpe] merges={done}/{total_merges_target} "
                    f"rate={rate:.2f}/s eta={eta/60.0:.1f}m"
                )

        occ = list(pair_occurrences.get(best_pair, ()))
        pair_occurrences[best_pair].clear()

        occ_by_seq: dict[int, list[int]] = defaultdict(list)
        for seq_id, i in occ:
            occ_by_seq[seq_id].append(i)

        for seq_id, positions in occ_by_seq.items():
            seq = sequences[seq_id]
            positions.sort()
            for i in positions:
                if i == -1 or not seq.alive[i]:
                    continue
                j = seq.next[i]
                if j == -1 or not seq.alive[j]:
                    continue
                if seq.sym[i] != a or seq.sym[j] != b:
                    continue

                w = seq.weight
                left = seq.prev[i]
                right = seq.next[j]

                # Decrement old pairs
                if left != -1:
                    left_sym = seq.sym[left]
                    old_pair = (left_sym, a)
                    _update_pair(old_pair, -w)
                    pair_occurrences[old_pair].discard((seq_id, left))
                else:
                    left_sym = None

                old_pair = (a, b)
                _update_pair(old_pair, -w)

                if right != -1:
                    right_sym = seq.sym[right]
                    old_pair = (b, right_sym)
                    _update_pair(old_pair, -w)
                    pair_occurrences[old_pair].discard((seq_id, j))
                else:
                    right_sym = None

                # Merge nodes i and j
                seq.sym[i] = merged_sym
                seq.alive[j] = False
                seq.next[i] = right
                if right != -1:
                    seq.prev[right] = i

                # Add new pairs
                if left != -1 and left_sym is not None:
                    new_pair = (left_sym, merged_sym)
                    _update_pair(new_pair, w)
                    pair_occurrences[new_pair].add((seq_id, left))

                if right != -1 and right_sym is not None:
                    new_pair = (merged_sym, right_sym)
                    _update_pair(new_pair, w)
                    pair_occurrences[new_pair].add((seq_id, i))

                # Handle overlapping occurrence when a == b
                if a == b and right != -1 and right_sym == b:
                    # The pair starting at j is invalidated by this merge.
                    _update_pair(best_pair, -w)
                    pair_occurrences[best_pair].discard((seq_id, j))

        # Keep heap growth bounded by skipping when no counts left
        if best_count <= 0:
            break

    return vocab, merges
