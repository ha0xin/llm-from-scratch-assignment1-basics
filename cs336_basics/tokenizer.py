from collections.abc import Iterable, Iterator

import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []

        # Build reverse vocab: bytes -> id
        self._bytes_to_id: dict[bytes, int] = {v: k for k, v in vocab.items()}

        # Add special tokens to vocab if not already present
        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")
            if token_bytes not in self._bytes_to_id:
                new_id = len(self.vocab)
                self.vocab[new_id] = token_bytes
                self._bytes_to_id[token_bytes] = new_id

        # Build merge ranking: pair -> priority (lower is earlier/higher priority)
        self._merge_rank: dict[tuple[bytes, bytes], int] = {
            pair: i for i, pair in enumerate(merges)
        }

        # Build special tokens pattern (longer tokens first for greedy matching)
        if self.special_tokens:
            sorted_tokens = sorted(self.special_tokens, key=len, reverse=True)
            self._special_pattern = re.compile(
                "(" + "|".join(re.escape(t) for t in sorted_tokens) + ")"
            )
        else:
            self._special_pattern = None

    def _apply_bpe(self, token_bytes: bytes) -> list[bytes]:
        """Apply BPE merges to a sequence of bytes.

        Args:
            token_bytes: The bytes to tokenize

        Returns:
            List of merged byte sequences
        """
        if len(token_bytes) == 0:
            return []

        # Start with individual bytes
        tokens = [bytes([b]) for b in token_bytes]

        while len(tokens) >= 2:
            # Find the pair with lowest merge rank (highest priority)
            best_pair = None
            best_rank = float("inf")

            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                if pair in self._merge_rank:
                    rank = self._merge_rank[pair]
                    if rank < best_rank:
                        best_rank = rank
                        best_pair = pair

            if best_pair is None:
                break

            # Merge all occurrences of the best pair
            merged = best_pair[0] + best_pair[1]
            new_tokens = []
            i = 0
            while i < len(tokens):
                if (
                    i < len(tokens) - 1
                    and tokens[i] == best_pair[0]
                    and tokens[i + 1] == best_pair[1]
                ):
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return tokens

    def encode(self, text: str) -> list[int]:
        """Encode text into a sequence of token IDs.

        Args:
            text: The text to encode

        Returns:
            List of token IDs
        """
        if not text:
            return []

        ids: list[int] = []

        # Split by special tokens if any
        if self._special_pattern:
            parts = self._special_pattern.split(text)
        else:
            parts = [text]

        for part in parts:
            if not part:
                continue

            # Check if this part is a special token
            if part in self.special_tokens:
                token_bytes = part.encode("utf-8")
                ids.append(self._bytes_to_id[token_bytes])
            else:
                # Pre-tokenize with regex pattern
                for match in re.finditer(PAT, part):
                    pre_token = match.group()
                    pre_token_bytes = pre_token.encode("utf-8")

                    # Apply BPE merges
                    merged_tokens = self._apply_bpe(pre_token_bytes)

                    # Convert to IDs
                    for token in merged_tokens:
                        ids.append(self._bytes_to_id[token])

        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode a sequence of token IDs into text.

        Args:
            ids: List of token IDs

        Returns:
            Decoded text string
        """
        # Concatenate all token bytes
        all_bytes = b"".join(self.vocab[id] for id in ids)

        # Decode to string, replacing invalid sequences
        return all_bytes.decode("utf-8", errors="replace")

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Encode an iterable of strings, yielding token IDs lazily.

        This is memory-efficient for large files.

        Args:
            iterable: An iterable of strings (e.g., file handle)

        Yields:
            Token IDs one at a time
        """
        for line in iterable:
            for id in self.encode(line):
                yield id
