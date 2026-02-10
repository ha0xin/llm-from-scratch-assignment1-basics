from collections.abc import Iterable, Iterator

import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PAT_RE = re.compile(PAT)


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
        self._special_set = set(self.special_tokens)
        # Cache merged pre-token string -> token ids for much faster repeated encoding.
        self._bpe_cache: dict[str, tuple[int, ...]] = {}
        self._bpe_cache_limit = 200_000

        # Tokenizer should append them to the vocabulary if they aren’t already there
        for token in self._special_set:
            if (token.encode("utf-8") not in vocab.values()):
                vocab[len(vocab)] = token.encode("utf-8")

        # Build reverse vocab: bytes -> id
        # We assume vocab is bijective
        self._bytes_to_id: dict[bytes, int] = {v: k for k, v in vocab.items()}
        self._single_byte_ids: list[int] = [
            self._bytes_to_id[bytes([i])] for i in range(256)
        ]
        self._special_str_to_id = {
            s: self._bytes_to_id[s.encode("utf-8")] for s in self._special_set
        }

        # Build merge ranking: pair -> priority
        self._merge_rank: dict[tuple[bytes, bytes], int] = {
            pair: i for i, pair in enumerate(merges)
        }
        # Fast path for BPE over token ids: packed key = (left_id << 32) | right_id
        self._merge_rank_key: dict[int, int] = {}
        self._merge_result_id_key: dict[int, int] = {}
        for rank, (a, b) in enumerate(merges):
            left_id = self._bytes_to_id[a]
            right_id = self._bytes_to_id[b]
            key = (left_id << 32) | right_id
            self._merge_rank_key[key] = rank
            merged_id = self._bytes_to_id.get(a + b)
            if merged_id is not None:
                self._merge_result_id_key[key] = merged_id

        # Build special tokens pattern (longer tokens first for greedy matching)
        if self.special_tokens:
            sorted_tokens = sorted(self.special_tokens, key=len, reverse=True)
            self._special_pattern = re.compile(
                "(" + "|".join(re.escape(t) for t in sorted_tokens) + ")"
            )
            self._special_literals = tuple(sorted_tokens)
        else:
            self._special_pattern = None
            self._special_literals = ()

    def _split_special_parts(self, text: str):
        if self._special_pattern:
            # Avoid regex split unless the text actually contains one of the literals.
            if any(tok in text for tok in self._special_literals):
                return self._special_pattern.split(text)
        return (text,)

    def _apply_bpe_ids(self, token_bytes: bytes) -> list[int]:
        """Apply BPE merges to a pre-token and return token IDs.

        Args:
            token_bytes: The bytes to tokenize

        Returns:
            List of merged token IDs
        """
        if len(token_bytes) == 0:
            return []

        # Start with byte token ids.
        single_byte_ids = self._single_byte_ids
        tokens = [single_byte_ids[b] for b in token_bytes]
        merge_rank_key = self._merge_rank_key
        merge_result_id_key = self._merge_result_id_key

        while len(tokens) >= 2:
            # Find the pair with lowest merge rank (highest priority)
            best_key = None
            best_rank = 1 << 60

            left = tokens[0]
            for idx in range(1, len(tokens)):
                right = tokens[idx]
                key = (left << 32) | right
                rank = merge_rank_key.get(key)
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_key = key
                left = right

            if best_key is None:
                break

            # Merge all occurrences of the best pair
            merged_id = merge_result_id_key.get(best_key)
            if merged_id is None:
                break
            best_left = best_key >> 32
            best_right = best_key & 0xFFFFFFFF
            new_tokens = []
            i = 0
            while i < len(tokens):
                if (
                    i < len(tokens) - 1
                    and tokens[i] == best_left
                    and tokens[i + 1] == best_right
                ):
                    new_tokens.append(merged_id)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return tokens

    def _encode_pretoken_ids(self, pre_token: str) -> tuple[int, ...]:
        cached = self._bpe_cache.get(pre_token)
        if cached is not None:
            return cached

        pre_token_bytes = pre_token.encode("utf-8")
        token_ids = tuple(self._apply_bpe_ids(pre_token_bytes))

        if len(self._bpe_cache) >= self._bpe_cache_limit:
            self._bpe_cache.clear()
        self._bpe_cache[pre_token] = token_ids
        return token_ids
    
    def _encode_iter(self, text: str):
        """Encode text into token IDs, yielding them one at a time.
        Args:
            text: The text to encode
        Yields:
            Token IDs one at a time
        """
        if not text:
            return

        special_set = self._special_set
        special_str_to_id = self._special_str_to_id
        encode_pretoken_ids = self._encode_pretoken_ids

        for part in self._split_special_parts(text):
            if not part:
                continue

            if part in special_set:
                yield special_str_to_id[part]
                continue

            for m in PAT_RE.finditer(part):
                yield from encode_pretoken_ids(m.group())

    def _encode_to_list(self, text: str) -> list[int]:
        if not text:
            return []

        out: list[int] = []
        special_set = self._special_set
        special_str_to_id = self._special_str_to_id
        encode_pretoken_ids = self._encode_pretoken_ids

        for part in self._split_special_parts(text):
            if not part:
                continue
            if part in special_set:
                out.append(special_str_to_id[part])
                continue
            for m in PAT_RE.finditer(part):
                out.extend(encode_pretoken_ids(m.group()))
        return out

    def encode(self, text: str) -> list[int]:
        """Encode text into a sequence of token IDs.

        Args:
            text: The text to encode

        Returns:
            List of token IDs
        """
        return self._encode_to_list(text)

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Encode an iterable of strings, yielding token IDs lazily.

        This is memory-efficient for large files.

        Args:
            iterable: An iterable of strings (e.g., file handle)

        Yields:
            Token IDs one at a time
        """
        for chunk in iterable:
            yield from self._encode_iter(chunk)

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
