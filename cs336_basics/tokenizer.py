from collections.abc import Iterator

import regex as re


class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []

    def encode(self, text: str) -> list[int]:
        pass

    def decode(self, ids: list[int]) -> str:
        pass

    def encode_iterable(self, iterable) -> Iterator[int]:
        pass