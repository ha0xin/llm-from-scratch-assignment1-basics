import os
import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

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
    pass
