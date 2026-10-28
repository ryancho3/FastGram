"""Fast, corpus-independent modified Kneser–Ney language models."""
from .model import NGramModel
from ._kernels import NUMBA_AVAILABLE
from .corpus import iter_jsonl, iter_lines, profile

__all__ = ["NGramModel", "NUMBA_AVAILABLE", "iter_jsonl", "iter_lines", "profile"]
__version__ = "0.1.0"
