"""Public model API and non-pickle, memory-mappable persistence."""
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from types import MappingProxyType

import numpy as np
from . import _kernels as kernels

ARRAY_NAMES = ("child_offsets", "child_labels", "edge_offsets", "edge_tokens",
               "log_probs", "log_backoffs", "unigrams")
UNK_ID, EOS_ID, BOS_ID = 0, 1, 2


def check_token(token):
    if type(token) not in (str, int):
        raise TypeError("Tokens must be strings or Python integers (not bool).")


class NGramModel:
    """Immutable array model; strings are sequences of Unicode code points.

    Input sequences exclude special IDs. BOS/EOS are handled by the scorer.
    Log probabilities use natural logarithms. The vocabulary preserves first
    occurrence order. Different input iteration orders may produce different IDs.
    """
    def __init__(self, metadata, arrays, validate=True):
        self._closed = False
        self.order = metadata["order"]
        self.unknown = metadata["unknown"]
        self.tokens = tuple(metadata["tokens"])
        self.token_to_id = MappingProxyType({token: i + 3 for i, token in enumerate(self.tokens)})
        self._metadata = metadata
        self._arrays = tuple(arrays[name] for name in ARRAY_NAMES)
        for name, array in zip(ARRAY_NAMES, self._arrays):
            setattr(self, "_" + name, array)
        if validate:
            self._validate()
        for array in self._arrays:
            array.flags.writeable = False

    @classmethod
    def fit(cls, sequences, *, order=12, unknown="unk", work_dir=None,
            count_batch_entries=100_000, sqlite_cache_mb=64,
            max_model_bytes=6 * 1024**3, probability_dtype="float32",
            discounts=None, fallback_discount=0.75, progress=None):
        """Train on independent preprocessed sequences, consuming the input once.

        SQLite provides disk-backed exact counting. max_model_bytes limits final
        numeric array allocation, NOT total RSS; see README for memory accounting.
        discounts can be a (D1, D2, D3+) tuple or a mapping from order to tuple.
        Missing/invalid empirical discount estimates use fallback_discount.
        """
        from .builder import build
        return build(cls, sequences, order=order, unknown=unknown, work_dir=work_dir,
                     count_batch_entries=count_batch_entries,
                     sqlite_cache_mb=sqlite_cache_mb, max_model_bytes=max_model_bytes,
                     probability_dtype=probability_dtype, discounts=discounts,
                     fallback_discount=fallback_discount, progress=progress)

    @property
    def vocabulary_size(self):
        """Number of IDs, including UNK, EOS, and the non-predicted BOS."""
        return len(self.tokens) + 3

    @property
    def array_bytes(self):
        return sum(array.nbytes for array in self._arrays)

    @property
    def metadata(self):
        return json.loads(json.dumps(self._metadata))

    def token_label(self, token_id):
        if type(token_id) is not int or not 0 <= token_id < self.vocabulary_size:
            raise ValueError("Invalid token ID.")
        if token_id < 3:
            return ("<UNK>", "<EOS>", "<BOS>")[token_id]
        return self.tokens[token_id - 3]

    def encode(self, tokens):
        def lookup():
            for token in tokens:
                check_token(token)
                value = self.token_to_id.get(token)
                if value is None:
                    if self.unknown == "error":
                        raise KeyError(f"Unknown token: {token!r}")
                    value = UNK_ID
                yield value
        return np.fromiter(lookup(), dtype=np.int64)

    def _ids(self, ids, validate=True):
        if self._closed:
            raise ValueError("Model is closed.")
        array = np.asarray(ids)
        if array.ndim != 1:
            raise ValueError("IDs must be one-dimensional.")
        if array.size and array.dtype.kind not in "iu":
            raise TypeError("IDs must be integers.")
        if validate and array.size:
            if array.min() < 0 or array.max() >= self.vocabulary_size:
                raise ValueError("ID outside the vocabulary.")
            if np.any((array == BOS_ID) | (array == EOS_ID)):
                raise ValueError("Input excludes BOS/EOS; use the bos/eos flags.")
        return np.ascontiguousarray(array, dtype=np.int64)

    def score(self, tokens, *, bos=True, eos=True):
        return self.score_ids(self.encode(tokens), bos=bos, eos=eos, validate=False)

    def score_ids(self, ids, *, bos=True, eos=True, validate=True):
        ids = self._ids(ids, validate)
        return float(kernels.score(ids, 0, len(ids), bool(bos), bool(eos),
                                   self.order, *self._arrays))

    def token_logprobs(self, tokens, *, bos=True, eos=True):
        return self.token_logprobs_ids(self.encode(tokens), bos=bos, eos=eos,
                                       validate=False)

    def token_logprobs_ids(self, ids, *, bos=True, eos=True, validate=True):
        ids = self._ids(ids, validate)
        return kernels.token_scores(ids, bool(bos), bool(eos), self.order, *self._arrays)

    def score_batch_ids(self, flat_ids, offsets, *, bos=True, eos=True, validate=True):
        ids = self._ids(flat_ids, validate)
        offsets = np.asarray(offsets)
        if offsets.ndim != 1 or offsets.dtype.kind not in "iu" or offsets.size == 0:
            raise ValueError("Offsets must be a nonempty integer vector.")
        # Validate offsets even on the trusted token path; malformed slices could
        # otherwise make compiled code read beyond the array.
        if (offsets[0] != 0 or offsets[-1] != len(ids)
                or np.any(offsets[1:] < offsets[:-1])):
            raise ValueError("Offsets must start at 0, be nondecreasing, and end at len(ids).")
        offsets = np.ascontiguousarray(offsets, dtype=np.int64)
        return kernels.score_batch(ids, offsets, bool(bos), bool(eos),
                                   self.order, *self._arrays)

    def next_logprobs_ids(self, context_ids, *, bos=True, validate=True):
        ids = self._ids(context_ids, validate)
        return kernels.next_log_probs(ids, bool(bos), self.order, *self._arrays)

    def next_probs_ids(self, context_ids, *, bos=True, validate=True):
        """Dense vector in ID order. Includes UNK/EOS; BOS has probability zero.

        Sums to one within storage precision; not renormalized, so entries agree
        with the scorer. eos=False scoring still uses this same distribution.
        """
        return np.exp(self.next_logprobs_ids(context_ids, bos=bos, validate=validate))

    def next_probs(self, context=(), *, bos=True):
        return self.next_probs_ids(self.encode(context), bos=bos, validate=False)

    def loss(self, tokens, *, mean=False, bos=True, eos=True):
        ids = self.encode(tokens)
        loss = -self.score_ids(ids, bos=bos, eos=eos, validate=False)
        if mean:
            count = len(ids) + int(eos)
            if count == 0:
                raise ValueError("Mean loss is undefined for zero predicted tokens.")
            loss /= count
        return loss

    def warmup(self):
        ids = np.empty(0, dtype=np.int64)
        self.score_ids(ids)
        self.token_logprobs_ids(ids)
        self.score_batch_ids(ids, np.array([0, 0], dtype=np.int64))
        self.next_probs_ids(ids)

    def save(self, path):
        """Create a new model directory atomically. Refuses to overwrite."""
        if self._closed:
            raise ValueError("Model is closed.")
        path = Path(path).absolute()
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".ngram-", dir=path.parent))
        try:
            for name, array in zip(ARRAY_NAMES, self._arrays):
                np.save(temporary / (name + ".npy"), array, allow_pickle=False)
            (temporary / "metadata.json").write_text(
                json.dumps(self._metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            if path.exists():
                raise FileExistsError(path)
            os.rename(temporary, path)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

    @classmethod
    def load(cls, path, *, mmap=True, validate=True):
        path = Path(path)
        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("format_version") != 1:
            raise ValueError("Unsupported model format.")
        if type(metadata.get("order")) is not int or metadata["order"] < 1:
            raise ValueError("Invalid model order.")
        if metadata.get("unknown") not in ("unk", "error"):
            raise ValueError("Invalid unknown-token policy.")
        for token in metadata["tokens"]:
            check_token(token)
        if len(set(metadata["tokens"])) != len(metadata["tokens"]):
            raise ValueError("Duplicate vocabulary tokens.")
        arrays = {name: np.load(path / (name + ".npy"),
                               mmap_mode="r" if mmap else None, allow_pickle=False)
                  for name in ARRAY_NAMES}
        try:
            return cls(metadata, arrays, validate=validate)
        except BaseException:
            for array in arrays.values():
                mapping = getattr(array, "_mmap", None)
                if mapping is not None:
                    mapping.close()
            raise

    def close(self):
        """Release mapped files, particularly before moving/deleting on Windows.

        Do not close while another thread is scoring with this model.
        """
        if not self._closed:
            self._closed = True
            for array in self._arrays:
                mapping = getattr(array, "_mmap", None)
                if mapping is not None:
                    mapping.close()

    def __enter__(self):
        if self._closed:
            raise ValueError("Model is closed.")
        return self

    def __exit__(self, *exc):
        self.close()

    def _validate(self):
        arrays = dict(zip(ARRAY_NAMES, self._arrays))
        for name, value in arrays.items():
            if value.ndim != 1 or not value.flags.c_contiguous or not value.dtype.isnative:
                raise ValueError(f"Invalid array layout: {name}")
            if name in ("child_offsets", "edge_offsets"):
                valid = value.dtype == np.dtype("int64")
            elif name in ("child_labels", "edge_tokens"):
                valid = value.dtype in (np.dtype("uint8"), np.dtype("uint16"), np.dtype("uint32"))
            else:
                valid = value.dtype in (np.dtype("float32"), np.dtype("float64"))
            if not valid:
                raise ValueError(f"Invalid array dtype: {name}")
        contexts = len(self._child_labels)
        if (contexts == 0 or len(self._child_offsets) != contexts + 1
                or len(self._edge_offsets) != contexts + 1
                or len(self._log_backoffs) != contexts
                or len(self._edge_tokens) != len(self._log_probs)
                or len(self._unigrams) != self.vocabulary_size
                or self._child_offsets[0] != 1 or self._child_offsets[-1] != contexts
                or self._edge_offsets[0] != 0 or self._edge_offsets[1] != 0
                or self._edge_offsets[-1] != len(self._edge_tokens)):
            raise ValueError("Invalid model array dimensions or offsets.")
        code = kernels.validate_arrays(*self._arrays)
        if code:
            raise ValueError(f"Corrupt model arrays (validation code {code}).")
        end = 1
        for _ in range(self.order - 1):
            end = int(self._child_offsets[end])
        if end != contexts:
            raise ValueError("Trie depth exceeds model order.")
        if not math.isclose(float(np.exp(self._unigrams.astype(np.float64)).sum()),
                            1.0, rel_tol=2e-6, abs_tol=2e-6):
            raise ValueError("Unigram probabilities are not normalized.")
