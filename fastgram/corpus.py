"""Streaming readers. Text is assumed to be preprocessed by the caller."""
import json
from .model import check_token


def iter_jsonl(path, *, field="text", encoding="utf-8"):
    """Yield a literal top-level field; ignore other metadata. Never normalize.

    A string is interpreted as characters by fit(); an array supplies explicit
    tokens. Blank physical lines are ignored; empty text fields are preserved.
    """
    with open(path, encoding=encoding) as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise TypeError("record must be an object")
                value = record[field]
                if not isinstance(value, (str, list)):
                    raise TypeError("field must be a string or token array")
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            yield value


def iter_lines(path, *, encoding="utf-8"):
    """Each line is an independent sequence; remove only line terminators."""
    with open(path, encoding=encoding) as stream:
        for line in stream:
            yield line.rstrip("\r\n")


def profile(sequences):
    """Exact streaming token/sequence statistics; retains only the vocabulary."""
    vocabulary = set()
    total = records = empty = maximum = 0
    minimum = None
    for sequence in sequences:
        length = 0
        for token in sequence:
            check_token(token)
            vocabulary.add(token)
            length += 1
        records += 1
        total += length
        empty += int(length == 0)
        minimum = length if minimum is None else min(minimum, length)
        maximum = max(maximum, length)
    return {"sequences": records, "tokens": total, "vocabulary_size": len(vocabulary),
            "empty_sequences": empty, "min_length": minimum, "max_length": maximum,
            "mean_length": total / records if records else None}
