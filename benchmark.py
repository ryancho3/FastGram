"""Reproducible order-12 smoke benchmark; not a real-corpus performance claim.

Run from this directory: python benchmark.py --records 500 --length 64 --rounds 20
All reported scoring rates exclude encoding, model construction, and JIT warmup.
"""
import argparse
import json
import platform
import time

import numpy as np
from fastgram import NGramModel, NUMBA_AVAILABLE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=int, default=500)
    parser.add_argument("--length", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--order", type=int, default=12)
    parser.add_argument("--output")
    args = parser.parse_args()
    if min(args.records, args.length, args.rounds, args.order) < 1:
        parser.error("All numerical arguments must be positive")
    rng = np.random.default_rng(731)
    # A structured training distribution with occasional random transitions.
    corpus = []
    for _ in range(args.records):
        token = int(rng.integers(0, 40))
        row = []
        for _ in range(args.length):
            row.append(str(token))
            token = ((token + 1) % 40 if rng.random() < 0.85
                     else int(rng.integers(0, 40)))
        corpus.append(row)
    start = time.perf_counter()
    model = NGramModel.fit(corpus, order=args.order)
    build_seconds = time.perf_counter() - start
    encoded = np.stack([model.encode(row) for row in corpus])
    random_ids = rng.integers(3, model.vocabulary_size, size=encoded.shape, dtype=np.int64)
    offsets = np.arange(0, encoded.size + 1, args.length, dtype=np.int64)
    start = time.perf_counter()
    model.warmup()
    warmup_seconds = time.perf_counter() - start
    measurements = {}
    for label, matrix in (("training_like", encoded), ("random_candidates", random_ids)):
        flat = matrix.reshape(-1)
        model.score_batch_ids(flat, offsets, validate=False)
        start = time.perf_counter()
        checksum = 0.0
        for _ in range(args.rounds):
            checksum += float(model.score_batch_ids(flat, offsets, validate=False).sum())
        seconds = time.perf_counter() - start
        measurements[label] = {
            "batch_seconds": seconds,
            "batch_predicted_tokens_per_second": args.rounds * (encoded.size + args.records) / seconds,
            "score_checksum": checksum,
        }
        start = time.perf_counter()
        single_checksum = 0.0
        for _ in range(args.rounds):
            for row in matrix:
                single_checksum += model.score_ids(row, validate=False)
        single_seconds = time.perf_counter() - start
        measurements[label]["single_call_predicted_tokens_per_second"] = (
            args.rounds * (encoded.size + args.records) / single_seconds)
        if not np.isclose(checksum, single_checksum, rtol=1e-10):
            raise AssertionError("Batch and individual scores disagree")
    result = {
        "python": platform.python_version(), "platform": platform.platform(),
        "numpy": np.__version__, "numba_available": NUMBA_AVAILABLE,
        "order": args.order, "training_records": args.records,
        "sequence_length": args.length, "rounds": args.rounds,
        "vocabulary_size_including_specials": model.vocabulary_size,
        "array_bytes": model.array_bytes, "contexts": model.metadata["contexts"],
        "edges": model.metadata["edges"], "build_seconds": build_seconds,
        "kernel_warmup_seconds": warmup_seconds, "measurements": measurements,
        "notes": "Synthetic corpus; warm resident arrays; includes EOS in token rates; no encoding or JIT in scoring rates."
    }
    if NUMBA_AVAILABLE:
        import numba
        result["numba"] = numba.__version__
    output = json.dumps(result, indent=2)
    print(output)
    if args.output:
        from pathlib import Path
        Path(args.output).write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
