"""Disk-backed exact counts, interpolated modified Kneser–Ney, array compiler.

See Heafield et al. (2013), https://aclanthology.org/P13-2121/.
The database is a temporary build artifact, never used during inference.
"""
from collections import Counter, deque
from itertools import groupby
import math
from pathlib import Path
import sqlite3
import struct
import tempfile
import time

import numpy as np
from ._kernels import compile_probabilities
from .model import check_token, BOS_ID, EOS_ID


def reverse_history(key):
    return b"".join(key[i:i + 4] for i in range(len(key) - 8, -1, -4))


def estimate_discounts(histogram, fallback=0.75):
    """Chen–Goodman count-of-counts estimates with explicit sparse-data fallback."""
    n1, n2, n3, n4 = (histogram.get(i, 0) for i in range(1, 5))
    y = n1 / (n1 + 2 * n2) if n1 + 2 * n2 else 0.0
    result, replaced = [], []
    for k, a, b in ((1, n1, n2), (2, n2, n3), (3, n3, n4)):
        value = k - (k + 1) * y * b / a if a and b and y else math.nan
        invalid = not math.isfinite(value) or not 0.0 < value < k
        result.append(float(fallback if invalid else value))
        replaced.append(bool(invalid))
    return tuple(result), replaced


def validate_discounts(values):
    values = tuple(float(value) for value in values)
    if len(values) != 3 or any(not math.isfinite(v) or not 0 < v < k
                               for k, v in enumerate(values, 1)):
        raise ValueError("Discounts require 0<D1<1, 0<D2<2, and 0<D3+<3.")
    return values


def build(model_class, sequences, *, order, unknown, work_dir, count_batch_entries,
          sqlite_cache_mb, max_model_bytes, probability_dtype, discounts,
          fallback_discount, progress):
    if type(order) is not int or order < 1:
        raise ValueError("order must be a positive integer.")
    if unknown not in ("unk", "error"):
        raise ValueError("unknown must be 'unk' or 'error'.")
    if type(count_batch_entries) is not int or count_batch_entries < 1:
        raise ValueError("count_batch_entries must be positive.")
    if type(sqlite_cache_mb) is not int or sqlite_cache_mb < 1:
        raise ValueError("sqlite_cache_mb must be positive.")
    if max_model_bytes is not None and (type(max_model_bytes) is not int or max_model_bytes < 1):
        raise ValueError("max_model_bytes must be a positive integer or None.")
    if probability_dtype not in ("float32", "float64"):
        raise ValueError("probability_dtype must be float32 or float64.")
    if not math.isfinite(fallback_discount) or not 0 < fallback_discount < 1:
        raise ValueError("fallback_discount must be between 0 and 1.")
    if discounts is not None:
        if isinstance(discounts, dict):
            if any(type(k) is not int or not 1 <= k <= order for k in discounts):
                raise ValueError("Discount mapping keys must be model orders in 1..order.")
            discounts = {k: validate_discounts(v) for k, v in discounts.items()}
        else:
            discounts = {k: validate_discounts(discounts) for k in range(1, order + 1)}
    discounts = discounts or {}
    started = time.perf_counter()

    def report(stage, **details):
        if progress is not None:
            progress({"stage": stage, **details})

    if work_dir is not None:
        Path(work_dir).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fastgram-build-", dir=work_dir) as temporary:
        db = sqlite3.connect(str(Path(temporary) / "counts.sqlite"))
        try:
            # Safe for a disposable, single-process DB; completed models are
            # separately saved atomically by NGramModel.save().
            db.execute("PRAGMA journal_mode=OFF")
            db.execute("PRAGMA synchronous=OFF")
            db.execute("PRAGMA temp_store=FILE")
            db.execute(f"PRAGMA cache_size={-sqlite_cache_mb * 1024}")
            db.execute("PRAGMA mmap_size=0")
            db.create_function("rev_history", 1, reverse_history, deterministic=True)
            db.execute(f"CREATE TABLE t{order}(key BLOB PRIMARY KEY, c INTEGER NOT NULL, "
                       "a INTEGER NOT NULL) WITHOUT ROWID")
            insert = (f"INSERT INTO t{order}(key,c,a) VALUES(?,?,?) "
                      "ON CONFLICT(key) DO UPDATE SET c=c+excluded.c, a=a+excluded.a")
            buffer = Counter()
            vocabulary, token_to_id = [], {}
            pack = struct.Struct(">" + "I" * order).pack
            sequence_count = token_count = 0

            def flush():
                if buffer:
                    db.executemany(insert, ((key, count, count) for key, count in buffer.items()))
                    db.commit()
                    buffer.clear()

            report("counting", order=order)
            for sequence in sequences:
                history = deque([BOS_ID] * (order - 1), maxlen=order)
                for token in sequence:
                    check_token(token)
                    token_id = token_to_id.get(token)
                    if token_id is None:
                        token_id = len(vocabulary) + 3
                        if token_id > np.iinfo(np.uint32).max:
                            raise OverflowError("Vocabulary exceeds uint32 IDs.")
                        vocabulary.append(token)
                        token_to_id[token] = token_id
                    history.append(token_id)
                    buffer[pack(*history)] += 1
                    token_count += 1
                    if len(buffer) >= count_batch_entries:
                        flush()
                history.append(EOS_ID)
                buffer[pack(*history)] += 1
                if len(buffer) >= count_batch_entries:
                    flush()
                sequence_count += 1
                if sequence_count % 10_000 == 0:
                    report("counting", sequences=sequence_count, tokens=token_count)
            flush()
            if sequence_count == 0:
                raise ValueError("Training needs at least one sequence; an empty sequence is allowed.")
            del token_to_id, buffer

            # Each row of t{k+1} denotes one DISTINCT (k+1)-gram. Grouping its
            # suffix counts distinct left extensions exactly, independent of
            # repetition counts. Raw counts are also propagated for BOS cases.
            for k in range(order - 1, 0, -1):
                report("continuation_counts", order=k)
                db.execute(f"CREATE TABLE t{k}(key BLOB PRIMARY KEY, c INTEGER NOT NULL, "
                           "a INTEGER NOT NULL) WITHOUT ROWID")
                db.execute(f"INSERT INTO t{k} SELECT substr(key,5), SUM(c), "
                           "CASE WHEN substr(key,5,4)=? THEN SUM(c) ELSE COUNT(*) END "
                           f"FROM t{k+1} GROUP BY substr(key,5)",
                           (struct.pack(">I", BOS_ID),))
                db.commit()

            discount_values, statistics = {}, {}
            for k in range(1, order + 1):
                histogram = dict(db.execute(f"SELECT a,COUNT(*) FROM t{k} WHERE a<=4 GROUP BY a"))
                if k in discounts:
                    values, fallback_flags = discounts[k], [False] * 3
                    source = "provided"
                else:
                    values, fallback_flags = estimate_discounts(histogram, fallback_discount)
                    source = "estimated"
                discount_values[k] = values
                statistics[str(k)] = {
                    "distinct": db.execute(f"SELECT COUNT(*) FROM t{k}").fetchone()[0],
                    "count_of_counts": {str(j): histogram.get(j, 0) for j in range(1, 5)},
                    "discounts": list(values), "discount_source": source,
                    "fallback_bins": fallback_flags,
                }

            report("context_index")
            db.execute("CREATE TABLE contexts(key BLOB PRIMARY KEY, id INTEGER NOT NULL UNIQUE, "
                       "depth INTEGER NOT NULL) WITHOUT ROWID")
            db.execute("INSERT INTO contexts VALUES(?,0,0)", (b"",))
            context_count = 1
            for depth in range(1, order):
                cursor = db.execute(f"SELECT DISTINCT rev_history(key) AS h FROM t{depth+1} ORDER BY h")
                records = []
                for (key,) in cursor:
                    records.append((key, context_count, depth))
                    context_count += 1
                    if len(records) >= count_batch_entries:
                        db.executemany("INSERT INTO contexts VALUES(?,?,?)", records)
                        records.clear()
                if records:
                    db.executemany("INSERT INTO contexts VALUES(?,?,?)", records)
                db.commit()
            edges = sum(statistics[str(k)]["distinct"] for k in range(2, order + 1))
            vocab_size = len(vocabulary) + 3
            token_dtype = np.dtype("uint8" if vocab_size <= 256 else
                                   "uint16" if vocab_size <= 65536 else "uint32")
            prob_dtype = np.dtype(probability_dtype)
            predicted_bytes = ((context_count + 1) * 16 + context_count * token_dtype.itemsize
                               + edges * (token_dtype.itemsize + prob_dtype.itemsize)
                               + (context_count + vocab_size) * prob_dtype.itemsize)
            if max_model_bytes is not None and predicted_bytes > max_model_bytes:
                raise MemoryError(f"Final arrays need {predicted_bytes:,} bytes; limit is "
                                  f"{max_model_bytes:,}. Reduce order/corpus or raise the limit. "
                                  "This version does not prune n-grams.")
            report("allocating_arrays", contexts=context_count, edges=edges, bytes=predicted_bytes)
            arrays = {
                "child_offsets": np.zeros(context_count + 1, dtype=np.int64),
                "child_labels": np.zeros(context_count, dtype=token_dtype),
                "edge_offsets": np.zeros(context_count + 1, dtype=np.int64),
                "edge_tokens": np.empty(edges, dtype=token_dtype),
                "log_probs": np.empty(edges, dtype=prob_dtype),
                "log_backoffs": np.zeros(context_count, dtype=prob_dtype),
                "unigrams": np.empty(vocab_size, dtype=prob_dtype),
            }
            parents = np.zeros(context_count, dtype=np.uint32 if context_count <= 2**32 else np.uint64)
            child_offsets = arrays["child_offsets"]
            for node, parent, key in db.execute(
                    "SELECT c.id,p.id,c.key FROM contexts c JOIN contexts p "
                    "ON p.key=substr(c.key,1,length(c.key)-4) WHERE c.id>0 ORDER BY c.id"):
                parents[node] = parent
                arrays["child_labels"][node] = int.from_bytes(key[-4:], "big")
                child_offsets[parent + 1] += 1
            np.cumsum(child_offsets, out=child_offsets)
            child_offsets += 1

            unigram_counts = [(int.from_bytes(key, "big"), count)
                              for key, count in db.execute("SELECT key,a FROM t1")]
            total = sum(count for _, count in unigram_counts)
            d = discount_values[1]
            mass = sum(d[min(count, 3) - 1] for _, count in unigram_counts) / total
            probabilities = np.full(vocab_size, mass / (vocab_size - 1), dtype=np.float64)
            probabilities[BOS_ID] = 0.0
            for token, count in unigram_counts:
                probabilities[token] += (count - d[min(count, 3) - 1]) / total
            with np.errstate(divide="ignore"):
                arrays["unigrams"][:] = np.log(probabilities)
            del unigram_counts, probabilities

            # Reorder n-gram entries into CSR context rows using a disk-backed
            # primary key. No all-context Python dictionary is constructed.
            db.execute("CREATE TABLE edges(context INTEGER,token INTEGER,a INTEGER NOT NULL, "
                       "PRIMARY KEY(context,token)) WITHOUT ROWID")
            for k in range(2, order + 1):
                report("probability_rows", order=k)
                last_history = None
                rows = []
                for key, count in db.execute(f"SELECT key,a FROM t{k} ORDER BY key"):
                    history = key[:-4]
                    if history != last_history:
                        context_id = db.execute("SELECT id FROM contexts WHERE key=?",
                                                (reverse_history(key),)).fetchone()[0]
                        last_history = history
                    rows.append((context_id, int.from_bytes(key[-4:], "big"), count))
                    if len(rows) >= count_batch_entries:
                        db.executemany("INSERT INTO edges VALUES(?,?,?)", rows)
                        rows.clear()
                if rows:
                    db.executemany("INSERT INTO edges VALUES(?,?,?)", rows)
                db.execute(f"DROP TABLE t{k}")
                db.commit()
            cursor = db.execute("SELECT e.context,c.depth,e.token,e.a FROM edges e "
                                "JOIN contexts c ON c.id=e.context ORDER BY e.context,e.token")
            index = 0
            for context, group in groupby(cursor, key=lambda row: row[0]):
                group = list(group)  # At most one vocabulary-sized row.
                d = discount_values[group[0][1] + 1]
                total = sum(row[3] for row in group)
                removed = sum(d[min(row[3], 3) - 1] for row in group)
                arrays["log_backoffs"][context] = math.log(removed / total)
                arrays["edge_offsets"][context] = index
                for _, _, token, count in group:
                    arrays["edge_tokens"][index] = token
                    arrays["log_probs"][index] = math.log((count - d[min(count, 3) - 1]) / total)
                    index += 1
                arrays["edge_offsets"][context + 1] = index
            if index != edges:
                raise RuntimeError("Internal edge count mismatch.")
            report("compiling_probabilities")
            compile_probabilities(parents, arrays["edge_offsets"], arrays["edge_tokens"],
                                  arrays["log_probs"], arrays["log_backoffs"], arrays["unigrams"])
            del parents
            metadata = {
                "format_version": 1, "estimator": "interpolated-modified-kneser-ney",
                "order": order, "unknown": unknown, "tokens": vocabulary,
                "boundary_policy": "order-1 BOS padding; one EOS per sequence",
                "unigram_policy": "discounted adjusted counts interpolated with uniform over non-BOS IDs",
                "log_base": "e", "probability_dtype": probability_dtype,
                "training_sequences": sequence_count, "training_tokens": token_count,
                "orders": statistics, "contexts": context_count, "edges": edges,
                "array_bytes": predicted_bytes, "pruning": "none",
                "build_seconds": time.perf_counter() - started,
            }
            model = model_class(metadata, arrays)
            report("complete", seconds=time.perf_counter() - started, array_bytes=model.array_bytes)
            return model
        finally:
            db.close()
