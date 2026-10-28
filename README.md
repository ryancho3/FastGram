# FastGram

Fast, compact n-gram language models in Python.

A corpus-independent, interpolated **modified Kneser–Ney (MKN)** model. The
implementation is Python-written, with NumPy arrays and optional Numba kernels.
No KenLM, C++ language-model wrapper, pandas, scipy, or other NLP package is used.
Python's standard-library `sqlite3` module provides temporary disk-backed counting.
It is not involved in inference. NumPy/Numba and the bundled SQLite engine contain
native code; this project has no custom native extension to compile.

This release prioritizes **full sequence scoring**. Partial rescoring and pruning
are deliberately not implemented yet.

## Run offline

Requires Python 3.10+, NumPy, and (for speed) Numba with a compatible llvmlite.
Use your approved installed versions. Do not upgrade an air-gapped environment
just to run this package. Tested versions are recorded in `benchmark-results.json`.

Extract the archive and run commands from the directory containing this README.
No installation or network access is needed if NumPy and Numba are already installed:

```bash
python -m unittest discover -s tests -v
python -m fastgram profile examples/jamo.jsonl
python -m fastgram train examples/jamo.jsonl demo-model --order 12
python -m fastgram score demo-model --text "ㅎㅏㄴㄱㅡㄹ"
python -m fastgram predict demo-model --text "ㅎㅏㄴ" --top 5
```

To import from elsewhere, place this project's directory on `PYTHONPATH` or copy
the inner `fastgram` package beside your script. An optional local package install
is `python -m pip install --no-deps --no-build-isolation .` if setuptools is installed.
Without Numba the same kernels execute in Python, correctly but much more slowly.
Inspect `fastgram.NUMBA_AVAILABLE`; a broken installed Numba raises its import
error rather than silently pretending acceleration works.

## Train and score

```python
from fastgram import NGramModel, iter_jsonl

# The selected field must already contain your preprocessed Jamo string or
# an explicit array of tokens. Other JSONL metadata is ignored.
model = NGramModel.fit(
    iter_jsonl("corpus.jsonl", field="text"),
    order=12,
    work_dir="temporary-build-files",  # a disk with substantial free space
    count_batch_entries=100_000,
    sqlite_cache_mb=64,
    max_model_bytes=6 * 1024**3,
    progress=print,
)
model.save("jamo-model")

ids = model.encode("ㅎㅏㄴㄱㅡㄹ")
model.warmup()                         # compile kernels before measuring speed
logp = model.score_ids(ids)            # higher is better, natural logarithm
loss = -logp                          # lower is better
mean_loss = model.loss("ㅎㅏㄴㄱㅡㄹ", mean=True)
terms = model.token_logprobs_ids(ids)  # optional allocation; includes final EOS
probs = model.next_probs_ids(ids)      # vector in token-ID order

# For IDs produced by encode() and left valid, skip repeated range validation:
fast_logp = model.score_ids(ids, validate=False)
loaded = NGramModel.load("jamo-model", mmap=True)
loaded.close()                        # release mapped files when finished
```

`max_model_bytes` is a **guard on final numeric array allocation**, not a hard
process-RAM cap. Build memory also includes the Python vocabulary, bounded count
buffer, SQLite cache, one temporary parent-ID array, and at most a vocabulary-sized
probability row. Start on a representative subset and measure peak process memory
before a full build on your machine. The bound does not limit temporary disk use.
The guard runs after counting/indexing, before final arrays are allocated.

The builder counts maximum-order grams in bounded batches. It derives exact
lower-order raw and continuation counts on disk, then constructs the arrays.
SQLite is a practical initial builder, not a promise of fast training at hundreds
of millions of entries. Temporary databases and sorts may require **many times**
the input file size. An interruption cleans up the build directory on normal
exception handling; a killed process can leave a disposable temporary directory.

No automatic text normalization, whitespace stripping, Jamo conversion, rare-token
replacement, or vocabulary cutoff occurs. Use a generator to apply your own
preprocessor before `fit()`. Each yielded sequence has independent boundaries.
Strings are sequences of Unicode code points; word tokens must be provided as
lists/tuples. Tokens can be strings or Python integers. Arbitrary objects are not
supported, which keeps vocabulary serialization portable and pickle-free.

## Scores and boundaries

- Training prepends `order - 1` virtual BOS tokens and predicts each real token
  followed by one EOS. It never predicts BOS.
- `score*` defaults to `bos=True, eos=True`. `bos=False` starts with an empty
  history; `eos=False` omits the final EOS score. Neither flag changes the trained
  probabilities or renormalizes away EOS.
- The mean loss denominator is `len(ids) + int(eos)`. Empty input without EOS has
  total score zero; mean loss is undefined and raises an error.
- Vocabulary IDs preserve first occurrence order: 0=UNK, 1=EOS, 2=BOS, real
  tokens start at 3. `encode()` never inserts boundaries. IDs 1 and 2 must not be
  included in scoring inputs. Literal text tokens like `"<EOS>"` remain ordinary
  real tokens and cannot collide with the internal IDs.
- `unknown="unk"` maps unseen tokens to ID 0; `unknown="error"` rejects them
  during encoding. UNK is one aggregate event, not a separate probability for
  every possible unseen Unicode token.
- Next-token vectors include UNK and EOS; BOS has probability zero. Use
  `token_label(id)` to display entries. Values sum to one within numerical
  precision and agree with scoring. No extra renormalization is applied.
- Arrays are read-only. `validate=False` is for trusted IDs only: invalid inputs
  can cause out-of-bounds accesses in compiled kernels. Default calls validate.

Batch scoring avoids one Python call per sentence:

```python
import numpy as np
sequences = [model.encode("ㅎㅏㄴ"), model.encode("ㄱㅏ"), model.encode("")]
flat = np.concatenate(sequences)
offsets = np.concatenate(([0], np.cumsum([len(x) for x in sequences])))
scores = model.score_batch_ids(flat, offsets)
```

## Statistical definition

For order N, highest-order counts are occurrence counts. For lower-order grams,
the adjusted count is the number of distinct tokens immediately preceding that
gram. Grams starting with BOS retain their ordinary counts. All global counts
are aggregated before computing continuation statistics.

At each order the builder estimates D1, D2, and D3+ using Chen–Goodman formulas
on adjusted-count histograms. Where those estimates are undefined or outside
`0 < Dk < k`, the relevant bin falls back to 0.75 (configurable). This is especially
important for small corpora and small alphabets. Metadata reports every fallback.
You can provide an explicit `(D1, D2, D3+)` tuple for all orders or a dictionary
mapping individual orders to tuples.

Each history combines its discounted counts with the shorter-context distribution.
For the unigram base, discounted adjusted counts are interpolated with a uniform
distribution over all non-BOS IDs, including UNK and EOS. This gives every
predictable ID positive probability. This explicit boundary/unigram policy need
not match another toolkit's choices, even when both models are called MKN.

Final interpolated probabilities and backoff weights are computed once at build
time. Inference performs no count estimation, continuation counting, or divisions.
Default storage is float32; accumulation and outputs are float64. A float64
storage option is available for validation or unusually sensitive ranking tasks.

References:
- Heafield et al. (2013): https://aclanthology.org/P13-2121/
- Chen and Goodman (1998): https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-10-98.pdf

## Array layout and query procedure

Histories form a **reverse-context trie**: newest token first. Thus a context's
trie parent is its shorter suffix. Context IDs are assigned breadth-first, with
sorted children, making each node's children a contiguous interval of IDs.

| Array | Content |
|---|---|
| `child_offsets[C+1]` | Ranges of child context IDs |
| `child_labels[C]` | Token added by each context node |
| `edge_offsets[C+1]` | Ranges of explicit next-token probability entries |
| `edge_tokens[E]` | Sorted next-token IDs within each context row |
| `log_probs[E]` | Full interpolated log probabilities for explicit entries |
| `log_backoffs[C]` | Log fallback weights |
| `unigrams[V]` | Dense base log probabilities, including special IDs |

Token labels use uint8/uint16/uint32 according to the learned vocabulary size.
Offsets are int64. There are no full n-gram tuple keys in inference and no hash
collisions. This is compact typed storage, not bit-level compression.

To score a token, start at its unigram probability and walk backward through at
most N-1 history tokens. At each matching context, replace the running value with
the stored full probability if an explicit entry exists; otherwise add its log
backoff weight. One traversal visits all applicable suffix contexts. Sorted child
and probability-row searches use a tiny-row linear scan or binary search.

Full scoring costs O(L N log V) in the worst case with these searches and makes
no per-token allocation. No incremental state is kept across calls. The dense
next-token routine traverses contexts once and updates a vector, costing up to
O(N V) plus row searches. The scorer does not allocate that vector.

Models save as a directory of `.npy` arrays plus JSON metadata. `load(mmap=True)`
maps arrays read-only; validation touches the arrays, and fast random access still
depends on the working set fitting RAM. Memory mapping is not a substitute for a
model-size budget. No pickle or Numba cache files are required in saved models.
Use `close()` or `with NGramModel.load(path) as model:` to release mapped files,
especially before deleting/moving a model directory on Windows. Never close a
model while another thread is using it.

## Verification and benchmarking

```bash
python -m unittest discover -s tests -v
python benchmark.py --records 500 --length 64 --rounds 20 --order 12
```

Tests use a separately written tuple/count-based reference estimator, exhaustive
small-context normalization checks, unseen-token/history tests, order-12 checks,
variable-length batches, boundary toggles, and mmap/save/load checks. They also
check that accelerated kernels compiled in Numba's nopython mode.

To test the Python fallback on Linux/macOS:
`FASTGRAM_DISABLE_NUMBA=1 python -m unittest discover -s tests -v`.
On Windows, set the same environment variable using your shell before running
the command, then unset it to restore acceleration.

The included benchmark uses a small synthetic 40-token corpus. It reports warm
resident-model throughput for training-like strings and random candidates,
separates batch and individual calls, and excludes encoding and JIT startup.
It does **not** establish throughput or memory use on your actual Jamo corpus.

## Current limitations and next steps

- No pruning, quantization, partial rescoring, online vocabulary growth, or
  stateful incremental decoding yet.
- The disk builder favors bounded memory and correctness over training speed.
- Final model arrays must fit the configured budget. If an unpruned order-12
  model exceeds it, this release stops with a size estimate instead of silently
  changing the statistical model. Try a smaller order/subset until pruning exists.
- Parallel batch scoring and dense lower-order lookup specializations can be
  benchmarked after measuring the real workload.
- This first version has been tested in the supplied Linux environment, not on
  other platforms. Run the included suite on your installed Python/NumPy/Numba versions.
