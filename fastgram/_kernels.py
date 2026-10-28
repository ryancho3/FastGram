"""Numeric kernels. No Python objects or allocations in the sum-only scorer."""
import math
import os
import numpy as np

NUMBA_AVAILABLE = False
if os.environ.get("FASTGRAM_DISABLE_NUMBA") != "1":
    try:
        from numba import njit
        NUMBA_AVAILABLE = True
    except ImportError:
        pass
if not NUMBA_AVAILABLE:
    def njit(*args, **kwargs):
        def decorate(function):
            return function
        return decorate


@njit(cache=True, inline="always")
def find_label(labels, start, end, target):
    # Short rows are common in high-order character models.
    if end - start <= 6:
        for index in range(start, end):
            value = labels[index]
            if value == target:
                return index
            if value > target:
                break
        return -1
    while start < end:
        mid = (start + end) // 2
        value = labels[mid]
        if value < target:
            start = mid + 1
        elif value > target:
            end = mid
        else:
            return mid
    return -1


@njit(cache=True)
def compile_probabilities(parents, edge_offsets, edge_tokens, log_probs,
                          log_backoffs, unigrams):
    """Replace log discounted components with log interpolated probabilities.

    Context IDs are breadth-first, so all suffix ancestors are compiled first.
    """
    for context in range(1, len(parents)):
        weight = math.exp(np.float64(log_backoffs[context]))
        for edge in range(edge_offsets[context], edge_offsets[context + 1]):
            token = edge_tokens[edge]
            suffix = int(parents[context])
            backoff = 0.0
            lower = np.float64(unigrams[token])
            while suffix != 0:
                match = find_label(edge_tokens, edge_offsets[suffix],
                                   edge_offsets[suffix + 1], token)
                if match >= 0:
                    lower = np.float64(log_probs[match])
                    break
                backoff += np.float64(log_backoffs[suffix])
                suffix = int(parents[suffix])
            probability = (math.exp(np.float64(log_probs[edge]))
                           + weight * math.exp(lower + backoff))
            log_probs[edge] = math.log(probability)


@njit(cache=True, inline="always")
def query_at(ids, position, start, token, bos, order, child_offsets,
             child_labels, edge_offsets, edge_tokens, log_probs,
             log_backoffs, unigrams):
    result = np.float64(unigrams[token])
    context = 0
    for depth in range(1, order):
        previous = position - depth
        if previous >= start:
            history_token = ids[previous]
        elif bos:
            history_token = 2
        else:
            break
        context = find_label(child_labels, child_offsets[context],
                             child_offsets[context + 1], history_token)
        if context < 0:
            break
        edge = find_label(edge_tokens, edge_offsets[context],
                          edge_offsets[context + 1], token)
        if edge >= 0:
            result = np.float64(log_probs[edge])
        else:
            result += np.float64(log_backoffs[context])
    return result


@njit(cache=True, nogil=True)
def score(ids, start, end, bos, eos, order, child_offsets, child_labels,
          edge_offsets, edge_tokens, log_probs, log_backoffs, unigrams):
    total = 0.0
    for position in range(start, end):
        total += query_at(ids, position, start, ids[position], bos, order,
                          child_offsets, child_labels, edge_offsets,
                          edge_tokens, log_probs, log_backoffs, unigrams)
    if eos:
        total += query_at(ids, end, start, 1, bos, order, child_offsets,
                          child_labels, edge_offsets, edge_tokens,
                          log_probs, log_backoffs, unigrams)
    return total


@njit(cache=True, nogil=True)
def token_scores(ids, bos, eos, order, child_offsets, child_labels,
                 edge_offsets, edge_tokens, log_probs, log_backoffs, unigrams):
    output = np.empty(len(ids) + int(eos), dtype=np.float64)
    for position in range(len(output)):
        token = ids[position] if position < len(ids) else 1
        output[position] = query_at(ids, position, 0, token, bos, order,
                                    child_offsets, child_labels, edge_offsets,
                                    edge_tokens, log_probs, log_backoffs, unigrams)
    return output


@njit(cache=True, nogil=True)
def score_batch(ids, offsets, bos, eos, order, child_offsets, child_labels,
                edge_offsets, edge_tokens, log_probs, log_backoffs, unigrams):
    result = np.empty(len(offsets) - 1, dtype=np.float64)
    for row in range(len(result)):
        result[row] = score(ids, offsets[row], offsets[row + 1], bos, eos,
                            order, child_offsets, child_labels, edge_offsets,
                            edge_tokens, log_probs, log_backoffs, unigrams)
    return result


@njit(cache=True, nogil=True)
def next_log_probs(ids, bos, order, child_offsets, child_labels,
                   edge_offsets, edge_tokens, log_probs, log_backoffs, unigrams):
    output = unigrams.astype(np.float64)
    context = 0
    for depth in range(1, order):
        previous = len(ids) - depth
        if previous >= 0:
            token = ids[previous]
        elif bos:
            token = 2
        else:
            break
        context = find_label(child_labels, child_offsets[context],
                             child_offsets[context + 1], token)
        if context < 0:
            break
        weight = np.float64(log_backoffs[context])
        for token in range(len(output)):
            output[token] += weight
        for edge in range(edge_offsets[context], edge_offsets[context + 1]):
            output[edge_tokens[edge]] = np.float64(log_probs[edge])
    return output


@njit(cache=True)
def validate_arrays(child_offsets, labels, edge_offsets, tokens, log_probs,
                    log_backoffs, unigrams):
    """Validate without allocating arrays proportional to the model."""
    contexts = len(labels)
    vocab_size = len(unigrams)
    for context in range(contexts):
        lo, hi = child_offsets[context], child_offsets[context + 1]
        if lo < 1 or hi < lo or hi > contexts or (hi > lo and lo <= context):
            return 1
        for index in range(lo, hi):
            if labels[index] >= vocab_size:
                return 2
            if index > lo and labels[index] <= labels[index - 1]:
                return 3
        lo, hi = edge_offsets[context], edge_offsets[context + 1]
        if lo < 0 or hi < lo or hi > len(tokens):
            return 4
        for index in range(lo, hi):
            if tokens[index] >= vocab_size or tokens[index] == 2:
                return 5
            if index > lo and tokens[index] <= tokens[index - 1]:
                return 6
            if not math.isfinite(log_probs[index]) or log_probs[index] > 1e-6:
                return 7
        if not math.isfinite(log_backoffs[context]) or log_backoffs[context] > 1e-6:
            return 8
    for token in range(vocab_size):
        if token == 2:
            if unigrams[token] != -math.inf:
                return 9
        elif not math.isfinite(unigrams[token]) or unigrams[token] > 1e-6:
            return 10
    return 0
