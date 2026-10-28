"""Independent count-based oracle plus API, persistence, and edge-case tests."""
from collections import Counter
import itertools
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
from fastgram import NGramModel, iter_jsonl, profile, NUMBA_AVAILABLE
from fastgram.builder import estimate_discounts


class ReferenceMKN:
    """Deliberately independent: count ALL orders directly in Python tuples."""
    def __init__(self, encoded, order, vocab_size, discounts):
        self.order, self.vocab_size = order, vocab_size
        raw = [Counter() for _ in range(order + 1)]
        for ids in encoded:
            padded = [2] * (order - 1) + list(ids) + [1]
            for end in range(order - 1, len(padded)):
                for k in range(1, order + 1):
                    raw[k][tuple(padded[end - k + 1:end + 1])] += 1
        self.counts = [Counter() for _ in range(order + 1)]
        self.counts[order] = raw[order]
        for k in range(1, order):
            extensions = {}
            for gram in raw[k + 1]:
                extensions.setdefault(gram[1:], set()).add(gram[0])
            self.counts[k] = Counter({gram: count if gram[0] == 2 else len(extensions[gram])
                                      for gram, count in raw[k].items()})
        self.discounts = discounts

    def probability(self, token, history):
        if token == 2:
            return 0.0
        history = tuple(history[-(self.order - 1):]) if self.order > 1 else ()
        k = len(history) + 1
        row = {gram[-1]: count for gram, count in self.counts[k].items()
               if gram[:-1] == history}
        total = sum(row.values())
        if not total:
            return self.probability(token, history[1:])
        ds = self.discounts[k]
        removed = sum(ds[min(count, 3) - 1] for count in row.values())
        count = row.get(token, 0)
        direct = (count - ds[min(count, 3) - 1]) / total if count else 0.0
        lower = self.probability(token, history[1:]) if history else 1 / (self.vocab_size - 1)
        return direct + removed / total * lower

    def score_terms(self, ids, bos, eos):
        history = [2] * (self.order - 1) if bos else []
        terms = []
        for token in list(ids) + ([1] if eos else []):
            terms.append(math.log(self.probability(token, history)))
            history.append(token)
        return np.array(terms)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = ["aba", "abac", "bca", "abc", "abac", "c", "", "aaaa"]
        cls.fixed = (0.4, 0.8, 1.1)
        cls.model = NGramModel.fit(cls.corpus, order=4, discounts=cls.fixed,
                                  probability_dtype="float64", count_batch_entries=3)
        cls.reference = ReferenceMKN([cls.model.encode(s) for s in cls.corpus], 4,
                                     cls.model.vocabulary_size, {k: cls.fixed for k in range(1, 5)})

    def test_independent_oracle_scores(self):
        for bos, eos in itertools.product((False, True), repeat=2):
            for text in ("", "a", "abac", "cccc", "unknown", "bacabc"):
                with self.subTest(bos=bos, eos=eos, text=text):
                    ids = self.model.encode(text)
                    expected = self.reference.score_terms(ids, bos, eos)
                    actual = self.model.token_logprobs_ids(ids, bos=bos, eos=eos)
                    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
                    self.assertAlmostEqual(self.model.score_ids(ids, bos=bos, eos=eos),
                                           float(expected.sum()), places=11)

    def test_all_small_context_distributions(self):
        ids = [0, 3, 4, 5]
        for length in range(4):
            for context in itertools.product(ids, repeat=length):
                for bos in (False, True):
                    prefix = [2] * 3 + list(context) if bos else list(context)
                    expected = [self.reference.probability(w, prefix)
                                for w in range(self.model.vocabulary_size)]
                    actual = self.model.next_probs_ids(context, bos=bos)
                    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
                    self.assertAlmostEqual(float(actual.sum()), 1.0, places=12)

    def test_estimated_discount_formula(self):
        ds, flags = estimate_discounts({1: 100, 2: 40, 3: 20, 4: 10})
        y = 100 / 180
        np.testing.assert_allclose(ds, (1 - 2*y*40/100, 2 - 3*y*20/40, 3 - 4*y*10/20))
        self.assertEqual(flags, [False] * 3)
        self.assertEqual(estimate_discounts({})[1], [True] * 3)

    def test_default_estimates_and_continuation_counts(self):
        m = NGramModel.fit(self.corpus, order=4, probability_dtype="float64")
        ds = {k: tuple(m.metadata["orders"][str(k)]["discounts"]) for k in range(1, 5)}
        ref = ReferenceMKN([m.encode(s) for s in self.corpus], 4, m.vocabulary_size, ds)
        for k in range(1, 5):
            stats = m.metadata["orders"][str(k)]
            hist = Counter(ref.counts[k].values())
            self.assertEqual(stats["distinct"], len(ref.counts[k]))
            self.assertEqual(stats["count_of_counts"], {str(j): hist[j] for j in range(1, 5)})
        np.testing.assert_allclose(m.token_logprobs("cba?"),
                                   ref.score_terms(m.encode("cba?"), True, True), atol=1e-12)

    def test_float32_matches_reference(self):
        m = NGramModel.fit(self.corpus, order=12, discounts=self.fixed)
        ref = ReferenceMKN([m.encode(s) for s in self.corpus], 12, m.vocabulary_size,
                           {k: self.fixed for k in range(1, 13)})
        np.testing.assert_allclose(m.token_logprobs("abc?aaaa"),
                                   ref.score_terms(m.encode("abc?aaaa"), True, True),
                                   atol=3e-6, rtol=3e-6)
        self.assertAlmostEqual(float(m.next_probs("abc").sum()), 1, places=6)

    def test_batch_empty_and_nonempty(self):
        texts = ["abc", "", "?", "abac", ""]
        encoded = [self.model.encode(s) for s in texts]
        offsets = np.concatenate(([0], np.cumsum([len(s) for s in encoded])))
        flat = np.concatenate(encoded)
        for bos, eos in itertools.product((False, True), repeat=2):
            expected = [self.model.score(s, bos=bos, eos=eos) for s in texts]
            np.testing.assert_allclose(self.model.score_batch_ids(flat, offsets, bos=bos, eos=eos), expected)
        self.assertEqual(self.model.score_batch_ids([], np.array([0])).shape, (0,))

    def test_distribution_agrees_with_each_score_term(self):
        m = self.model
        ids = m.encode("ab?ca")
        terms = m.token_logprobs_ids(ids)
        for index, token in enumerate(ids):
            self.assertAlmostEqual(math.log(m.next_probs_ids(ids[:index])[token]), terms[index], places=12)
        self.assertAlmostEqual(math.log(m.next_probs_ids(ids)[1]), terms[-1], places=12)

    def test_order_one_and_empty_record(self):
        for corpus, order in ((["abc", "a"], 1), ([""], 12)):
            m = NGramModel.fit(corpus, order=order)
            self.assertAlmostEqual(float(m.next_probs().sum()), 1, places=6)
            self.assertTrue(math.isfinite(m.score("?")))
            self.assertEqual(m.score("", eos=False), 0)
        with self.assertRaises(ValueError):
            NGramModel.fit([], order=2)

    def test_unicode_integer_and_reserved_text_tokens(self):
        corpus = [["<EOS>", "ᄀ", "ㄱ", 10], ["<UNK>", 11, "ᄀ"]]
        m = NGramModel.fit(corpus, order=3)
        self.assertNotEqual(m.token_to_id["ᄀ"], m.token_to_id["ㄱ"])
        self.assertGreaterEqual(m.token_to_id["<EOS>"], 3)
        self.assertGreaterEqual(m.token_to_id[10], 3)
        self.assertEqual(m.encode(["unseen"]).tolist(), [0])
        self.assertTrue(math.isfinite(m.score(corpus[0])))

    def test_larger_vocabulary_uses_uint16(self):
        m = NGramModel.fit([list(range(300))], order=2)
        self.assertEqual(m._edge_tokens.dtype, np.dtype("uint16"))
        self.assertAlmostEqual(float(m.next_probs([299]).sum()), 1, places=6)

    def test_save_load_mmap_and_plain(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "model"
            self.model.save(path)
            for mmap in (False, True):
                m = NGramModel.load(path, mmap=mmap)
                self.assertEqual(m.score("ab?c"), self.model.score("ab?c"))
                np.testing.assert_array_equal(m.next_probs("ab"), self.model.next_probs("ab"))
                self.assertFalse(m._log_probs.flags.writeable)
                m.close()
            with self.assertRaises(FileExistsError):
                self.model.save(path)

    def test_reject_corrupt_model(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "model"
            self.model.save(path)
            values = np.load(path / "child_offsets.npy")
            values[1] = 10**9
            np.save(path / "child_offsets.npy", values)
            with self.assertRaises(ValueError):
                NGramModel.load(path)

    def test_context_manager_releases_model(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "model"
            self.model.save(path)
            with NGramModel.load(path) as model:
                self.assertTrue(math.isfinite(model.score("ab")))
            with self.assertRaises(ValueError):
                model.score("ab")
            with self.assertRaises(ValueError):
                model.save(Path(temp) / "closed-copy")
            model.close()  # Idempotent.

    def test_validation(self):
        for ids in ([-1], [999], [1], [2], np.array([2**64-1], dtype=np.uint64)):
            with self.assertRaises(ValueError):
                self.model.score_ids(ids)
        for ids in ([3.5], [True], [[3]]):
            with self.assertRaises((TypeError, ValueError)):
                self.model.score_ids(ids)
        for offsets in ([1, 2], [0, 3, 2], [0], [-1, 2]):
            with self.assertRaises(ValueError):
                self.model.score_batch_ids([3, 4], offsets)
        for arguments in ({"order": 0}, {"discounts": (1, 2, 3)},
                          {"probability_dtype": "float16"}, {"unknown": "guess"}):
            with self.assertRaises(ValueError):
                NGramModel.fit(["abc"], **arguments)

    def test_memory_limit_and_temporary_cleanup(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(MemoryError):
                NGramModel.fit(["abc"], max_model_bytes=1, work_dir=temp)
            self.assertEqual(list(Path(temp).iterdir()), [])

    def test_unknown_error_and_mean_loss(self):
        m = NGramModel.fit(["ab"], unknown="error")
        with self.assertRaises(KeyError):
            m.score("c")
        self.assertAlmostEqual(m.loss("ab", mean=True), -m.score("ab") / 3)
        with self.assertRaises(ValueError):
            m.loss("", mean=True, eos=False)

    def test_jsonl_and_profile(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "data.jsonl"
            path.write_text('\n'.join(json.dumps(x) for x in
                                      [{"text": "ab", "ignored": 123}, {"text": ["a", 9]}, {"text": ""}]),
                            encoding="utf-8")
            stats = profile(iter_jsonl(path))
            self.assertEqual(stats["tokens"], 4)
            self.assertEqual(stats["vocabulary_size"], 3)
            self.assertEqual(stats["empty_sequences"], 1)
            m = NGramModel.fit(iter_jsonl(path), order=3)
            self.assertEqual(m.metadata["training_tokens"], 4)

    @unittest.skipUnless(NUMBA_AVAILABLE, "Numba not installed or explicitly disabled")
    def test_kernels_compile_in_nopython_mode(self):
        from fastgram import _kernels
        self.model.warmup()
        for name in ("score", "token_scores", "score_batch", "next_log_probs"):
            self.assertTrue(getattr(_kernels, name).nopython_signatures)


if __name__ == "__main__":
    unittest.main()
