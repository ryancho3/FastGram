"""Offline-friendly command line: profile, train, score, predict."""
import argparse
import json
import sys

from . import NGramModel, NUMBA_AVAILABLE, iter_jsonl, iter_lines, profile


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m fastgram")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("profile", "train"):
        command = commands.add_parser(name)
        command.add_argument("corpus")
        command.add_argument("--format", choices=("jsonl", "lines"), default="jsonl")
        command.add_argument("--field", default="text", help="Literal JSONL field containing preprocessed text/tokens")
        if name == "train":
            command.add_argument("output")
            command.add_argument("--order", type=int, default=12)
            command.add_argument("--work-dir")
            command.add_argument("--count-batch-entries", type=int, default=100_000)
            command.add_argument("--sqlite-cache-mb", type=int, default=64)
            command.add_argument("--max-model-gib", type=float, default=6.0)
            command.add_argument("--probability-dtype", choices=("float32", "float64"), default="float32")
            command.add_argument("--unknown", choices=("unk", "error"), default="unk")
    for name in ("score", "predict"):
        command = commands.add_parser(name)
        command.add_argument("model")
        content = command.add_mutually_exclusive_group(required=True)
        content.add_argument("--text")
        content.add_argument("--tokens-json", help="JSON array of explicit string/integer tokens")
        command.add_argument("--no-bos", action="store_true")
        if name == "score":
            command.add_argument("--no-eos", action="store_true")
        else:
            command.add_argument("--top", type=int, default=10)
    args = parser.parse_args(argv)
    if args.command in ("train", "profile"):
        corpus = (iter_jsonl(args.corpus, field=args.field) if args.format == "jsonl"
                  else iter_lines(args.corpus))
        if args.command == "profile":
            output = profile(corpus)
        else:
            def progress(event):
                print(json.dumps(event), file=sys.stderr, flush=True)
            model = NGramModel.fit(corpus, order=args.order, unknown=args.unknown,
                                   work_dir=args.work_dir,
                                   count_batch_entries=args.count_batch_entries,
                                   sqlite_cache_mb=args.sqlite_cache_mb,
                                   max_model_bytes=int(args.max_model_gib * 1024**3),
                                   probability_dtype=args.probability_dtype, progress=progress)
            model.save(args.output)
            output = model.metadata
            output.pop("tokens")
            output["numba_available"] = NUMBA_AVAILABLE
    else:
        model = NGramModel.load(args.model)
        tokens = args.text if args.text is not None else json.loads(args.tokens_json)
        if args.tokens_json is not None and not isinstance(tokens, list):
            parser.error("--tokens-json must be a JSON array")
        ids = model.encode(tokens)
        if args.command == "score":
            value = model.score_ids(ids, bos=not args.no_bos, eos=not args.no_eos)
            count = len(ids) + int(not args.no_eos)
            output = {"log_probability": value, "negative_log_likelihood": -value,
                      "mean_negative_log_likelihood": -value/count if count else None,
                      "predicted_tokens": count}
        else:
            if args.top < 1:
                parser.error("--top must be positive")
            probabilities = model.next_probs_ids(ids, bos=not args.no_bos)
            top = sorted((i for i in range(model.vocabulary_size) if i != 2),
                         key=lambda i: probabilities[i], reverse=True)[:args.top]
            output = [{"id": i, "token": model.token_label(i), "special": i < 3,
                       "probability": float(probabilities[i])} for i in top]
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
