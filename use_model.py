import argparse
import gzip
import math
import pickle
import random
from collections import Counter
from pathlib import Path

from train_ngram import BOS, EOS, NGramLanguageModel, tokenize


def load_model(model_path: Path) -> NGramLanguageModel:
    with gzip.open(model_path, "rb") as file:
        payload = pickle.load(file)
    return NGramLanguageModel(
        n=payload["n"],
        k=payload["k"],
        lambdas=payload["lambdas"],
        vocab=payload["vocab"],
        vocab_set=set(payload["vocab"]),
        counts_by_order=payload["counts_by_order"],
        context_totals=payload["context_totals"],
    )


def score_text(model: NGramLanguageModel, text: str) -> dict:
    tokens = tokenize(text)
    if not tokens:
        return {"tokens": [], "log_probability": float("-inf"), "perplexity": float("inf")}
    log_probability = model.sentence_log_probability(tokens)
    perplexity = math.exp(-log_probability / (len(tokens) + 1))
    return {"tokens": tokens, "log_probability": log_probability, "perplexity": perplexity}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate or score Darija text with a trained n-gram model.")
    parser.add_argument("--model", type=Path, default=Path("outputs/darija_ngram_model.pkl.gz"))
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--max-tokens", type=int, default=20)
    parser.add_argument("--score", type=str, default="")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    random.seed(args.seed)
    model = load_model(args.model)

    if args.score:
        print(score_text(model, args.score))
        return

    print(model.generate(args.prompt, max_tokens=args.max_tokens))


if __name__ == "__main__":
    main()
