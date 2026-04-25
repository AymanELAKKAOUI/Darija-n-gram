import argparse
import gzip
import json
import math
import pickle
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


TOKEN_PATTERN = re.compile(r"[\u0600-\u06FFA-Za-z0-9]+(?:['’_-][\u0600-\u06FFA-Za-z0-9]+)*")
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
WHITESPACE_PATTERN = re.compile(r"\s+")
ARABIC_VARIANT_PATTERN = re.compile(r"[أإآٱ]")
IGNORED_FILES = {"links_processed.txt", "amlignore", "DS_Store"}
UNK = "<unk>"
BOS = "<s>"
EOS = "</s>"


def normalize_text(text: str) -> str:
    text = text.replace("\ufeff", " ")
    text = URL_PATTERN.sub(" ", text)
    text = text.replace("…", " ")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("’", "'").replace("`", "'")
    text = ARABIC_VARIANT_PATTERN.sub("ا", text)
    text = text.replace("ؤ", "و").replace("ئ", "ي")
    text = WHITESPACE_PATTERN.sub(" ", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    cleaned = normalize_text(text)
    tokens = [token.lower() for token in TOKEN_PATTERN.findall(cleaned)]
    return tokens


def read_sentences(corpus_dir: Path) -> tuple[list[list[str]], dict]:
    sentences: list[list[str]] = []
    stats = {
        "files_used": 0,
        "files_skipped": 0,
        "raw_lines": 0,
        "usable_sentences": 0,
    }
    for path in sorted(corpus_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.name in IGNORED_FILES or path.suffix.lower() != ".txt":
            stats["files_skipped"] += 1
            continue
        text = path.read_bytes().decode("utf-8", errors="replace")
        stats["files_used"] += 1
        for line in text.splitlines():
            stats["raw_lines"] += 1
            tokens = tokenize(line)
            if len(tokens) < 2:
                continue
            sentences.append(tokens)
            stats["usable_sentences"] += 1
    return sentences, stats


def replace_rare_tokens(sentences: list[list[str]], min_freq: int) -> tuple[list[list[str]], set[str], int]:
    counts = Counter(token for sentence in sentences for token in sentence)
    vocab = {token for token, freq in counts.items() if freq >= min_freq}
    vocab.update({UNK, BOS, EOS})
    replaced = 0
    normalized_sentences: list[list[str]] = []
    for sentence in sentences:
        normalized = []
        for token in sentence:
            if token in vocab:
                normalized.append(token)
            else:
                normalized.append(UNK)
                replaced += 1
        normalized_sentences.append(normalized)
    return normalized_sentences, vocab, replaced


@dataclass
class NGramLanguageModel:
    n: int
    k: float
    lambdas: list[float]
    vocab: list[str]
    vocab_set: set[str]
    counts_by_order: dict[int, Counter]
    context_totals: dict[int, Counter]

    @classmethod
    def train(cls, sentences: list[list[str]], n: int, k: float, min_freq: int) -> tuple["NGramLanguageModel", dict]:
        normalized_sentences, vocab_set, replaced = replace_rare_tokens(sentences, min_freq=min_freq)
        counts_by_order = {order: Counter() for order in range(1, n + 1)}
        context_totals = {order: Counter() for order in range(2, n + 1)}

        for sentence in normalized_sentences:
            padded = [BOS] * (n - 1) + sentence + [EOS]
            for index in range(len(padded)):
                for order in range(1, n + 1):
                    start = index - order + 1
                    if start < 0:
                        continue
                    ngram = tuple(padded[start : index + 1])
                    counts_by_order[order][ngram] += 1
                    if order > 1:
                        context_totals[order][ngram[:-1]] += 1

        lambda_total = sum(range(1, n + 1))
        lambdas = [order / lambda_total for order in range(1, n + 1)]
        training_stats = {
            "vocab_size": len(vocab_set),
            "replaced_with_unk": replaced,
            "tokens_after_padding": sum(sum(len(sentence) + 1 for sentence in normalized_sentences) for _ in [0]),
        }
        return cls(
            n=n,
            k=k,
            lambdas=lambdas,
            vocab=sorted(vocab_set),
            vocab_set=vocab_set,
            counts_by_order=counts_by_order,
            context_totals=context_totals,
        ), training_stats

    def map_token(self, token: str) -> str:
        token = token.lower()
        return token if token in self.vocab_set else UNK

    def probability(self, context: list[str], token: str) -> float:
        token = self.map_token(token)
        context = [self.map_token(part) for part in context]
        vocab_size = len(self.vocab)
        probability = 0.0
        max_order = min(self.n, len(context) + 1)
        lambda_slice = self.lambdas[:max_order]
        lambda_norm = sum(lambda_slice)
        normalized_lambdas = [value / lambda_norm for value in lambda_slice]

        for offset, order in enumerate(range(1, max_order + 1)):
            if order == 1:
                numerator = self.counts_by_order[1][(token,)] + self.k
                denominator = sum(self.counts_by_order[1].values()) + (self.k * vocab_size)
            else:
                reduced_context = tuple(context[-(order - 1) :])
                numerator = self.counts_by_order[order][reduced_context + (token,)] + self.k
                denominator = self.context_totals[order][reduced_context] + (self.k * vocab_size)
            probability += normalized_lambdas[offset] * (numerator / denominator)
        return probability

    def sentence_log_probability(self, sentence: list[str]) -> float:
        mapped = [self.map_token(token) for token in sentence]
        padded = [BOS] * (self.n - 1) + mapped + [EOS]
        total = 0.0
        for index in range(self.n - 1, len(padded)):
            token = padded[index]
            context = padded[max(0, index - self.n + 1) : index]
            total += math.log(self.probability(context, token))
        return total

    def perplexity(self, sentences: list[list[str]]) -> float:
        token_count = 0
        log_probability = 0.0
        for sentence in sentences:
            token_count += len(sentence) + 1
            log_probability += self.sentence_log_probability(sentence)
        return math.exp(-log_probability / max(token_count, 1))

    def next_token_distribution(self, context: list[str], top_k: int = 10) -> list[tuple[str, float]]:
        scores = [(token, self.probability(context, token)) for token in self.vocab if token not in {BOS}]
        scores.sort(key=lambda item: item[1], reverse=True)
        return scores[:top_k]

    def generate(self, prompt: str, max_tokens: int = 20) -> str:
        context = tokenize(prompt)
        generated = list(context)
        for _ in range(max_tokens):
            candidates = self.next_token_distribution(generated[-(self.n - 1) :], top_k=12)
            filtered = [(token, prob) for token, prob in candidates if token not in {BOS}]
            if not filtered:
                break
            total = sum(prob for _, prob in filtered)
            threshold = random.random() * total
            cumulative = 0.0
            next_token = EOS
            for token, prob in filtered:
                cumulative += prob
                if cumulative >= threshold:
                    next_token = token
                    break
            if next_token == EOS:
                break
            generated.append(next_token)
        return " ".join(generated)


def split_data(sentences: list[list[str]], validation_fraction: float, seed: int) -> tuple[list[list[str]], list[list[str]]]:
    shuffled = list(sentences)
    random.Random(seed).shuffle(shuffled)
    validation_size = max(1, int(len(shuffled) * validation_fraction))
    validation = shuffled[:validation_size]
    training = shuffled[validation_size:]
    return training, validation


def train_and_select(
    sentences: list[list[str]],
    candidate_orders: list[int],
    k: float,
    min_freq: int,
    validation_fraction: float,
    seed: int,
) -> tuple[NGramLanguageModel, dict]:
    train_sentences, validation_sentences = split_data(sentences, validation_fraction=validation_fraction, seed=seed)
    experiments = []
    best_model = None
    best_metrics = None

    for order in candidate_orders:
        model, training_stats = NGramLanguageModel.train(train_sentences, n=order, k=k, min_freq=min_freq)
        perplexity = model.perplexity(validation_sentences)
        record = {
            "n": order,
            "validation_perplexity": perplexity,
            "training_stats": training_stats,
        }
        experiments.append(record)
        if best_metrics is None or perplexity < best_metrics["validation_perplexity"]:
            best_model = model
            best_metrics = record

    assert best_model is not None and best_metrics is not None
    final_model, final_training_stats = NGramLanguageModel.train(sentences, n=best_model.n, k=k, min_freq=min_freq)
    summary = {
        "candidate_orders": candidate_orders,
        "validation_fraction": validation_fraction,
        "seed": seed,
        "experiments": experiments,
        "selected_order": best_model.n,
        "selected_validation_perplexity": best_metrics["validation_perplexity"],
        "final_training_stats": final_training_stats,
    }
    return final_model, summary


def save_model(model: NGramLanguageModel, output_path: Path) -> None:
    payload = {
        "n": model.n,
        "k": model.k,
        "lambdas": model.lambdas,
        "vocab": model.vocab,
        "counts_by_order": model.counts_by_order,
        "context_totals": model.context_totals,
    }
    with gzip.open(output_path, "wb") as file:
        pickle.dump(payload, file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a probabilistic n-gram LM on Darija text.")
    parser.add_argument("--corpus-dir", type=Path, default=Path("corpus/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--candidate-orders", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--k", type=float, default=0.5, help="Add-k smoothing value.")
    parser.add_argument("--min-freq", type=int, default=2, help="Replace tokens seen fewer than this with <unk>.")
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sentences, corpus_stats = read_sentences(args.corpus_dir)
    if not sentences:
        raise SystemExit(f"No usable sentences found in {args.corpus_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model, summary = train_and_select(
        sentences=sentences,
        candidate_orders=args.candidate_orders,
        k=args.k,
        min_freq=args.min_freq,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    model_path = args.output_dir / "darija_ngram_model.pkl.gz"
    save_model(model, model_path)

    summary["corpus_stats"] = corpus_stats
    summary["model_path"] = str(model_path)
    summary["sample_generations"] = {
        "ana": model.generate("ana", max_tokens=12),
        "had": model.generate("had", max_tokens=12),
        "allah": model.generate("allah", max_tokens=12),
    }

    metrics_path = args.output_dir / "training_summary.json"
    metrics_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
