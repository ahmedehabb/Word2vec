"""
Minimal, pedagogy-first Word2Vec (naive softmax) using only NumPy.

We stay close to the math laid out in the prompt:
  P(o|c) = exp(u_o^T v_c) / sum_w exp(u_w^T v_c)
  J      = -log P(o|c)
with separate center vectors V (v_i) and context vectors U (u_i).

This file purposefully keeps full softmax (no negatives) to match the
derivation, even though it is inefficient for large vocabularies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from datasets import load_dataset

from validations import download_analogies, load_analogies, analogy_accuracy


def softmax(x: np.ndarray) -> np.ndarray:
	"""Numerically stable softmax over the last axis."""
	x_shifted = x - np.max(x)
	exps = np.exp(x_shifted)
	return exps / np.sum(exps)


def sigmoid(x: np.ndarray) -> np.ndarray:
	"""Numerically stable sigmoid."""
	return 1.0 / (1.0 + np.exp(-x))


@dataclass
class Vocab:
	tokens: List[str]

	def __post_init__(self) -> None:
		self.token_to_idx = {t: i for i, t in enumerate(self.tokens)}

	def __len__(self) -> int:  # convenience
		return len(self.tokens)

	def idx(self, token: str) -> int:
		return self.token_to_idx[token]


@dataclass
class Parameters:
	center: np.ndarray  # shape: (V, d)
	context: np.ndarray  # shape: (V, d)


class Word2VecModel:
	"""Base interface for word2vec variants."""

	def __init__(self, params: Parameters) -> None:
		self.params = params

	def forward(self, center_idx: int, context_idx: int):
		raise NotImplementedError

	def backward(self, cache):
		raise NotImplementedError


class SoftmaxModel(Word2VecModel):
	"""Full softmax word2vec."""

	def forward(self, center_idx: int, context_idx: int):
		v_c = self.params.center[center_idx]
		scores = self.params.context @ v_c
		probs = softmax(scores)
		loss = -np.log(probs[context_idx])
		cache = (center_idx, context_idx, v_c, probs)
		return loss, cache

	def backward(self, cache):
		center_idx, context_idx, v_c, probs = cache
		grad_context = probs[:, None] * v_c[None, :]
		grad_context[context_idx] -= v_c
		expected_context = probs @ self.params.context
		grad_v_c = expected_context - self.params.context[context_idx]
		grad_center = np.zeros_like(self.params.center)
		grad_center[center_idx] = grad_v_c
		return grad_center, grad_context


class NegSamplingModel(Word2VecModel):
	"""Negative sampling word2vec."""

	def __init__(
		self,
		params: Parameters,
		K: int,
		neg_dist: np.ndarray,
		rng: np.random.Generator,
	) -> None:
		super().__init__(params)
		self.K = K
		self.neg_dist = neg_dist
		self.rng = rng

	def forward(self, center_idx: int, context_idx: int):
		v_c = self.params.center[center_idx]
		neg_indices = sample_negative_indices(
			vocab_size=self.params.context.shape[0],
			dist=self.neg_dist,
			K=self.K,
			rng=self.rng,
			exclude={center_idx, context_idx},
		)

		pos_score = self.params.context[context_idx] @ v_c
		pos_sig = sigmoid(pos_score)
		neg_vectors = self.params.context[neg_indices]
		neg_scores = neg_vectors @ v_c
		neg_sig = sigmoid(-neg_scores)

		pos_loss = -np.log(pos_sig + 1e-12)
		neg_loss = -np.sum(np.log(neg_sig + 1e-12))
		loss = pos_loss + neg_loss

		cache = (
			center_idx,
			context_idx,
			v_c,
			pos_sig,
			neg_scores,
			neg_indices,
		)
		return loss, cache

	def backward(self, cache):
		center_idx, context_idx, v_c, pos_sig, neg_scores, neg_indices = cache
		neg_sig_pos = sigmoid(neg_scores)

		grad_center_vec = (pos_sig - 1.0) * self.params.context[context_idx]
		grad_center_vec += np.sum(neg_sig_pos[:, None] * self.params.context[neg_indices], axis=0)

		grad_center = np.zeros_like(self.params.center)
		grad_center[center_idx] = grad_center_vec

		grad_context = np.zeros_like(self.params.context)
		grad_context[context_idx] = (pos_sig - 1.0) * v_c
		grad_context[neg_indices] += neg_sig_pos[:, None] * v_c[None, :]

		return grad_center, grad_context


def init_params(vocab_size: int, dim: int, seed: int = 42) -> Parameters:
	"""Small random init to break symmetry."""
	rng = np.random.default_rng(seed)
	bound = 1.0 / dim**0.5
	center = rng.uniform(-bound, bound, size=(vocab_size, dim))
	context = rng.uniform(-bound, bound, size=(vocab_size, dim))
	return Parameters(center=center, context=context)


def load_text8_tokens(max_tokens: int | None = None) -> List[str]:
	"""Load text8 tokens via Hugging Face datasets (afmck/text8)."""

	ds = load_dataset("afmck/text8", split="train")
	texts = ds["text"] if "text" in ds.column_names else []
	if not texts:
		raise ValueError("text8 dataset missing 'text' column")

	text = " ".join(texts)
	tokens = text.split()
	if max_tokens is not None:
		tokens = tokens[:max_tokens]
	return tokens


def build_vocab_from_tokens(tokens: List[str], max_size: int) -> Tuple[Vocab, np.ndarray]:
	"""Keep the top max_size most frequent tokens (same cutoff idea as the paper)."""
	counts = {}
	for t in tokens:
		counts[t] = counts.get(t, 0) + 1
	sorted_tokens = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
	keep_items = sorted_tokens[:max_size]
	keep_tokens = [t for t, _ in keep_items]
	keep_freqs = np.array([f for _, f in keep_items], dtype=np.float64)
	return Vocab(tokens=keep_tokens), keep_freqs


def generate_pairs(tokens: List[str], vocab: Vocab, window: int) -> List[Tuple[int, int]]:
	pairs: List[Tuple[int, int]] = []
	lookup = vocab.token_to_idx
	for i, tok in enumerate(tokens):
		if tok not in lookup:
			continue
		center_idx = lookup[tok]
		start = max(0, i - window)
		end = min(len(tokens), i + window + 1)
		for j in range(start, end):
			if i == j:
				continue
			ctx_tok = tokens[j]
			if ctx_tok not in lookup:
				continue
			pairs.append((center_idx, lookup[ctx_tok]))
	return pairs


def make_unigram_distribution(freqs: np.ndarray, power: float = 0.75) -> np.ndarray:
	"""Unigram distribution with exponent, normalized to 1."""
	adjusted = freqs**power
	return adjusted / adjusted.sum()




def sample_negative_indices(
	vocab_size: int, dist: np.ndarray, K: int, rng: np.random.Generator, exclude: set
) -> List[int]:
	"""Sample K negatives, avoiding excluded indices."""
	negatives: List[int] = []
	while len(negatives) < K:
		candidate = rng.choice(vocab_size, p=dist)
		if candidate in exclude:
			continue
		negatives.append(int(candidate))
	return negatives


"""Model-specific forward/backward live in the model classes below."""


def sgd_step(
	pairs: List[Tuple[int, int]],
	model: Word2VecModel,
	lr: float,
	log_every: int | None = None,
) -> float:
	"""One SGD sweep using the provided model."""
	total_loss = 0.0
	for idx, (center_idx, outside_idx) in enumerate(pairs, start=1):
		loss, cache = model.forward(center_idx, outside_idx)
		grad_center, grad_context = model.backward(cache)

		model.params.center -= lr * grad_center
		model.params.context -= lr * grad_context
		total_loss += loss
		if log_every and idx % log_every == 0:
			avg_loss = total_loss / idx
			print(
				f"  progress: {idx}/{len(pairs)} ({idx/len(pairs):.1%}), "
				f"avg_loss={avg_loss:.4f}, last_loss={loss:.4f}"
			)
	return total_loss / max(1, len(pairs))


if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument("--mode", choices=["toy", "text8"], default="toy")
	parser.add_argument("--max_tokens", type=int, default=200_000)
	parser.add_argument("--vocab_size", type=int, default=5_000)
	parser.add_argument("--dim", type=int, default=100)
	parser.add_argument("--window", type=int, default=2)
	parser.add_argument("--epochs", type=int, default=3)
	parser.add_argument("--lr", type=float, default=0.025)
	parser.add_argument("--loss", choices=["softmax", "neg"], default="softmax")
	parser.add_argument("--neg_k", type=int, default=5, help="number of negatives per pair")
	parser.add_argument("--eval_analogies", action="store_true", help="evaluate Google analogies each epoch")
	args = parser.parse_args()

	if args.mode == "toy":
		corpus = "king queen man woman king woman".split()
		tokens = corpus
		window = 1
		vocab = Vocab(tokens=sorted(set(tokens)))
		freqs = np.ones(len(vocab), dtype=np.float64)
	else:
		tokens = load_text8_tokens(max_tokens=args.max_tokens)
		window = args.window
		vocab, freqs = build_vocab_from_tokens(tokens, max_size=args.vocab_size)

	pairs = generate_pairs(tokens, vocab, window=window)
	params = init_params(vocab_size=len(vocab), dim=args.dim)
	log_every = max(1, len(pairs) // 10)
	rng = np.random.default_rng(123)
	neg_dist = make_unigram_distribution(freqs) if args.loss == "neg" else None

	if args.loss == "softmax":
		model: Word2VecModel = SoftmaxModel(params)
	else:
		assert neg_dist is not None
		model = NegSamplingModel(params=params, K=args.neg_k, neg_dist=neg_dist, rng=rng)

	for epoch in range(args.epochs):
		print(f"epoch {epoch}...")
		loss = sgd_step(pairs=pairs, model=model, lr=args.lr, log_every=log_every)
		print(
			f"epoch={epoch} loss={loss:.4f} pairs={len(pairs)} V={len(vocab)} dim={args.dim}"
		)

		if args.eval_analogies:
			analogies_path = download_analogies()
			questions = load_analogies(analogies_path)
			emb = model.params.center
			acc, counts = analogy_accuracy(vocab.token_to_idx, emb, questions)
			acc_items = ", ".join(
				f"{k}: {v*100:.2f}%" for k, v in sorted(acc.items())
			)
			print(
				f"analogies accuracy: {acc_items}; evaluated={counts['evaluated']} skipped={counts['skipped']}"
			)
