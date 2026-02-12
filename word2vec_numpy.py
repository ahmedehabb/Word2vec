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


def forward_softmax(
	center_idx: int, outside_idx: int, params: Parameters
) -> Tuple[float, np.ndarray, np.ndarray]:
	"""
	Forward pass: compute softmax probs and loss for a (center, context) pair.

	Returns loss, probabilities over vocab, and the center vector snapshot v_c.
	"""

	v_c = params.center[center_idx]  # v_c
	scores = params.context @ v_c  # u_w^T v_c for all w
	probs = softmax(scores)  # P(w|c)
	loss = -np.log(probs[outside_idx])
	return loss, probs, v_c


def backward_softmax(
	probs: np.ndarray,
	center_idx: int,
	outside_idx: int,
	params: Parameters,
	v_c: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
	"""
	Backward pass: compute gradients given forward outputs.

	Returns grad_center (shape (V, d), only center row nonzero) and
	grad_context (shape (V, d), dense across vocab).
	"""

	grad_context = probs[:, None] * v_c[None, :]  # P(w|c) v_c
	grad_context[outside_idx] -= v_c  # (P(o|c) - 1) v_c at the true word

	expected_context = probs @ params.context  # sum_w P(w|c) u_w
	grad_v_c = expected_context - params.context[outside_idx]

	grad_center = np.zeros_like(params.center)
	grad_center[center_idx] = grad_v_c
	return grad_center, grad_context


def naive_softmax_loss_and_grads(
	center_idx: int, outside_idx: int, params: Parameters
) -> Tuple[float, np.ndarray, np.ndarray]:
	"""
	Compute loss and gradients for one (center, context) pair using full softmax.

	Returns (loss, grad_center, grad_context) where grad_center has shape (V, d)
	but is zero everywhere except the row for center_idx; grad_context has shape
	(V, d) and is dense because every u_w participates in the denominator.
	"""

	loss, probs, v_c = forward_softmax(center_idx, outside_idx, params)
	grad_center, grad_context = backward_softmax(
		probs=probs,
		center_idx=center_idx,
		outside_idx=outside_idx,
		params=params,
		v_c=v_c,
	)
	return loss, grad_center, grad_context


def neg_sampling_loss_and_grads(
	center_idx: int,
	out_idx: int,
	neg_indices: List[int],
	params: Parameters,
) -> Tuple[float, np.ndarray, np.ndarray]:
	"""Negative sampling loss/gradients for one center/out pair and K negatives."""
	v_c = params.center[center_idx]

	# Positive term
	pos_score = params.context[out_idx] @ v_c
	pos_sig = sigmoid(pos_score)
	pos_loss = -np.log(pos_sig + 1e-12)

	# Negative terms
	neg_vectors = params.context[neg_indices]
	neg_scores = neg_vectors @ v_c
	neg_sig = sigmoid(-neg_scores)
	neg_loss = -np.sum(np.log(neg_sig + 1e-12))

	loss = pos_loss + neg_loss

	grad_center_vec = (pos_sig - 1.0) * params.context[out_idx]
	grad_center_vec += np.sum(sigmoid(neg_scores)[:, None] * neg_vectors, axis=0)

	grad_center = np.zeros_like(params.center)
	grad_center[center_idx] = grad_center_vec

	grad_context = np.zeros_like(params.context)
	grad_context[out_idx] = (pos_sig - 1.0) * v_c
	grad_context[neg_indices] += sigmoid(neg_scores)[:, None] * v_c[None, :]

	return loss, grad_center, grad_context


def sgd_step(
	pairs: List[Tuple[int, int]],
	params: Parameters,
	lr: float,
	log_every: int | None = None,
	loss_type: str = "softmax",
	K: int = 5,
	neg_dist: np.ndarray | None = None,
	rng: np.random.Generator | None = None,
) -> float:
	"""One SGD sweep with optional negative sampling and progress logging."""
	total_loss = 0.0
	for idx, (center_idx, outside_idx) in enumerate(pairs, start=1):
		if loss_type == "softmax":
			loss, grad_center, grad_context = naive_softmax_loss_and_grads(
				center_idx, outside_idx, params
			)
		elif loss_type == "neg":
			assert neg_dist is not None and rng is not None
			neg_indices = sample_negative_indices(
				vocab_size=params.context.shape[0],
				dist=neg_dist,
				K=K,
				rng=rng,
				exclude={outside_idx, center_idx},
			)
			loss, grad_center, grad_context = neg_sampling_loss_and_grads(
				center_idx, outside_idx, neg_indices, params
			)
		else:
			raise ValueError(f"unknown loss_type {loss_type}")

		params.center -= lr * grad_center
		params.context -= lr * grad_context
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

	for epoch in range(args.epochs):
		print(f"epoch {epoch}...")
		loss = sgd_step(
			pairs=pairs,
			params=params,
			lr=args.lr,
			log_every=log_every,
			loss_type=args.loss,
			K=args.neg_k,
			neg_dist=neg_dist,
			rng=rng,
		)
		print(
			f"epoch={epoch} loss={loss:.4f} pairs={len(pairs)} V={len(vocab)} dim={args.dim}"
		)

	print("center embeddings shape", params.center.shape)
