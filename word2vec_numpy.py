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
from typing import Iterable, List, Tuple

import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
	"""Numerically stable softmax over the last axis."""
	x_shifted = x - np.max(x)
	exps = np.exp(x_shifted)
	return exps / np.sum(exps)


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


def sgd_step(
	pairs: List[Tuple[int, int]], params: Parameters, lr: float
) -> float:
	"""One SGD sweep over provided (center_idx, outside_idx) pairs."""
	total_loss = 0.0
	for center_idx, outside_idx in pairs:
		loss, grad_center, grad_context = naive_softmax_loss_and_grads(
			center_idx, outside_idx, params
		)
		params.center -= lr * grad_center
		params.context -= lr * grad_context
		total_loss += loss
	return total_loss / max(1, len(pairs))


def demo() -> None:
	"""Small sanity check on a toy corpus."""
	corpus = "king queen man woman king woman".split()
	vocab = Vocab(tokens=sorted(set(corpus)))
	pairs: List[Tuple[int, int]] = []

	window = 1
	for i, token in enumerate(corpus):
		center_idx = vocab.idx(token)
		for j in range(max(0, i - window), min(len(corpus), i + window + 1)):
			if i == j:
				continue
			pairs.append((center_idx, vocab.idx(corpus[j])))

	params = init_params(vocab_size=len(vocab), dim=8)
	lr = 0.1

	# Simple training loop for demonstration; not optimized.
	for epoch in range(100):
		# Re-create generator because we consume pairs to count size in sgd_step.
		loss = sgd_step(pairs=list(pairs), params=params, lr=lr)
		print(f"epoch={epoch} loss={loss:.4f}")

	print("center embeddings shape", params.center.shape)


if __name__ == "__main__":
	demo()
