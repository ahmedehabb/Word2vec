from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def download_analogies(data_dir: Path = Path("data")) -> Path:
    """Download the Google analogy question set if missing."""
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / "questions-words.txt"
    if path.exists():
        return path
    url = "https://raw.githubusercontent.com/tmikolov/word2vec/master/questions-words.txt"
    print(f"downloading {url} ...")
    import urllib.request

    urllib.request.urlretrieve(url, path)
    return path


def load_analogies(path: Path) -> List[Tuple[str, Tuple[str, str, str, str]]]:
    """Parse analogy questions into (category, (a,b,c,d))."""
    items: List[Tuple[str, Tuple[str, str, str, str]]] = []
    current_cat = ""
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        if line.startswith(":"):
            current_cat = line[1:].strip()
            continue
        parts = line.strip().split()
        if len(parts) != 4:
            continue
        items.append((current_cat, (parts[0], parts[1], parts[2], parts[3])))
    return items


def analogy_accuracy(
    vocab_to_idx: Dict[str, int],
    embeddings: np.ndarray,
    questions: List[Tuple[str, Tuple[str, str, str, str]]],
    topn: int = 1,
) -> Tuple[Dict[str, float], Dict[str, int]]:
    """Compute analogy accuracy overall and per category.

    Returns (accuracies, counts) where accuracies are fractions in [0,1] and counts
    contains raw integers for 'evaluated' and 'skipped'.
    """

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
    emb_norm = embeddings / norms
    idx = vocab_to_idx

    correct_per_cat: Dict[str, int] = {}
    total_per_cat: Dict[str, int] = {}
    skipped = 0

    for cat, (a, b, c, d) in questions:
        if a not in idx or b not in idx or c not in idx or d not in idx:
            skipped += 1
            continue
        va, vb, vc = emb_norm[idx[a]], emb_norm[idx[b]], emb_norm[idx[c]]
        query = vb - va + vc
        query_norm = query / (np.linalg.norm(query) + 1e-9)
        scores = emb_norm @ query_norm
        scores[[idx[a], idx[b], idx[c]]] = -np.inf
        top_pred = int(np.argmax(scores)) if topn == 1 else int(np.argsort(scores)[-topn:][::-1][0])

        cat_key = cat or "overall"
        total_per_cat[cat_key] = total_per_cat.get(cat_key, 0) + 1
        if top_pred == idx[d]:
            correct_per_cat[cat_key] = correct_per_cat.get(cat_key, 0) + 1

    total = sum(total_per_cat.values())
    correct = sum(correct_per_cat.values())
    acc: Dict[str, float] = {"overall": correct / total if total else 0.0}
    for cat_key, tot in total_per_cat.items():
        acc[cat_key] = correct_per_cat.get(cat_key, 0) / tot

    counts: Dict[str, int] = {"evaluated": int(total), "skipped": int(skipped)}
    return acc, counts
