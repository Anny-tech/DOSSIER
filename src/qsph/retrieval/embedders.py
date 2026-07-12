"""Embedding models for retrieval.

The embedder maps text to a vector so passages can be ranked by similarity to a
query. Two concerns kept separate on purpose:

  * BaseEmbedder + cosine: the interface every retriever depends on.
  * Concrete embedders: a real HuggingFace one (PhysBERT by default, MatSciBERT
    as the documented ablation arm) that needs torch + weights, and a
    dependency-free HashingEmbedder for offline tests of the retrieval plumbing.

The HashingEmbedder is explicitly NOT semantic -- it exists so the chunker,
retriever, and evidence interface can be tested end-to-end in CI without
downloading models. Real retrieval quality claims must use a real encoder.

PhysBERT is the sensible default: ComProScanner reported it beating
all-mpnet-base-v2 on materials/physics vocabulary, and it is the same embedding
family already in scope for this project.
"""

from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod


class BaseEmbedder(ABC):
    """Maps texts to dense vectors; provides cosine similarity."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text."""

    @staticmethod
    def cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b, strict=True))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na == 0.0 or nb == 0.0:
            return 0.0
        return dot / (na * nb)


class HuggingFaceEmbedder(BaseEmbedder):  # pragma: no cover - needs torch+weights
    """Encoder-based embedder. Default PhysBERT; MatSciBERT for the ablation.

    Mean-pools the last hidden state, the standard way to turn a BERT-family
    encoder into a sentence embedding.
    """

    def __init__(self, model_name: str = "thellert/physbert_cased"):
        try:
            import torch  # noqa: PLC0415
            from transformers import AutoModel, AutoTokenizer  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "HuggingFaceEmbedder requires: pip install 'qsph[retrieval]'"
            ) from exc
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name)
        self._model.eval()

    def embed(self, texts: list[str]) -> list[list[float]]:
        torch = self._torch
        enc = self._tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt", max_length=512
        )
        with torch.no_grad():
            out = self._model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        summed = (out * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return (summed / counts).tolist()


class HashingEmbedder(BaseEmbedder):
    """Deterministic, dependency-free embedder for offline plumbing tests only.

    Hashes token unigrams into a fixed-width vector. Carries only lexical-overlap
    signal (shared words -> higher similarity), which is enough to test that the
    retriever ranks a passage mentioning the query terms above one that does
    not. NEVER use for real retrieval-quality claims.
    """

    def __init__(self, dim: int = 128):
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            for tok in text.lower().split():
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                vec[h % self.dim] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            vectors.append([v / norm for v in vec])
        return vectors
