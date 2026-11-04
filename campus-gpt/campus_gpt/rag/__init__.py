"""BM25 retrieval for CampusGPT chunk corpora.

The retriever loads a JSONL file produced by
``python -m campus_gpt.ingest.build`` and ranks chunks with
:class:`rank_bm25.BM25Okapi`.  Subject and module filters are applied before
ranking, and results below ``min_score`` are discarded.

An optional semantic fallback is available when ``use_embeddings=True`` and the
optional ``sentence-transformers`` package is installed.  It is imported lazily
so the base package has no heavyweight model dependency.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from rank_bm25 import BM25Okapi

__all__ = ["Retriever", "STOPWORDS", "tokenize"]

LOGGER = logging.getLogger(__name__)

# Small, hand-picked English stopword list.  Subject codes such as TST101 are
# never removed because they are not in this set.
STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "being",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "he",
        "her",
        "hers",
        "him",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "may",
        "me",
        "might",
        "my",
        "of",
        "on",
        "or",
        "our",
        "ours",
        "she",
        "should",
        "so",
        "than",
        "that",
        "the",
        "their",
        "theirs",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "to",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
        "yours",
    }
)

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> List[str]:
    """Lowercase, split on non-alphanumeric characters, and drop stopwords.

    Tokens such as ``TST101`` stay as a single token (``tst101``).
    """

    if not text:
        return []
    return [
        token
        for token in (match.group(0).lower() for match in _TOKEN_RE.finditer(text))
        if token not in STOPWORDS
    ]


class Retriever:
    """Rank chunk dictionaries from a JSONL corpus using BM25.

    Parameters
    ----------
    chunks_path:
        Path to a JSONL file where each line is a chunk dictionary.
    use_embeddings:
        If true, try to use ``sentence-transformers`` as a semantic fallback
        when BM25 finds nothing above ``min_score``.  The import is lazy and a
        missing optional dependency is not an error.
    """

    def __init__(self, chunks_path: str | Path, use_embeddings: bool = False) -> None:
        self.chunks_path = str(chunks_path)
        self.use_embeddings = bool(use_embeddings)
        #: Minimum BM25 score required for a non-empty result.
        self.min_score: float = 0.1

        self.chunks: List[Dict[str, Any]] = self._load_chunks(self.chunks_path)
        self._tokenized_chunks: List[List[str]] = [
            tokenize(str(chunk.get("text", ""))) for chunk in self.chunks
        ]

        #: BM25 index, or ``None`` when the corpus is empty.
        self.bm25: Optional[BM25Okapi] = (
            BM25Okapi(self._tokenized_chunks) if self._tokenized_chunks else None
        )

        # Lazily-initialised semantic fallback state.
        self._embedding_model: Any = None
        self._chunk_embeddings: Any = None
        self._embedding_import_failed = False

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    @staticmethod
    def _load_chunks(chunks_path: str | Path) -> List[Dict[str, Any]]:
        """Load chunk dictionaries from a JSONL file."""

        path = Path(chunks_path)
        if not path.exists():
            raise FileNotFoundError(f"Chunks file not found: {path}")

        chunks: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    LOGGER.warning(
                        "Skipping malformed JSON on line %d of %s",
                        line_number,
                        path,
                    )
                    continue
                if not isinstance(value, dict):
                    LOGGER.warning(
                        "Skipping non-object JSON on line %d of %s",
                        line_number,
                        path,
                    )
                    continue
                chunks.append(value)

        return chunks

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def _matches_filters(
        self,
        chunk: Dict[str, Any],
        subject: Optional[str],
        module: Optional[int],
    ) -> bool:
        if subject is not None:
            chunk_subject = chunk.get("subject")
            if chunk_subject is None:
                return False
            if str(chunk_subject).strip().upper() != str(subject).strip().upper():
                return False

        if module is not None:
            chunk_module = chunk.get("module")
            if chunk_module is None:
                return False
            try:
                if int(chunk_module) != int(module):
                    return False
            except (TypeError, ValueError):
                return False

        return True

    def retrieve(
        self,
        query: str,
        k: int = 5,
        subject: Optional[str] = None,
        module: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Return the ``k`` best chunks for ``query``.

        Filters are applied before scoring, so a filtered-out chunk can never
        displace an in-scope result.  If the best score is below
        :attr:`min_score`, an empty list is returned.
        """

        if k <= 0 or not self.chunks or self.bm25 is None:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        candidate_indices = [
            index
            for index, chunk in enumerate(self.chunks)
            if self._matches_filters(chunk, subject=subject, module=module)
        ]
        if not candidate_indices:
            return []

        scores = self.bm25.get_scores(query_tokens)
        ranked = sorted(
            ((float(scores[index]), index) for index in candidate_indices),
            key=lambda item: (-item[0], item[1]),
        )

        # BM25Okapi's IDF is not positive for terms that occur in a large
        # fraction of a tiny corpus (and is negative for a one-document
        # corpus).  In that case all raw scores can be <= 0 even when there is
        # a genuine term match.  Fall back to a deterministic overlap score so
        # small synthetic corpora remain useful.
        if ranked and ranked[0][0] <= 0:
            ranked = self._overlap_scores(query_tokens, candidate_indices)

        results = [
            {"chunk": self.chunks[index], "score": score}
            for score, index in ranked[:k]
        ]

        if results and results[0]["score"] >= self.min_score:
            return results

        # BM25 found nothing sufficiently relevant.  If requested, try the
        # optional semantic fallback.
        if self.use_embeddings:
            fallback = self._embedding_retrieve(
                query, candidate_indices, k=k, min_score=self.min_score
            )
            if fallback:
                return fallback

        return []

    def _overlap_scores(
        self,
        query_tokens: Sequence[str],
        candidate_indices: Sequence[int],
    ) -> List[tuple[float, int]]:
        """Small-corpus scoring fallback used when BM25 scores are non-positive."""

        query_set = set(query_tokens)
        ranked: List[tuple[float, int]] = []
        for index in candidate_indices:
            tokens = self._tokenized_chunks[index]
            matched = query_set.intersection(tokens)
            if not matched:
                score = 0.0
            else:
                coverage = len(matched) / len(query_set)
                term_frequency = sum(tokens.count(term) for term in matched)
                score = 0.5 + coverage + term_frequency / (len(tokens) + 1)
            ranked.append((float(score), index))

        ranked.sort(key=lambda item: (-item[0], item[1]))
        return ranked

    # ------------------------------------------------------------------
    # Optional semantic fallback
    # ------------------------------------------------------------------
    def _load_embedding_model(self) -> Any:
        """Lazily import and construct a sentence-transformers model."""

        if self._embedding_import_failed:
            return None
        if self._embedding_model is not None:
            return self._embedding_model

        try:
            from sentence_transformers import SentenceTransformer

            self._embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as exc:  # optional dependency or model download failed
            self._embedding_import_failed = True
            LOGGER.warning(
                "Embedding fallback unavailable (%s); using BM25 only", exc
            )
            return None

        return self._embedding_model

    def _embedding_retrieve(
        self,
        query: str,
        candidate_indices: Sequence[int],
        k: int,
        min_score: float,
    ) -> List[Dict[str, Any]]:
        """Rank candidates by cosine similarity as a best-effort fallback."""

        model = self._load_embedding_model()
        if model is None:
            return []

        try:
            import numpy as np

            if self._chunk_embeddings is None:
                texts = [str(self.chunks[i].get("text", "")) for i in range(len(self.chunks))]
                self._chunk_embeddings = model.encode(
                    texts, normalize_embeddings=True, show_progress_bar=False
                )

            query_vector = model.encode(
                [query], normalize_embeddings=True, show_progress_bar=False
            )[0]
            chunk_vectors = np.asarray(self._chunk_embeddings)
            similarities = chunk_vectors @ np.asarray(query_vector)
            ranked = sorted(
                (
                    (float(similarities[index]), index)
                    for index in candidate_indices
                ),
                key=lambda item: (-item[0], item[1]),
            )
        except Exception as exc:  # pragma: no cover - depends on optional package
            LOGGER.warning("Embedding retrieval failed: %s", exc)
            return []

        if not ranked or ranked[0][0] < min_score:
            return []

        return [
            {"chunk": self.chunks[index], "score": score}
            for score, index in ranked[:k]
        ]


# Convenient module-level alias for callers that only need the tokenizer.
_tokenize = tokenize
