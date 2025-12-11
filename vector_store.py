from typing import Dict, List
from pathlib import Path
import time

from sentence_transformers import SentenceTransformer

from models import Passage, RetrievalResult


class VectorStore:
    """
    Tiny in-memory vector store:
    - loads a corpus from disk
    - embeds documents
    - performs dot-product similarity search
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self._vecs: Dict[str, List[float]] = {}
        self._docs: Dict[str, str] = {}

    def load_corpus(self, folder: str = "corpus") -> None:
        docs: Dict[str, str] = {}
        for path in Path(folder).glob("*"):
            if path.is_file():
                docs[path.stem] = path.read_text()
        self._docs = docs

        if not docs:
            self._vecs = {}
            return

        texts = list(docs.values())
        embeddings = self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=False)

        self._vecs = {}
        for doc_id, vec in zip(docs.keys(), embeddings):
            self._vecs[doc_id] = list(vec)

    def search(self, query: str, top_k: int = 3) -> RetrievalResult:
        if not self._vecs:
            return RetrievalResult(sql_rows=None, passages=[], diagnostics={"note": "empty corpus"})

        t0 = time.time()
        qvec = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=False)[0]
        qvec = list(qvec)

        def dot(a: List[float], b: List[float]) -> float:
            return float(sum(x * y for x, y in zip(a, b)))

        scored = [(doc_id, dot(qvec, v)) for doc_id, v in self._vecs.items()]
        scored.sort(key=lambda x: x[1], reverse=True)
        hits = scored[:top_k]

        passages = [
            Passage(
                doc_id=doc_id,
                text=self._docs.get(doc_id, ""),
                meta={},
                score=float(score),
            )
            for doc_id, score in hits
        ]

        diagnostics = {
            "latency_ms": (time.time() - t0) * 1000,
            "hits": hits,
        }
        return RetrievalResult(sql_rows=None, passages=passages, diagnostics=diagnostics)
