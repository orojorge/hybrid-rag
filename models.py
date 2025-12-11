from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class NLQuery:
    text: str


@dataclass
class Intent:
    kind: str # "sql", "text", or "hybrid"
    confidence: float


@dataclass
class Passage:
    doc_id: str
    text: str
    meta: Dict[str, Any]
    score: float


@dataclass
class RetrievalResult:
    sql_rows: Optional[List[Dict[str, Any]]] = None
    passages: Optional[List[Passage]] = None
    diagnostics: Optional[Dict[str, Any]] = None


@dataclass
class Answer:
    text: str
    citations: Dict[str, List[str]]
    traces: Dict[str, Any]
