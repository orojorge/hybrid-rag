from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Protocol, runtime_checkable
from enum import Enum, auto

@dataclass
class NLQuery:
    text: str
    user_context: Dict[str, Any] | None = None

@dataclass
class Intent:
    kind: str # "lookup_sql" | "lookup_text" | "hybrid"
    confidence: float
    constraints: Dict[str, Any]

@dataclass
class Entity:
    name: str
    type: str # "lang", "package", "cve", "date", etc.
    value: Any
    span: Tuple[int, int]

@dataclass
class SubQuery:
    id: str
    modality: str # "sql" | "text"
    goal: str
    constraints: Dict[str, Any]

@dataclass
class QueryPlan:
    steps: List[SubQuery] # execution order matters
    rationale: str

@dataclass
class SQLSpec:
    tables: List[str]
    select: List[str]
    where: Dict[str, Any]
    group_by: List[str] | None = None
    order_by: List[Tuple[str, str]] | None = None
    limit: int = 50

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
    diagnostics: Dict[str, Any] | None = None

@dataclass
class Answer:
    text: str
    citations: Dict[str, List[str]] # doc_ids, table refs
    traces: Dict[str, Any]


# - - - - - NL Answer: Data Models- - - - -
@dataclass
class AnswerSections:
    """Intermediary structure used by the surface realizer (how to phrase)"""
    thesis: Optional[str] = None # one-line direct answer
    key_facts: List[str] = field(default_factory=list)
    explanation: List[str] = field(default_factory=list)
    next_step: Optional[str] = None
    uncertainty_note: Optional[str] = None
    citations_inline: List[str] = field(default_factory=list)

class GoalType(Enum):
    """Structure used by AnswerComposer"""
    AGGREGATE_LOOKUP = auto()
    DETAIL_FETCH = auto()
    TREND_ANALYSIS = auto()
    COMPARE = auto()
    RETRIEVE_CONTEXT = auto()
    MITIGATION_LOOKUP = auto()

@dataclass
class EvidenceRow:
    """A normalized, typed row from SQL for grounding."""
    table: str
    values: Dict[str, Any] # e.g., {"severity":"critical","cnt":2,"package":"flask"}
    weight: float = 1.0

@dataclass
class EvidencePassage:
    """A normalized, typed passage from vector/text store."""
    doc_id: str
    text: str
    score: float
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EvidenceBundle:
    """Selected evidence ready for NLG (post fusion, post selection)."""
    sql: List[EvidenceRow] = field(default_factory=list)
    passages: List[EvidencePassage] = field(default_factory=list)

@dataclass
class Claim:
    """Atomic statement the answer will assert, auditable."""
    text: str
    kind: str # "fact" | "explanation" | "action"
    grounded_in_sql: bool = False
    grounded_in_text: bool = False
    evidence_refs: List[str] = field(default_factory=list)
    confidence: float = 0.0

@dataclass
class AnswerSpec:
    """Template-level knobs for realization."""
    max_words: int = 140
    min_words: int = 60
    inline_citations: bool = True
    style_concise: bool = True
    goal_hint: Optional[GoalType] = None

@dataclass
class ComposedAnswer:
    """Final object returned by the synthesizer."""
    text: str
    citations: Dict[str, List[str]]
    claims: List[Claim]
    meta: Dict[str, Any] # e.g., {"grounding_score":0.92, "length":118}


# - - - - - NL Answer: Interfaces- - - - -
@runtime_checkable
class ContentSelector(Protocol):
    """Chooses which rows/passages to talk about, given fused retrieval."""
    def select(self, fused: EvidenceBundle, q: NLQuery, plan: QueryPlan) -> EvidenceBundle: ...

@runtime_checkable
class FactExtractor(Protocol):
    """Turns selected SQL rows into human facts with normalized labels/units."""
    def to_facts(self, evidence: EvidenceBundle, q: NLQuery) -> List[Claim]: ...

@runtime_checkable
class RationaleExtractor(Protocol):
    """Turns selected passages into explanation/why/how claims."""
    def to_explanations(self, evidence: EvidenceBundle, q: NLQuery) -> List[Claim]: ...

@runtime_checkable
class MicroPlanner(Protocol):
    """Arranges claims into a narrative plan (thesis → facts → explanation → action)."""
    def plan(self, q: NLQuery, spec: AnswerSpec, claims: List[Claim], goal: GoalType) -> AnswerSections: ...

@runtime_checkable
class SurfaceRealizer(Protocol):
    """Converts planned sections into final text with cohesion and length control."""
    def realize(self, sections: AnswerSections, spec: AnswerSpec) -> str: ...

@runtime_checkable
class CitationManager(Protocol):
    """Builds inline anchors and structured citation maps from claims/evidence."""
    def build(self, claims: List[Claim], evidence: EvidenceBundle) -> Tuple[List[str], Dict[str, List[str]]]: ...

@runtime_checkable
class ClaimAuditor(Protocol):
    """Checks that numeric/textual facts are supported by evidence; assigns confidence."""
    def audit(self, claims: List[Claim], evidence: EvidenceBundle) -> Tuple[List[Claim], float]:  # (claims, grounding_score)
        ...

@runtime_checkable
class Redactor(Protocol):
    """Applies PII/secret redaction to the final text and claims."""
    def redact(self, text: str, claims: List[Claim]) -> Tuple[str, List[Claim]]: ...

@runtime_checkable
class LengthController(Protocol):
    """Ensures answer fits desired length budget without losing key info."""
    def fit(self, text: str, spec: AnswerSpec, sections: AnswerSections) -> str: ...


# - - - - - NL Answer Orchestrator- - - - -
class AnswerComposer:
    """
    High-level façade the pipeline calls instead of the old AnswerSynthesizer.
    It wires: selection → extraction → planning → realization → audit → citation → redaction → length fit.
    """
    def __init__(
        self,
        selector: ContentSelector,
        fact_extractor: FactExtractor,
        rationale_extractor: RationaleExtractor,
        microplanner: MicroPlanner,
        realizer: SurfaceRealizer,
        auditor: ClaimAuditor,
        citer: CitationManager,
        redactor: Redactor,
        length_ctrl: LengthController,
    ):
        self.selector = selector
        self.fact_extractor = fact_extractor
        self.rationale_extractor = rationale_extractor
        self.microplanner = microplanner
        self.realizer = realizer
        self.auditor = auditor
        self.citer = citer
        self.redactor = redactor
        self.length_ctrl = length_ctrl

    def compose(
        self,
        q: NLQuery,
        plan: QueryPlan,
        fused_retrieval: EvidenceBundle,
        spec: AnswerSpec,
        goal: GoalType
    ) -> ComposedAnswer:
        # 1) Select
        selected = self.selector.select(fused_retrieval, q, plan)
        # 2) Extract claims
        fact_claims = self.fact_extractor.to_facts(selected, q)
        rationale_claims = self.rationale_extractor.to_explanations(selected, q)
        claims = fact_claims + rationale_claims
        # 3) Micro-plan
        sections = self.microplanner.plan(q, spec, claims, goal)
        # 4) Realize surface text
        text = self.realizer.realize(sections, spec)
        # 5) Audit & confidence
        audited_claims, grounding_score = self.auditor.audit(claims, selected)
        # 6) Citations
        inline_cits, citation_map = self.citer.build(audited_claims, selected)
        if spec.inline_citations and inline_cits:
            sections.citations_inline = inline_cits
            text = self.realizer.realize(sections, spec)  # re-realize to insert anchors if needed
        # 7) Redact
        text, audited_claims = self.redactor.redact(text, audited_claims)
        # 8) Length control
        text = self.length_ctrl.fit(text, spec, sections)
        # Done
        meta = {
            "grounding_score": grounding_score,
            "length": len(text.split()),
            "goal": goal.name,
        }
        return ComposedAnswer(text=text, citations=citation_map, claims=audited_claims, meta=meta)