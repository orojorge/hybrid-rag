from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import argparse
import json
import sqlite3
import sys
import time
import uuid
from data_model import ( ContentSelector, FactExtractor, RationaleExtractor, MicroPlanner, SurfaceRealizer,
    ClaimAuditor, CitationManager, Redactor, LengthController, EvidenceBundle, AnswerSpec, GoalType,
    ComposedAnswer, AnswerComposer, Claim, AnswerSections, EvidenceRow, EvidencePassage )


# - - - - - Data models - - - - -
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


# - - - - - Components - - - - -
class QueryUnderstanding:
    def classify_intent(self, q: NLQuery) -> Intent:
        text = q.text.lower()
        # Extremely simple heuristic for the skeleton
        if any(k in text for k in ["count", "list", "how many", "top", "most", "least"]):
            if any(k in text for k in ["why", "how to", "explain", "mitigate", "mitigation"]):
                kind = "hybrid"
            else:
                kind = "lookup_sql"
        elif any(k in text for k in ["explain", "why", "how"]):
            kind = "lookup_text"
        else:
            kind = "hybrid"
        return Intent(kind=kind, confidence=0.55, constraints={})

    def extract_entities(self, q: NLQuery) -> List[Entity]:
        # Toy entity detection for demo only
        ents: List[Entity] = []
        lower = q.text.lower()
        for lang in ["python", "javascript", "java", "go", "rust"]:
            idx = lower.find(lang)
            if idx >= 0:
                ents.append(Entity(name="language", type="lang", value=lang, span=(idx, idx+len(lang))))
        return ents


class QueryDecomposer:
    def decompose(self, q: NLQuery, intent: Intent, entities: List[Entity]) -> List[SubQuery]:
        subs: List[SubQuery] = []
        lang = next((e.value for e in entities if e.type == "lang"), None)

        def sid() -> str: return str(uuid.uuid4())[:8]

        if intent.kind == "lookup_sql":
            subs.append(SubQuery(id=sid(), modality="sql", goal="aggregate/lookup", constraints={"language": lang}))
        elif intent.kind == "lookup_text":
            subs.append(SubQuery(id=sid(), modality="text", goal="retrieve/context", constraints={"language": lang}))
        else:  # hybrid
            subs.append(SubQuery(id=sid(), modality="sql", goal="aggregate/lookup", constraints={"language": lang}))
            subs.append(SubQuery(id=sid(), modality="text", goal="retrieve/context", constraints={"language": lang}))

        return subs


class Planner:
    def build_plan(self, subqs: List[SubQuery]) -> QueryPlan:
        # Simple rule: run SQL before text so we can use IDs later (future enhancement)
        steps = sorted(subqs, key=lambda s: 0 if s.modality == "sql" else 1)
        return QueryPlan(steps=steps, rationale="SQL first to prime context; then text for explanations.")


class SchemaInspector:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def describe(self) -> Dict[str, Any]:
        # Return minimal schema info for demo
        return {"tables": {"vulns": ["id", "package", "language", "severity", "year"]}}

    def find_columns(self, semantic_hint: str) -> List[Tuple[str, str]]:
        # Heuristic mapping for demo
        mapping = {
            "language": [("vulns", "language")],
            "severity": [("vulns", "severity")],
            "package":  [("vulns", "package")],
            "year":     [("vulns", "year")],
        }
        return mapping.get(semantic_hint, [])


class SQLQueryBuilder:
    def to_sql(self, spec: SQLSpec, schema: Dict[str, Any]) -> Tuple[str, List[Any]]:
        # Whitelist tables/columns
        allowed_tables = set(schema["tables"].keys())
        for t in spec.tables:
            if t not in allowed_tables:
                raise ValueError(f"Table not allowed: {t}")
        allowed_cols = set()
        for t in spec.tables:
            for c in schema["tables"][t]:
                allowed_cols.add((t, c))

        def col_ok(c: str) -> bool:
            if "." in c:
                t, cc = c.split(".", 1)
                return (t, cc) in allowed_cols
            # single column: allow if present in any table
            return any((t, c) in allowed_cols for t in spec.tables)

        selects = [c for c in spec.select if col_ok(c)]
        if not selects:
            selects = ["*"]

        sql = f"SELECT {', '.join(selects)} FROM {', '.join(spec.tables)}"
        params: List[Any] = []

        if spec.where:
            clauses = []
            for k, v in spec.where.items():
                # enforce column whitelist
                if not col_ok(k):
                    continue
                if isinstance(v, (list, tuple)) and v:
                    placeholders = ",".join("?" for _ in v)
                    clauses.append(f"{k} IN ({placeholders})")
                    params.extend(list(v))
                elif v is None:
                    clauses.append(f"{k} IS NULL")
                else:
                    clauses.append(f"{k} = ?")
                    params.append(v)
            if clauses:
                sql += " WHERE " + " AND ".join(clauses)

        if spec.group_by:
            safe_gb = [c for c in spec.group_by if col_ok(c)]
            if safe_gb:
                sql += " GROUP BY " + ", ".join(safe_gb)

        if spec.order_by:
            safe_ob = []
            for c, direction in spec.order_by:
                if col_ok(c) and direction.upper() in ("ASC", "DESC"):
                    safe_ob.append(f"{c} {direction.upper()}")
            if safe_ob:
                sql += " ORDER BY " + ", ".join(safe_ob)

        if spec.limit:
            sql += " LIMIT ?"
            params.append(int(spec.limit))

        return sql, params

    def sanitize_constraints(self, constraints: Dict[str, Any]) -> SQLSpec:
        where: Dict[str, Any] = {}
        if constraints.get("language"):
            where["vulns.language"] = constraints["language"].capitalize()
        # Demo aggregation: count by severity
        return SQLSpec(
            tables=["vulns"],
            select=["vulns.severity", "COUNT(*) AS cnt"],
            where=where,
            group_by=["vulns.severity"],
            order_by=[("cnt", "DESC")],
            limit=10
        )


class SQLClient:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def execute(self, sql: str, params: List[Any]) -> List[Dict[str, Any]]:
        cur = self.conn.cursor()
        cur.execute(sql, params)
        cols = [d[0] for d in cur.description] if cur.description else []
        rows = cur.fetchall()
        return [dict(zip(cols, r)) for r in rows]


class Embedder:
    # Placeholder embedder; returns fixed-size bag-of-chars vector for determinism.
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)

    def _embed(self, text: str, dim: int = 32) -> List[float]:
        v = [0.0] * dim
        for i, ch in enumerate(text[:256]):
            v[i % dim] += (ord(ch) % 31) / 31.0
        # L2 normalize
        norm = sum(x*x for x in v) ** 0.5 or 1.0
        return [x / norm for x in v]


class VectorIndex:
    # Tiny in-memory index
    def __init__(self):
        self._vecs: Dict[str, List[float]] = {}
        self._meta: Dict[str, Dict[str, Any]] = {}

    def upsert(self, ids: List[str], vectors: List[List[float]], metas: List[Dict[str, Any]] | None = None):
        for i, vid in enumerate(ids):
            self._vecs[vid] = vectors[i]
            if metas:
                self._meta[vid] = metas[i]

    def search(self, vector: List[float], top_k: int) -> List[Tuple[str, float]]:
        def dot(a, b): return sum(x*y for x, y in zip(a, b))
        sims = [(vid, dot(vector, v)) for vid, v in self._vecs.items()]
        sims.sort(key=lambda x: x[1], reverse=True)
        return sims[:top_k]

    def meta(self, vid: str) -> Dict[str, Any]:
        return self._meta.get(vid, {})


class TextStore:
    def __init__(self):
        self._docs: Dict[str, str] = {}

    def upsert(self, docs: Dict[str, str]):
        self._docs.update(docs)

    def get(self, ids: List[str]) -> List[Passage]:
        out: List[Passage] = []
        for vid in ids:
            if vid in self._docs:
                out.append(Passage(doc_id=vid, text=self._docs[vid], meta={}, score=0.0))
        return out


class SQLRetriever:
    def __init__(self, inspector: SchemaInspector, builder: SQLQueryBuilder, client: SQLClient):
        self.inspector = inspector
        self.builder = builder
        self.client = client

    def retrieve(self, q: SubQuery) -> RetrievalResult:
        t0 = time.time()
        schema = self.inspector.describe()
        spec = self.builder.sanitize_constraints(q.constraints or {})
        sql, params = self.builder.to_sql(spec, schema)
        rows = self.client.execute(sql, params)
        dt = (time.time() - t0) * 1000
        return RetrievalResult(sql_rows=rows, passages=None, diagnostics={"latency_ms": dt, "sql": sql, "params": params})


class VectorRetriever:
    def __init__(self, embedder: Embedder, index: VectorIndex, store: TextStore):
        self.embedder = embedder
        self.index = index
        self.store = store

    def retrieve(self, q: SubQuery) -> RetrievalResult:
        t0 = time.time()
        qvec = self.embedder.embed_query(q.goal + " " + json.dumps(q.constraints or {}))
        hits = self.index.search(qvec, top_k=3)
        ids = [h[0] for h in hits]
        passages = []
        for vid, score in hits:
            meta = self.index.meta(vid)
            passages.append(Passage(doc_id=vid, text=meta.get("text", ""), meta=meta, score=float(score)))
        dt = (time.time() - t0) * 1000
        return RetrievalResult(sql_rows=None, passages=passages, diagnostics={"latency_ms": dt, "hits": hits})


class FusionRanker:
    def fuse(self, results: List[RetrievalResult]) -> RetrievalResult:
        fused = RetrievalResult(sql_rows=[], passages=[], diagnostics={"stages": []})
        for r in results:
            if r.sql_rows:
                fused.sql_rows.extend(r.sql_rows)
            if r.passages:
                fused.passages.extend(r.passages)
            if fused.diagnostics is not None and r.diagnostics:
                fused.diagnostics["stages"].append(r.diagnostics)
        return fused


class AnswerSynthesizer:
    # WIP: Replace by AnswerComposer
    def synthesize(self, q: NLQuery, fused: RetrievalResult, plan: QueryPlan) -> Answer:
        parts = []
        cits: Dict[str, List[str]] = {"sql": [], "docs": []}

        if fused.sql_rows:
            parts.append(f"Structured findings (top {min(5, len(fused.sql_rows))}):")
            for row in fused.sql_rows[:5]:
                parts.append(f"  - {row}")
            cits["sql"].append("vulns")

        if fused.passages:
            parts.append("Relevant context:")
            for p in fused.passages[:3]:
                snippet = (p.text[:140] + "...") if len(p.text) > 140 else p.text
                parts.append(f'  - [{p.doc_id}] {snippet}')
                cits["docs"].append(p.doc_id)

        if not parts:
            parts.append("No direct matches found. Consider refining the question.")

        traces = {
            "plan": [vars(s) for s in plan.steps],
            "diagnostics": fused.diagnostics,
        }
        return Answer(text="\n".join(parts), citations=cits, traces=traces)


class TopKSelector(ContentSelector):
    def __init__(self, k_rows: int = 5, k_passages: int = 2):
        self.k_rows, self.k_passages = k_rows, k_passages

    def select(self, fused: EvidenceBundle, q, plan) -> EvidenceBundle:
        return EvidenceBundle(sql=fused.sql[:self.k_rows], passages=sorted(fused.passages, key=lambda x: -x.score)[:self.k_passages])


class SimpleFactExtractor(FactExtractor):
    def to_facts(self, evidence: EvidenceBundle, q) -> List[Claim]:
        claims: List[Claim] = []
        # Heuristic: if rows have cnt/severity/package, form compact facts
        for r in evidence.sql:
            v = r.values
            if "severity" in v and "cnt" in v:
                txt = f"{v['severity'].capitalize()} issues: {v['cnt']}"
            elif "package" in v and "severity" in v:
                txt = f"{v['package']} has {v['severity']} issues"
            else:
                txt = ", ".join(f"{k}={v[k]}" for k in list(v.keys())[:3])
            claims.append(Claim(text=txt, kind="fact", grounded_in_sql=True, evidence_refs=["SQL:vulns"], confidence=0.5))
        return claims


class SimpleRationaleExtractor(RationaleExtractor):
    def to_explanations(self, evidence: EvidenceBundle, q) -> List[Claim]:
        out: List[Claim] = []
        for p in evidence.passages:
            snippet = p.text.strip()
            if len(snippet) > 180: snippet = snippet[:177] + "…"
            out.append(Claim(text=snippet, kind="explanation", grounded_in_text=True, evidence_refs=[f"DOC:{p.doc_id}"], confidence=0.4))
        return out


class DefaultMicroPlanner(MicroPlanner):
    def plan(self, q, spec: AnswerSpec, claims: List[Claim], goal: GoalType) -> AnswerSections:
        facts = [c.text for c in claims if c.kind == "fact"][:3]
        expls = [c.text for c in claims if c.kind == "explanation"][:2]
        thesis = None
        if facts:
            thesis = f"In brief: {facts[0]}."
        elif expls:
            thesis = f"In brief: {expls[0]}"
        next_step = "Consider applying recommended mitigations and pinning safe versions."
        return AnswerSections(thesis=thesis, key_facts=facts, explanation=expls, next_step=next_step)


class PlainRealizer(SurfaceRealizer):
    def realize(self, sections: AnswerSections, spec: AnswerSpec) -> str:
        parts = []
        if sections.thesis: parts.append(sections.thesis)
        if sections.key_facts: parts.append("Key facts: " + "; ".join(sections.key_facts) + ".")
        if sections.explanation: parts.append("Why/How: " + " ".join(sections.explanation))
        if sections.next_step: parts.append("Next step: " + sections.next_step)
        if spec.inline_citations and sections.citations_inline:
            parts.append(" " + " ".join(sections.citations_inline))
        return " ".join(parts)


class SimpleCitations(CitationManager):
    def build(self, claims: List[Claim], evidence: EvidenceBundle) -> Tuple[List[str], Dict[str, List[str]]]:
        sql_ids = ["vulns"] if evidence.sql else []
        doc_ids = [p.doc_id for p in evidence.passages]
        anchors = []
        if sql_ids: anchors.append("[SQL:vulns]")
        anchors += [f"[DOC:{d}]" for d in doc_ids[:3]]
        return anchors, {"sql": sql_ids, "docs": doc_ids}


class LiteralAuditor(ClaimAuditor):
    def audit(self, claims: List[Claim], evidence: EvidenceBundle) -> Tuple[List[Claim], float]:
        # Basic rule: facts backed by any SQL row get +0.4, text-backed +0.3; cap at 0.9
        out = []
        for c in claims:
            conf = 0.0
            if c.grounded_in_sql and evidence.sql: conf += 0.4
            if c.grounded_in_text and evidence.passages: conf += 0.3
            c.confidence = min(0.9, conf or 0.2)
            out.append(c)
        grounding_score = sum(c.confidence for c in out) / max(1, len(out))
        return out, grounding_score


class PassthroughRedactor(Redactor):
    def redact(self, text: str, claims: List[Claim]) -> Tuple[str, List[Claim]]:
        # Hook for PII/secret removal; no-op for now.
        return text, claims


class WordBudgetFitter(LengthController):
    def fit(self, text: str, spec: AnswerSpec, sections: AnswerSections) -> str:
        words = text.split()
        if len(words) > spec.max_words:
            return " ".join(words[:spec.max_words]) + "…"
        if len(words) < spec.min_words and sections.explanation:
            # Light pad by repeating part of explanation (better would be to add another fact/summary)
            pad = " ".join(sections.explanation)[: max(0, 10 * (spec.min_words - len(words)))]
            return (text + " " + pad).strip()
        return text


class Guardrails:
    def validate_question(self, q: NLQuery) -> None:
        if not q.text or not q.text.strip():
            raise ValueError("Empty question.")
        if len(q.text) > 2000:
            raise ValueError("Question too long.")

    def validate_sql(self, sql: str) -> None:
        banned = [";--", "DROP ", "DELETE ", "UPDATE ", "INSERT "]
        if any(b in sql.upper() for b in banned):
            raise ValueError("Potentially dangerous SQL.")
        # read-only enforced by builder (no non-SELECT)

    def redact(self, answer: Answer) -> Answer:
        # Placeholder for Personally Identifiable Information (PII) redaction
        return answer


class Cache:
    # WIP: memoize embeddings, SQL results, and full answers.
    def __init__(self):
        self._kv: Dict[str, Any] = {}

    def get(self, key: str) -> Optional[Any]:
        return self._kv.get(key)

    def set(self, key: str, value: Any, ttl_s: int = 300) -> None:
        self._kv[key] = value


class PipelineController:
    def __init__(self, components: Dict[str, Any]):
        self.guardrails: Guardrails = components["guardrails"]
        self.understanding: QueryUnderstanding = components["understanding"]
        self.decomposer: QueryDecomposer = components["decomposer"]
        self.planner: Planner = components["planner"]
        self.sql_retriever: SQLRetriever = components["sql_retriever"]
        self.vec_retriever: VectorRetriever = components["vec_retriever"]
        self.fuser: FusionRanker = components["fuser"]
        self.synth: AnswerSynthesizer = components["synth"]

    def run(self, text: str) -> Answer:
        q = NLQuery(text=text)
        self.guardrails.validate_question(q)

        intent = self.understanding.classify_intent(q)
        entities = self.understanding.extract_entities(q)
        subqs = self.decomposer.decompose(q, intent, entities)
        plan = self.planner.build_plan(subqs)

        results: List[RetrievalResult] = []
        for step in plan.steps:
            if step.modality == "sql":
                r = self.sql_retriever.retrieve(step)
            else:
                r = self.vec_retriever.retrieve(step)
            results.append(r)

        fused = self.fuser.fuse(results)
        composer = build_answer_composer()
        evidence = to_evidence_bundle(fused)
        spec = AnswerSpec(max_words=140, min_words=60, inline_citations=True, style_concise=True, goal_hint=GoalType.AGGREGATE_LOOKUP)
        goal = GoalType.AGGREGATE_LOOKUP

        # ans = self.synth.synthesize(q, fused, plan)
        ans = composer.compose(q, plan, evidence, spec, goal)
        ans = self.guardrails.redact(ans)
        return ans


# - - - - - Minimal bootstrapping (toy DB + toy corpus) - - - - -
def boot_inmemory_sqlite() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE vulns (
            id INTEGER PRIMARY KEY,
            package TEXT,
            language TEXT,
            severity TEXT,
            year INTEGER
        )
    """)
    demo_rows = [
        (1, "requests", "Python", "high", 2024),
        (2, "flask",    "Python", "critical", 2024),
        (3, "lodash",   "Javascript", "medium", 2023),
        (4, "express",  "Javascript", "high", 2024),
        (5, "numpy",    "Python", "low", 2024),
        (6, "flask",    "Python", "critical", 2025),
    ]
    cur.executemany("INSERT INTO vulns(id, package, language, severity, year) VALUES (?, ?, ?, ?, ?)", demo_rows)
    conn.commit()
    return conn

def boot_text_corpus(embedder: Embedder, index: VectorIndex, store: TextStore):
    docs = {
        "doc_flask_crit": "Flask critical vulnerabilities often relate to unsafe debug mode and serialization. Mitigate by disabling debug in prod, pinning safe versions, and validating inputs.",
        "doc_py_mitig":  "Python project mitigation: use virtual environments, pin dependencies, run `pip-audit`, enforce code reviews, and apply security headers when serving HTTP.",
        "doc_js_ctx":    "Common JS issues include prototype pollution and XSS in templating. Sanitize inputs and use trusted templating engines."
    }
    store.upsert({k: v for k, v in docs.items()})
    vecs = embedder.embed_texts(list(docs.values()))
    metas = [{"text": v} for v in docs.values()]
    index.upsert(ids=list(docs.keys()), vectors=vecs, metas=metas)


# - - - - - CLI - - - - -
def build_pipeline() -> PipelineController:
    # SQL
    conn = boot_inmemory_sqlite()
    inspector = SchemaInspector(conn)
    builder = SQLQueryBuilder()
    sql_client = SQLClient(conn)
    sql_ret = SQLRetriever(inspector, builder, sql_client)
    # Vector
    embedder = Embedder()
    vindex = VectorIndex()
    tstore = TextStore()
    boot_text_corpus(embedder, vindex, tstore)
    vec_ret = VectorRetriever(embedder, vindex, tstore)
    # Other
    components = dict(
        guardrails=Guardrails(),
        understanding=QueryUnderstanding(),
        decomposer=QueryDecomposer(),
        planner=Planner(),
        sql_retriever=sql_ret,
        vec_retriever=vec_ret,
        fuser=FusionRanker(),
        synth=AnswerSynthesizer(),
    )
    return PipelineController(components)

def build_answer_composer() -> AnswerComposer:
    return AnswerComposer(
        selector=TopKSelector(k_rows=5, k_passages=2),
        fact_extractor=SimpleFactExtractor(),
        rationale_extractor=SimpleRationaleExtractor(),
        microplanner=DefaultMicroPlanner(),
        realizer=PlainRealizer(),
        auditor=LiteralAuditor(),
        citer=SimpleCitations(),
        redactor=PassthroughRedactor(),
        length_ctrl=WordBudgetFitter(),
    )

def to_evidence_bundle(fused: RetrievalResult) -> EvidenceBundle:
    sql = []
    for r in (fused.sql_rows or []):
        sql.append(EvidenceRow(table="vulns", values=r, weight=1.0))
    passages = []
    for p in (fused.passages or []):
        passages.append(EvidencePassage(doc_id=p.doc_id, text=p.text, score=float(getattr(p, "score", 0.0)), meta=getattr(p, "meta", {})))
    return EvidenceBundle(sql=sql, passages=passages)

def main():
    parser = argparse.ArgumentParser(description="Hybrid Query System (skeleton)")
    parser.add_argument("question", nargs="?", help="Natural language question")
    parser.add_argument("--trace", action="store_true", help="Print JSON trace to stderr")
    parser.add_argument("--repl", action="store_true", help="Interactive REPL")
    args = parser.parse_args()

    pipeline = build_pipeline()

    def ask(q: str):
        ans = pipeline.run(q)
        print(f"\n{ans.text}")
        print("\nCitations:", json.dumps(ans.citations))
        if args.trace:
            print("\n--- TRACE ---", file=sys.stderr)
            # print(json.dumps(ans.traces, indent=2), file=sys.stderr)
            print(ans.claims)
            print(json.dumps(ans.meta, indent=2), file=sys.stderr)

    if args.repl:
        print("Hybrid Query REPL. Type 'exit' to quit.")
        while True:
            try:
                q = input("> ").strip()
            except EOFError:
                break
            if q.lower() in ("exit", "quit"):
                break
            if not q:
                continue
            ask(q)
    else:
        if not args.question:
            parser.error("Provide a question or use --repl")
        ask(args.question)


if __name__ == "__main__":
    main()
