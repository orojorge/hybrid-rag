from typing import Optional, List, Dict, Any
import argparse
import json
import sys
import textwrap
import requests

from models import NLQuery, Intent, RetrievalResult, Answer, Passage
from sql_backend import boot_inmemory_sqlite, SQLRetriever
from vector_store import VectorStore
from answer_synthesizer import AnswerSynthesizer



class QueryInterpreter:
    """
    Very simple query understanding:
    - classify intent: sql / text / hybrid
    - optionally detect a location keyword for filtering structured rows.
    """

    SUPPORTED_LOCATIONS = [
        "paris", "france", "mexico", "new york city", "china", "xiamen",
        "usa", "germany", "italy", "spain", "puerto escondido" ]

    def interpret(self, q: NLQuery) -> tuple[Intent, Optional[str]]:
        text = q.text.lower()

        has_why = any(k in text for k in ["why", "how", "explain"])
        has_agg = any(k in text for k in ["count", "list", "how many", "top", "most", "least"])

        if has_agg and has_why:
            kind = "hybrid"
        elif has_agg:
            kind = "sql"
        elif has_why:
            kind = "text"
        else:
            kind = "hybrid"

        loc = next((l for l in self.SUPPORTED_LOCATIONS if l in text), None)
        return Intent(kind=kind, confidence=0.6), loc



def fuse_results(sql_res: Optional[RetrievalResult], vec_res: Optional[RetrievalResult]) -> RetrievalResult:
    """
    Simple hybrid fusion:
    - concatenate rows and passages
    - gather diagnostics from both stages
    """
    sql_rows: List[Dict[str, Any]] = []
    passages: List[Passage] = []
    diagnostics: Dict[str, Any] = {"stages": []}

    if sql_res:
        sql_rows.extend(sql_res.sql_rows or [])
        if sql_res.diagnostics:
            diagnostics["stages"].append({"sql": sql_res.diagnostics})

    if vec_res:
        passages.extend(vec_res.passages or [])
        if vec_res.diagnostics:
            diagnostics["stages"].append({"vector": vec_res.diagnostics})

    return RetrievalResult(sql_rows=sql_rows, passages=passages, diagnostics=diagnostics)



class PipelineController:
    """
    End-to-end RAG pipeline:
    1) interpret query
    2) hybrid retrieval (SQL + vector)
    3) fuse
    4) synthesize answer with citations
    """

    def __init__(self, sql_retriever: SQLRetriever, vec_store: VectorStore, interpreter: QueryInterpreter, synthesizer: AnswerSynthesizer):
        self.sql_retriever = sql_retriever
        self.vec_store = vec_store
        self.interpreter = interpreter
        self.synthesizer = synthesizer

    def run(self, text: str) -> Answer:
        text = (text or "").strip()
        if not text:
            return Answer(
                text="Please provide a non-empty question.",
                citations={},
                traces={"error": "empty_question"} )

        q = NLQuery(text=text)
        intent, loc = self.interpreter.interpret(q)

        sql_res: Optional[RetrievalResult] = None
        vec_res: Optional[RetrievalResult] = None

        if intent.kind in ("sql", "hybrid"):
            sql_res = self.sql_retriever.retrieve(location=loc, limit=10)

        if intent.kind in ("text", "hybrid"):
            vec_res = self.vec_store.search(query=q.text, top_k=3)

        fused = fuse_results(sql_res, vec_res)
        return self.synthesizer.synthesize(q, intent, fused)



# - - - - - - - - - - Boot & CLI - - - - - - - - - - #

def build_pipeline() -> PipelineController:
    conn = boot_inmemory_sqlite()
    sql_ret = SQLRetriever(conn)

    vec_store = VectorStore()
    vec_store.load_corpus("corpus")

    interpreter = QueryInterpreter()
    synthesizer = AnswerSynthesizer(model_name="gemma3:4b")

    return PipelineController(sql_ret, vec_store, interpreter, synthesizer)


def main():
    parser = argparse.ArgumentParser(description="Hybrid RAG demo (SQL + vector, no frameworks)")
    parser.add_argument("--trace", action="store_true", help="Print internal traces as JSON to stderr")
    args = parser.parse_args()

    pipeline = build_pipeline()

    print("\nWhat can I find for you today?")
    while True:
        try:
            q = input("> ").strip()
        except EOFError:
            break

        if q.lower() in ("exit", "quit"):
            break
        if not q:
            continue

        ans = pipeline.run(q)
        print("\n" + ans.text)
        print("\nCitations:", json.dumps(ans.citations))

        if args.trace:
            print("\n\n- - - - - TRACE - - - - -", file=sys.stderr)
            print(json.dumps(ans.traces, indent=2), file=sys.stderr)


if __name__ == "__main__":
    main()
