from typing import List, Dict, Tuple
import textwrap
import requests

from models import NLQuery, Intent, RetrievalResult, Answer, Passage


class AnswerSynthesizer:
    """
    Turn fused retrieval results into a grounded answer by calling
    a local LLM served by Ollama, while keeping explicit citations
    and traces. No deterministic fallback.
    """

    def __init__(self, model_name: str = "gemma3:4b", api_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.api_url = api_url.rstrip("/")

    def _build_context(self, fused: RetrievalResult) -> Tuple[str, Dict[str, List[str]]]:
        """
        Prepare a compact text context for the LLM and compute citations.
        Returns (context_text, citations_dict).
        """
        parts: List[str] = []

        # Structured summary (SQL)
        if fused.sql_rows:
            parts.append("Structured results from the work table:")
            for row in fused.sql_rows[:3]:
                name = row.get("name", "Unknown project")
                client = row.get("client") or "Unknown client"
                loc = row.get("location") or "Unknown location"
                year = row.get("year") or "Unknown year"
                parts.append(f"- {name} for {client} in {loc} ({year})")

        # Context snippets (vector passages)
        if fused.passages:
            if parts:
                parts.append("")
            parts.append("Context from retrieved documents:")
            for p in fused.passages[:3]:
                text = p.text.strip()
                if len(text) > 400:
                    text = text[:397] + "…"
                parts.append(f"[DOC {p.doc_id}] {text}")

        if not parts:
            parts.append("No matching projects were found in the structured data or documents.")

        context_text = "\n".join(parts)

        citations = {
            "sql": ["work"] if fused.sql_rows else [],
            "docs": [p.doc_id for p in fused.passages] if fused.passages else [],
        }

        return context_text, citations

    def _call_ollama(self, prompt: str) -> str:
        """
        Call the local Ollama HTTP API with a simple /generate request.
        """
        url = f"{self.api_url}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
        }
        resp = requests.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        return (data.get("response") or "").strip()

    def synthesize(self, q: NLQuery, intent: Intent, fused: RetrievalResult) -> Answer:
        """
        Produce the final grounded answer using the local LLM.
        """
        context_text, citations = self._build_context(fused)

        system_instructions = """
        You are a helpful assistant that answers questions about architecture projects.
        You must base your answer ONLY on the structured results and document excerpts
        provided below. If the context is insufficient to answer confidently, say so.

        Requirements:
        - Be concise (3-6 sentences).
        - Mention specific projects, locations, years, clients when relevant.
        - Do NOT invent projects or details not present in the context.
        - Do not output citations; the caller will handle that.
        """

        prompt = textwrap.dedent(f"""
            {system_instructions}

            User question:
            {q.text}

            Retrieved context:
            {context_text}

            Now write a grounded answer to the user question.
            """).strip()

        traces: Dict[str, any] = {
            "intent": intent.kind,
            "diagnostics": fused.diagnostics,
            "question": q.text,
            "llm_model": self.model_name,
        }

        try:
            llm_answer = self._call_ollama(prompt)
            answer_text = llm_answer if llm_answer else "LLM returned an empty response."
        except Exception as e:
            answer_text = "Error contacting the local LLM. Please check your Ollama setup."
            traces["llm_error"] = str(e)

        return Answer(
            text=answer_text,
            citations=citations,
            traces=traces
        )
