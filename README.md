# **Hybrid RAG Demo (SQL + Vector + Local LLM) for OMA**

A minimal Retrieval-Augmented Generation pipeline built without LangChain or frameworks.
It demonstrates core RAG concepts end-to-end:

* **Query understanding** → classify intent (SQL, text, or hybrid)
* **Structured retrieval** → SQLite table acting as metadata store
* **Unstructured retrieval** → tiny vector store embedding project documents
* **Hybrid fusion** → merge SQL and vector results
* **LLM synthesis** → grounded answer using a local Ollama LLM

The goal is to show clear understanding of **how RAG works internally** by implementing each step directly.

---

## **Architecture**

```
User Query
   ↓
Query Interpreter → Intent: sql / text / hybrid
   ↓
Retrievers: SQL + Vector (SentenceTransformers)
   ↓
Fusion Layer (merge rows + passages)
   ↓
LLM Synthesizer (Ollama local model)
   ↓
Final Answer + Citations + Traces
```

---

## **Key Ideas**

### **1. Structured Retrieval (SQL)**

* Uses SQLite as a mock knowledge base.
* Supports filtering based on query hints (e.g., location).
* Returns factual rows to ground the answer.

### **2. Vector Search (Text Retrieval)**

* Loads a corpus into an in-memory vector store.
* Embeds documents with SentenceTransformers.
* Performs dot-product ranking for semantic retrieval.

### **3. Hybrid Search**

If the query asks both for facts and explanations, the pipeline:

* Retrieves from SQL and
* Retrieves semantically similar text passages
  → then fuses them for the LLM.

### **4. LLM Answer Synthesis**

* Calls a local Ollama model (e.g., `gemma3`).
* Produces a concise grounded answer.
* No hallucinations: LLM is instructed to only use retrieved context.
* Citations and diagnostics are returned for transparency.

---

## **How to Run**

```bash
pip install -r requirements.txt
ollama pull gemma3:4b
ollama serve
python main.py
```

Then:

```
> who is OMA?
```

---

## **Search Scenarios**

### **Hybrid Search (SQL + Vector)**

Use when the question mixes facts + explanations:

> “Which projects were built in China, and what is special about them?”

* SQL → finds projects in China
* Vector → retrieves descriptions mentioning design features
* LLM → synthesizes answer from both

---

### **SQL-Only Retrieval**

Questions requesting counts, lists, filters, or metadata:

> “List completed projects in Paris.”

Intent → `sql`
Only structured rows are used.

---

### **Text-Only Retrieval**

Questions asking for explanations or reasoning:

> “Explain the design approach behind the Mushroom Pavilion.”

Intent → `text`
Only vector passages and document excerpts shape the answer.