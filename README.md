# Wiki-Benchmark: RAG vs. Karpathy Wiki

A high-performance research framework designed to benchmark **Advanced Retrieval-Augmented Generation (RAG)** against the **"Karpathy Wiki"** paradigm (LLM-Compiled Knowledge Base). 

This framework provides a scientifically rigorous, symmetrically-armed comparison to identify the **Economic Crossover Point** where pre-computed knowledge synthesis outperforms dynamic runtime retrieval.

---

## 🚀 Key Features

- **HPC-Scale Architecture**: 100% Non-blocking I/O stack using `AsyncOpenAI`, `httpx`, and `aiofiles`. Multiplexes thousands of concurrent queries without event-loop starvation.
- **Scientific Symmetry**: Both RAG and Wiki paths utilize the exact same **Hybrid Retrieval Engine** (SQLite FTS5 + ChromaDB Vector), ensuring results are driven by **Data Structure**, not search capabilities.
- **Guided Graph Traversal**: The Wiki engine performs local semantic scoring of contextual anchors (link + surrounding sentence) using `SentenceTransformers` for sub-millisecond, zero-cost routing.
- **Deterministic Metrics**: Eliminates "vibe-check" judging. Uses strict JSON extraction for final answers and tiered NLP metrics (**F1 Score**, **Exact Match**).
- **Model Lockdown**: Enforces use of the exact same model version for synthesis, librarianship, and answering to eliminate cross-intelligence bias.
- **Idempotent Execution**: SQLite-backed stateful task queue allows benchmark runs to resume instantly after transient failures or HPC preemption.

---

## 🏗️ Architecture

### 1. Ingestion (The "Compiler" Phase)
- **Recursive Token Splitting**: Chunks data into 400-token units with 100-token overlaps, snapping to semantic boundaries (paragraphs/sentences).
- **LLM Librarian**: Synthesizes raw chunks into an interlinked Markdown graph with `[[Internal Links]]`.
- **Hybrid Indexing**: Synchronizes updates into both ChromaDB (Vector) and SQLite FTS5 (Keyword/BM25).

### 2. Retrieval (The "Read" Phase)
- **Advanced RAG**: Query expansion (HyDE-style) + Hybrid Search + Full-Context Re-ranking.
- **Karpathy Wiki**: Hybrid Anchor Search + Relevance-Guided Walk + Global FTS5 Fallback for "orphan" pages.

### 3. Evaluation (The "Audit" Phase)
- **TCO Tracking**: Measures total token cost and wall-clock time for both Ingestion and Querying.
- **Negative Set Testing**: Evaluates "Hallucination Precision" using unanswerable questions.

---

## 🛠️ Getting Started

### 1. Configuration
Create a `.env` file in the root directory:
```bash
# Core API Config
NVIDIA_API_KEY=your_api_key
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
NVIDIA_MODEL=meta/llama-3.1-70b-instruct # Locked for all agents

# Benchmarking Config
MAX_CONCURRENT_TASKS=50
CHROMA_HOST=localhost
CHROMA_PORT=8000
```

### 2. Installation
```bash
python -m venv venv
source venv/bin/activate
pip install -e .
```

### 3. Execution
**Step 1: Build the Knowledge Bases**
```bash
python main.py
```
**Step 2: Run the Benchmark**
```bash
python run_eval.py
```

---

## 📊 Economic Metric: The Crossover Point
The framework outputs a detailed `results.json` to help you calculate the query volume ($N$) where the Wiki's $100\times$ ingestion cost is amortized:

$$N_{queries} = \frac{Cost_{WikiIngest} - Cost_{RAGIngest}}{Cost_{RAGQuery} - Cost_{WikiQuery}}$$

---

## 📂 Directory Structure
- `data/01_raw`: Source unstructured data.
- `data/03_rag_db`: Symmetrical RAG index (Chroma + SQLite).
- `data/04_wiki`: Compiled Markdown Knowledge Base.
- `src/agents/`: LLM logic (Librarian, Synthesizer).
- `src/utils/`: High-performance engines (AdvancedRAG, Metrics).
- `benchmarks/`: Evaluation datasets and stateful checkpoints.

---

## ⚖️ License
MIT
