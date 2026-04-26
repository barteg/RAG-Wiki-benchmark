# Wiki-Benchmark: RAG vs Karpathy Wiki

This project is a research framework designed to benchmark Traditional Retrieval-Augmented Generation (RAG) against the "Karpathy Wiki" approach (LLM-Compiled Knowledge Base) using structured LangGraph agents.

## Directory Structure

- `data/01_raw`: Source unstructured data (PDFs, TXT, HTML).
- `data/02_processed`: Cleaned data ready for ingestion.
- `data/03_rag_db`: Traditional RAG Vector Database baseline.
- `data/04_wiki`: The compiled, interlinked Markdown knowledge base.
- `src/agents/`: LLM agents (Librarian, Synthesizer, Linter).
- `src/eval/`: Multi-hop generation and Judge metrics.
- `benchmarks/`: Golden datasets and benchmarking output.

## Quickstart

1. Set your `GEMINI_API_KEY` in a `.env` file at the root of the project. 
   - Get your API key from the **[Google AI Studio](https://aistudio.google.com/app/apikey)**.
   ```bash
   echo "GEMINI_API_KEY=your_key_here" > .env
   ```

2. Activate the virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -e .
   ```

3. Run the Librarian Agent to process the sample dataset:
   ```bash
   python main.py
   ```
