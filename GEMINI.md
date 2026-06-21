# Wiki-Benchmark Project Instructions & Context

## 🎯 Current Objectives
- Benchmark **Advanced RAG** vs **Karpathy Wiki** (Compiled Knowledge Base).
- Track **Economic Crossover Point** for pre-computed vs runtime retrieval.
- Running large-scale evaluations on **Athena HPC cluster**.

## 📊 Active Session State (Last Updated: 2026-06-10)
- **Primary User:** Bartosz Janikula (plgbjanikula).
- **Remote Host:** `athena.cyfronet.pl` (Athena HPC).
- **Last Job ID:** `2662414` (Submitted 2026-06-09).
- **Execution Script:** `athena_run.sh` -> `benchmark.slurm`.
- **Expected Results:** `benchmarks/comparative_results.json` (on remote) or `results_athena.json` (local summary).

## 🛠️ Operational Commands
- **Check Remote Status:** `ssh plgbjanikula@athena.cyfronet.pl 'squeue -u plgbjanikula'`
- **Tail Remote Logs:** `ssh plgbjanikula@athena.cyfronet.pl 'tail -f ~/wiki_benchmark/logs/benchmark_*.out'`
- **Sync Results:** Use `scp` to pull `comparative_results.json` from `~/wiki_benchmark/benchmarks/`.

## 🏗️ Technical Architecture
- **Ingestion:** Recursive splitting -> LLM Librarian -> Hybrid Index (Chroma + SQLite FTS5).
- **Retrieval:** RAG (HyDE + Hybrid) vs Wiki (Guided Graph Walk).
- **Model:** `CYFRAGOVPL/Llama-PLLuM-70B-chat-250801` (Locked for consistency).

## ⚖️ Rules & Conventions
- **Naming:** Always use "Bartosz Janikula" (no 'ł') in reports.
- **Verification:** Always verify job completion on Athena before attempting to process results.
- **Environment:** Remote execution uses `venv_athena`. Local development uses `venv`.
