import os
# AGGRESSIVE ONNX/OPENMP FIX
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAX_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["ORT_TBB_THREADS"] = "1"
os.environ["ORT_INTRA_OP_NUM_THREADS"] = "1"
os.environ["ORT_INTER_OP_NUM_THREADS"] = "1"
os.environ["OMP_PROC_BIND"] = "FALSE"

import json
import time
import csv
import asyncio
from pathlib import Path
from dotenv import load_dotenv

from src.agents.librarian import LibrarianAgent
from src.agents.synthesizer import SynthesizerAgent
from src.utils.rag_baseline import AdvancedRAG

async def main():
    load_dotenv()
    project_root = Path(__file__).parent
    raw_dir = project_root / "data" / "01_raw"
    wiki_dir = project_root / "data" / "04_wiki"
    rag_db_dir = project_root / "data" / "03_rag_db"
    
    # Model Lockdown
    librarian = LibrarianAgent()
    synthesizer = SynthesizerAgent()
    rag = AdvancedRAG(db_path=rag_db_dir)
    
    csv_path = project_root / "benchmarks" / "ingest_metrics.csv"
    if not csv_path.exists():
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "source", "action", "active_latency", "tokens_total"])

    # Load index safely
    index_path = wiki_dir / "index.json"
    index_data = {"pages": {}}
    processed_sources = set()
    if index_path.exists():
        with open(index_path, "r") as f:
            index_data = json.load(f)
            for page in index_data.get("pages", {}).values():
                processed_sources.update(page.get("sources", []))

    print(f"Starting Parallel Ingestion (Model Locked: {os.getenv('LLM_MODEL') or os.getenv('NVIDIA_MODEL')})...")

    from concurrent.futures import ThreadPoolExecutor
    rag_executor = ThreadPoolExecutor(max_workers=1)
    db_lock = asyncio.Lock()
    # Concurrency control: 10 concurrent files to avoid rate limits and too much I/O contention
    ingest_semaphore = asyncio.Semaphore(10)

    async def ingest_file(raw_file):
        if raw_file.name in processed_sources:
            return

        async with ingest_semaphore:
            print(f"[Ingest] {raw_file.name} - START")
            # Use blocking open for stability on HPC filesystem
            with open(raw_file, mode='r', encoding="utf-8") as f:
                content = f.read()

            try:
                # 1. RAG Ingest
                print(f"  [RAG] {raw_file.name} - Ingesting...")
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(rag_executor, rag.ingest, content, raw_file.name)
                print(f"  [RAG] {raw_file.name} - Done.")
                
                # 2. Librarian Routing
                print(f"  [Librarian] {raw_file.name} - Routing...")
                lib_res = await librarian.route_text(content, raw_file.name, wiki_dir)
                print(f"  [Librarian] {raw_file.name} - Decision: {lib_res.action} -> {lib_res.target_page}")
                
                # 3. Wiki Synthesis
                print(f"  [Synthesizer] {raw_file.name} - Synthesizing into {lib_res.target_page}...")
                # We lock the synthesizer/index update because it modifies shared Markdown files/sqlite
                async with db_lock:
                    sync_res = await synthesizer.synthesize(lib_res.target_page, content, raw_file.name, wiki_dir)
                
                active_latency = lib_res.active_time + sync_res["active_time"]
                total_tokens = lib_res.tokens["total"] + sync_res["tokens"]
                
                async with db_lock:
                    # Update metrics
                    with open(csv_path, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([time.time(), raw_file.name, lib_res.action, active_latency, total_tokens])
                    
                    # Update In-Memory Index
                    _update_index_in_memory(index_data, lib_res.target_page, raw_file.name, active_latency, total_tokens)
                    
                    # Atomic Save Index every 5 files to prevent data loss but reduce I/O
                    if len(index_data["pages"]) % 5 == 0:
                        with open(index_path, "w") as f:
                            json.dump(index_data, f, indent=4)
                
                print(f"[Ingest] {raw_file.name} - FINISHED")
                
            except Exception as e:
                print(f"  [ERROR] {raw_file.name}: {e}")

    files = sorted(raw_dir.glob("*.txt"))
    
    # Support for smaller test runs via environment variable
    LIMIT = int(os.getenv("BENCHMARK_LIMIT", "0"))
    if LIMIT > 0:
        files = files[:LIMIT]

    # Process with asyncio.gather for parallelism
    tasks = [ingest_file(f) for f in files]
    await asyncio.gather(*tasks)
    
    # Final Index Save
    with open(index_path, "w") as f:
        json.dump(index_data, f, indent=4)

def _update_index_in_memory(data, page, source, latency, tokens):
    if page not in data["pages"]:
        data["pages"][page] = {"sources": [], "total_ingest_time": 0, "total_tokens": 0}
    
    if source not in data["pages"][page]["sources"]:
        data["pages"][page]["sources"].append(source)
    
    data["pages"][page]["total_ingest_time"] += latency
    data["pages"][page]["total_tokens"] += tokens

if __name__ == "__main__":
    asyncio.run(main())
