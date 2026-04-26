import os
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
    
    # Model Lockdown: All agents use the same NVIDIA_MODEL
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
    processed_sources = set()
    if index_path.exists():
        with open(index_path, "r") as f:
            idx = json.load(f)
            for page in idx.get("pages", {}).values():
                processed_sources.update(page.get("sources", []))

    print(f"Starting Parallel Ingestion (Model Locked: {os.getenv('NVIDIA_MODEL')})...")
    
    semaphore = asyncio.Semaphore(10) # Safe limit for ingestion

    async def ingest_file(raw_file):
        if raw_file.name in processed_sources:
            return
        
        async with semaphore:
            print(f"[Ingest] {raw_file.name}")
            import aiofiles
            async with aiofiles.open(raw_file, mode='r', encoding="utf-8") as f:
                content = await f.read()
                
            try:
                # 1. RAG Ingest
                await asyncio.to_thread(rag.ingest, content, raw_file.name)

                # 2. Librarian Routing
                lib_res = await librarian.route_text(content, raw_file.name, wiki_dir)
                
                # 3. Wiki Synthesis
                sync_res = await synthesizer.synthesize(lib_res.target_page, content, raw_file.name, wiki_dir)
                
                active_latency = lib_res.active_time + sync_res["active_time"]
                total_tokens = lib_res.tokens["total"] + sync_res["tokens"]
                
                with open(csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([time.time(), raw_file.name, lib_res.action, active_latency, total_tokens])
                
                _update_index_with_stats(lib_res.target_page, raw_file.name, active_latency, total_tokens, wiki_dir)
                
            except Exception as e:
                print(f"  Error {raw_file.name}: {e}")

    files = sorted(raw_dir.glob("*.txt"))
    await asyncio.gather(*[ingest_file(f) for f in files])

def _update_index_with_stats(page, source, latency, tokens, wiki_dir):
    index_path = wiki_dir / "index.json"
    data = {"pages": {}}
    if index_path.exists():
        with open(index_path, "r") as f:
            data = json.load(f)
    
    if page not in data["pages"]:
        data["pages"][page] = {"sources": [], "total_ingest_time": 0, "total_tokens": 0}
    
    if source not in data["pages"][page]["sources"]:
        data["pages"][page]["sources"].append(source)
    
    data["pages"][page]["total_ingest_time"] += latency
    data["pages"][page]["total_tokens"] += tokens
    
    with open(index_path, "w") as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    asyncio.run(main())
