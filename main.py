import os
import json
import time
import csv
from pathlib import Path
from dotenv import load_dotenv

from src.agents.librarian import LibrarianAgent
from src.agents.synthesizer import SynthesizerAgent
from src.agents.linter import LinterAgent
from src.utils.rag_baseline import RAGBaseline

def main():
    load_dotenv()
    project_root = Path(__file__).parent
    raw_dir = project_root / "data" / "01_raw"
    wiki_dir = project_root / "data" / "04_wiki"
    rag_db_dir = project_root / "data" / "03_rag_db"
    
    librarian = LibrarianAgent()
    synthesizer = SynthesizerAgent()
    rag = RAGBaseline(db_path=rag_db_dir)
    
    csv_path = project_root / "benchmarks" / "ingest_metrics.csv"
    if not csv_path.exists():
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "source", "action", "active_latency", "tokens_total"])

    index_path = wiki_dir / "index.json"
    processed_sources = set()
    if index_path.exists():
        with open(index_path, "r") as f:
            idx = json.load(f)
            for page in idx.get("pages", {}).values():
                processed_sources.update(page.get("sources", []))

    print(f"Starting Batch Ingestion (Clean Latency Mode)...")
    
    for raw_file in sorted(raw_dir.glob("*.txt")):
        if raw_file.name in processed_sources:
            continue

        print(f"\n[Ingest] {raw_file.name}")
        with open(raw_file, "r", encoding="utf-8") as f:
            content = f.read()
            
        try:
            rag.ingest(content, raw_file.name)

            # 1. Route (Returns its own active_time)
            lib_res = librarian.route_text(content, raw_file.name, wiki_dir)
            
            # 2. Sync (Returns its own active_time)
            sync_res = synthesizer.synthesize(lib_res.target_page, content, raw_file.name, wiki_dir)
            
            # Combined Active Latency (excl. rate limit waits)
            active_latency = lib_res.active_time + sync_res["active_time"]
            total_tokens = lib_res.tokens["total"] + sync_res["tokens"]["total"]
            
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([time.time(), raw_file.name, lib_res.action, active_latency, total_tokens])
            
            _update_index_with_stats(lib_res.target_page, raw_file.name, active_latency, total_tokens, wiki_dir)
            
            print(f"  Done: {active_latency:.2f}s active, {total_tokens} tokens")
            
        except Exception as e:
            print(f"  Error {raw_file.name}: {e}")

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
    main()
