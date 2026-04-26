import asyncio
import json
import os
import sqlite3
import time
from pathlib import Path
from dotenv import load_dotenv
from src.utils.rag_baseline import AdvancedRAG
from src.eval.judge import EvalJudge

# CONCURRENCY CONTROL
MAX_CONCURRENT_TASKS = int(os.getenv("MAX_CONCURRENT_TASKS", "20"))
semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)

def init_checkpoint_db(project_root: Path):
    db_path = project_root / "benchmarks" / "checkpoints.sqlite"
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS results (
            item_id TEXT PRIMARY KEY,
            result_json TEXT,
            wall_clock_time FLOAT,
            tokens_total INTEGER
        )
    """)
    return conn

async def benchmark_item(item, rag, judge, wiki_dir, checkpoint_conn):
    item_id = item.get("id", str(hash(item["question"])))
    cursor = checkpoint_conn.execute("SELECT result_json FROM results WHERE item_id = ?", (item_id,))
    if cursor.fetchone(): return None # Already done

    async with semaphore:
        q = item["question"]
        gold = item["answer"]
        start = time.perf_counter()
        try:
            res = await judge.evaluate(q, gold, rag, wiki_dir)
            duration = time.perf_counter() - start
            res_dict = res.model_dump()
            
            checkpoint_conn.execute(
                "INSERT INTO results (item_id, result_json, wall_clock_time, tokens_total) VALUES (?, ?, ?, ?)", 
                (item_id, json.dumps(res_dict), duration, res_dict['wiki_metrics']['tokens'] + res_dict['rag_metrics']['tokens'])
            )
            checkpoint_conn.commit()
            return res_dict
        except Exception as e:
            print(f"Error {item_id}: {e}")
            return None

async def run_benchmark():
    load_dotenv()
    project_root = Path(__file__).parent
    wiki_dir = project_root / "data" / "04_wiki"
    rag_db_dir = project_root / "data" / "03_rag_db"

    checkpoint_conn = init_checkpoint_db(project_root)
    rag = AdvancedRAG(db_path=rag_db_dir)
    judge = EvalJudge()

    with open(project_root / "benchmarks" / "hotpot_eval.json", "r") as f:
        hotpot_set = json.load(f)

    # Economic Crossover Variables
    with open(wiki_dir / "index.json", "r") as f:
        idx = json.load(f)
    
    total_wiki_ingest_cost = sum(p.get("total_tokens", 0) for p in idx["pages"].values())
    
    print(f"Starting Symmetrical Benchmark (Model Locked: {os.getenv('NVIDIA_MODEL')})")
    print(f"Wiki Build Cost: {total_wiki_ingest_cost} tokens")
    
    start_total = time.perf_counter()
    tasks = [benchmark_item(item, rag, judge, wiki_dir, checkpoint_conn) for item in hotpot_set]
    await asyncio.gather(*tasks)
    
    # Final Summary Generation for AI Verifier
    cursor = checkpoint_conn.execute("SELECT result_json, wall_clock_time FROM results")
    rows = cursor.fetchall()
    
    all_results = []
    total_lat_rag = 0
    total_lat_wiki = 0
    for r_json, wall_time in rows:
        d = json.loads(r_json)
        all_results.append(d)
        total_lat_rag += d['rag_metrics']['latency']
        total_lat_wiki += d['wiki_metrics']['latency']

    summary = {
        "benchmark_metadata": {
            "model": os.getenv("NVIDIA_MODEL"),
            "retention_period_days": 30, # Knowledge Half-life
            "wiki_ingest_tokens": total_wiki_ingest_cost,
            "rag_ingest_tokens": 0, # Baseline
        },
        "results": all_results,
        "averages": {
            "avg_rag_latency": total_lat_rag / len(all_results),
            "avg_wiki_latency": total_lat_wiki / len(all_results)
        }
    }
    
    with open(project_root / "benchmarks" / "results.json", "w") as f:
        json.dump(summary, f, indent=4)
    print(f"Done. Results saved to benchmarks/results.json")

if __name__ == "__main__":
    asyncio.run(run_benchmark())
