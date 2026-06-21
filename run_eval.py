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
            res_dict["level"] = item.get("level", "unknown")
            
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

    def get_stratified_data(path, limit):
        data = json.load(open(path, "r"))
        if limit <= 0: return data
        target = limit // 3
        res, counts = [], {"easy": 0, "medium": 0, "hard": 0}
        for item in data:
            lvl = item.get("level", "medium")
            if counts.get(lvl, 0) < target:
                res.append(item)
                counts[lvl] = counts.get(lvl, 0) + 1
        rem = limit - len(res)
        if rem > 0:
            for item in data:
                if item not in res:
                    res.append(item)
                    rem -= 1
                    if rem == 0: break
        return res

    LIMIT = int(os.getenv("BENCHMARK_LIMIT", "120"))
    data_file = project_root / "hotpot_train_v1.1.json"
    if not data_file.exists():
        data_file = project_root / "benchmarks" / "hotpot_eval.json"
        
    hotpot_set = get_stratified_data(data_file, LIMIT)

    # Economic Crossover Variables
    with open(wiki_dir / "index.json", "r") as f:
        idx = json.load(f)
    
    total_wiki_ingest_cost = sum(p.get("total_tokens", 0) for p in idx["pages"].values())
    
    print(f"Starting Symmetrical Benchmark (Model Locked: {os.getenv('LLM_MODEL') or os.getenv('NVIDIA_MODEL')})")
    print(f"Wiki Build Cost: {total_wiki_ingest_cost} tokens")
    
    start_total = time.perf_counter()
    tasks = [benchmark_item(item, rag, judge, wiki_dir, checkpoint_conn) for item in hotpot_set]
    await asyncio.gather(*tasks)
    
    # Final Summary Generation for AI Verifier
    cursor = checkpoint_conn.execute("SELECT result_json, wall_clock_time FROM results")
    rows = cursor.fetchall()
    
    all_results = []
    stats = {
        "rag": {"latency": 0, "tokens": 0, "f1": 0, "em": 0},
        "wiki": {"latency": 0, "tokens": 0, "f1": 0, "em": 0},
        "zero_shot": {"latency": 0, "tokens": 0, "f1": 0, "em": 0}
    }
    
    count = len(rows)
    for r_json, wall_time in rows:
        d = json.loads(r_json)
        all_results.append(d)
        for key in ["rag", "wiki", "zero_shot"]:
            m = d[f"{key}_metrics"]
            stats[key]["latency"] += m["latency"]
            stats[key]["tokens"] += m["tokens"]
            stats[key]["f1"] += m["f1"]
            stats[key]["em"] += 1 if m["em"] else 0

    # Economic Crossover Calculation (N)
    # N = (Wiki_Ingest_Cost - RAG_Ingest_Cost) / (RAG_Query_Cost - Wiki_Query_Cost)
    avg_wiki_q_cost = stats["wiki"]["tokens"] / count
    avg_rag_q_cost = stats["rag"]["tokens"] / count
    
    cost_diff_q = avg_rag_q_cost - avg_wiki_q_cost
    if cost_diff_q > 0:
        n_crossover = total_wiki_ingest_cost / cost_diff_q
    else:
        n_crossover = float('inf') # Wiki is more expensive per query or equal

    summary = {
        "benchmark_metadata": {
            "model": os.getenv("LLM_MODEL") or os.getenv("NVIDIA_MODEL"),
            "wiki_ingest_tokens": total_wiki_ingest_cost,
            "economic_crossover_queries": n_crossover,
            "data_leakage_risk": stats["zero_shot"]["em"] / count > 0.5
        },
        "averages": {
            "rag": {k: v / count for k, v in stats["rag"].items()},
            "wiki": {k: v / count for k, v in stats["wiki"].items()},
            "zero_shot": {k: v / count for k, v in stats["zero_shot"].items()}
        },
        "raw_results": all_results
    }
    
    with open(project_root / "benchmarks" / "results.json", "w") as f:
        json.dump(summary, f, indent=4)
    print(f"Done. Crossover N: {n_crossover:.2f} queries. Leakage Risk: {summary['benchmark_metadata']['data_leakage_risk']}")

if __name__ == "__main__":
    asyncio.run(run_benchmark())
