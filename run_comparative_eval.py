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
MAX_CONCURRENT_TASKS = int(os.getenv("MAX_CONCURRENT_TASKS", "5"))
semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)

async def benchmark_item(item, rag, judge, wiki_dir):
    async with semaphore:
        q = item["question"]
        gold = item["answer"]
        
        try:
            # 1. RAG
            start = time.perf_counter()
            rag_res = await rag.query(q)
            rag_lat = time.perf_counter() - start
            
            # 2. Wiki Fast (FTS Only)
            from src.agents.synthesizer import SynthesizerAgent
            synth = SynthesizerAgent()
            start = time.perf_counter()
            wiki_fast_res = await synth.answer_from_wiki(q, wiki_dir, fast_mode=True)
            wiki_fast_lat = time.perf_counter() - start
            
            # 3. Wiki Smart (Guided Walk)
            start = time.perf_counter()
            wiki_smart_res = await synth.answer_from_wiki(q, wiki_dir, fast_mode=False)
            wiki_smart_lat = time.perf_counter() - start
            
            return {
                "question": q,
                "gold": gold,
                "level": item.get("level", "unknown"),
                "rag": {"answer": rag_res["answer"], "latency": rag_lat, "tokens": rag_res["tokens"]},
                "wiki_fast": {"answer": wiki_fast_res["answer"], "latency": wiki_fast_lat, "tokens": wiki_fast_res["tokens"]},
                "wiki_smart": {"answer": wiki_smart_res["answer"], "latency": wiki_smart_lat, "tokens": wiki_smart_res["tokens"]}
            }
        except Exception as e:
            print(f"  [ERROR] Failed to benchmark item '{q[:50]}...': {e}")
            return {
                "question": q,
                "gold": gold,
                "error": str(e)
            }

async def run_benchmark():
    load_dotenv()
    project_root = Path(__file__).parent
    wiki_dir = project_root / "data" / "04_wiki"
    rag_db_dir = project_root / "data" / "03_rag_db"

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

    LIMIT = 50
    data_file = project_root / "hotpot_train_v1.1.json"
    if not data_file.exists():
        data_file = project_root / "benchmarks" / "hotpot_eval.json"
        
    hotpot_set = get_stratified_data(data_file, LIMIT)
        
    print(f"Starting Comparative Benchmark (n=50)")
    tasks = [benchmark_item(item, rag, judge, wiki_dir) for item in hotpot_set]
    results = await asyncio.gather(*tasks)
    
    with open(project_root / "benchmarks" / "comparative_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("Done. Results saved to benchmarks/comparative_results.json")

if __name__ == "__main__":
    asyncio.run(run_benchmark())
