import os
import json
from pathlib import Path
from dotenv import load_dotenv
from src.utils.rag_baseline import RAGBaseline
from src.eval.judge import EvalJudge


def run_benchmark():
    load_dotenv()
    api_key = os.environ.get("AIzaSyAROBvBcUF2Ic7bSCUEw1LM1IlktwnCZPk")
    project_root = Path(__file__).parent
    wiki_dir = project_root / "data" / "04_wiki"
    rag_db_dir = project_root / "data" / "03_rag_db"

    rag = RAGBaseline(db_path=rag_db_dir)
    judge = EvalJudge()

    # Load HotpotQA eval set
    eval_path = project_root / "benchmarks" / "hotpot_eval.json"
    with open(eval_path, "r") as f:
        hotpot_set = json.load(f)

    # Load index to get ingest times
    with open(wiki_dir / "index.json", "r") as f:
        index_data = json.load(f)

    total_ingest_time = sum(p.get("total_ingest_time", 0) for p in index_data["pages"].values())
    avg_ingest_time = total_ingest_time / len(index_data["pages"]) if index_data["pages"] else 0

    results = []
    print(f"\nStarting Benchmark Evaluation on {len(hotpot_set)} HotpotQA questions...")
    print(f"--- PRE-COMPUTATION STATS ---")
    print(f"Total Wiki Ingest Time: {total_ingest_time:.2f}s")
    print(f"Avg Ingest Time per Page: {avg_ingest_time:.2f}s")
    print(f"-----------------------------")
    for item in hotpot_set:
        q = item["question"]
        gold_answer = item["answer"]
        print(f"\n[Question]: {q}")
        res = judge.evaluate(q, rag, wiki_dir)

        # Add metrics to console output
        print(f"  RAG Latency: {res.rag_metrics['latency']:.2f}s | Tokens: {res.rag_metrics['tokens']}")
        print(f"  Wiki Latency: {res.wiki_metrics['latency']:.2f}s | Tokens: {res.wiki_metrics['tokens']}")
        print(f"  Judge Verdict: Synthesis Score {res.judge_comparison.synthesis}/5")

        # Add gold answer to results for reference
        res_dict = res.model_dump()
        res_dict["gold_answer"] = gold_answer
        results.append(res_dict)
    # Save results
    output_path = project_root / "benchmarks" / "results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nBenchmark Complete! Results saved to {output_path}")


if __name__ == "__main__":
    run_benchmark()
