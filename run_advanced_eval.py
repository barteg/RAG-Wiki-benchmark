import asyncio
import json
import os
import time
from pathlib import Path
from dotenv import load_dotenv
from openai import AsyncOpenAI
import matplotlib.pyplot as plt

from src.utils.rag_baseline import AdvancedRAG
from src.agents.synthesizer import SynthesizerAgent

load_dotenv()
MAX_CONCURRENT_TASKS = int(os.getenv("MAX_CONCURRENT_TASKS", "8"))
semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)

client = AsyncOpenAI(
    base_url=os.getenv("LLM_BASE_URL") or os.getenv("NVIDIA_BASE_URL"),
    api_key=os.getenv("LLM_API_KEY") or os.getenv("NVIDIA_API_KEY")
)
model = os.getenv("LLM_MODEL") or os.getenv("NVIDIA_MODEL")

async def generate_with_metrics(prompt: str) -> dict:
    start = time.perf_counter()
    try:
        res = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            stream_options={"include_usage": True},
            timeout=60.0
        )
        ttft = None
        content = ""
        tokens = 0
        async for chunk in res:
            if ttft is None:
                ttft = time.perf_counter() - start
            if len(chunk.choices) > 0 and chunk.choices[0].delta.content:
                content += chunk.choices[0].delta.content
            if hasattr(chunk, 'usage') and chunk.usage:
                tokens = chunk.usage.completion_tokens
        
        e2el = time.perf_counter() - start
        if tokens == 0:
            tokens = len(content.split())
        
        tpot = (e2el - ttft) / max(1, tokens) if ttft else 0
        return {"answer": content, "ttft": ttft, "tpot": tpot, "e2el": e2el, "tokens": tokens}
    except Exception as e:
        return {"answer": f"ERROR: {e}", "ttft": 0, "tpot": 0, "e2el": time.perf_counter() - start, "tokens": 0}

async def benchmark_item(item, rag, synth, wiki_dir, index):
    async with semaphore:
        q = item["question"]
        gold = item["answer"]
        print(f"[{index}] Starting: {q[:30]}...")
        
        # 1. RAG Retrieval
        rag_start = time.perf_counter()
        hyde_prompt = f"Write a brief hypothetical answer to the following question. Question: {q}"
        try:
            expanded = (await client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": hyde_prompt}], temperature=0.3, timeout=30.0
            )).choices[0].message.content
        except:
            expanded = q
        rag_docs = await rag._hybrid_retrieve(q, expanded)
        rag_retrieval_time = time.perf_counter() - rag_start
        
        rag_context_str = "\n\n---\n\n".join(rag_docs)
        rag_prompt = f"Answer using JSON. Q: {q} CONTEXT: {rag_context_str}"
        rag_gen = await generate_with_metrics(rag_prompt)
        
        # 2. Wiki Retrieval
        wiki_start = time.perf_counter()
        wiki_ctx = await synth._retrieve_wiki_graph(q, wiki_dir)
        wiki_retrieval_time = time.perf_counter() - wiki_start
        
        wiki_prompt = f"Answer using JSON. Q: {q} CONTEXT: {wiki_ctx}"
        wiki_gen = await generate_with_metrics(wiki_prompt)
        
        # Contextual Precision (simple metric: is gold answer in context?)
        rag_ctx_prec = 1 if gold.lower() in rag_context_str.lower() else 0
        wiki_ctx_prec = 1 if gold.lower() in wiki_ctx.lower() else 0
        
        return {
            "query_index": index,
            "level": item.get("level", "unknown"),
            "rag_ttft": rag_gen["ttft"],
            "rag_tpot": rag_gen["tpot"],
            "rag_retrieval_latency": rag_retrieval_time,
            "rag_e2el": rag_retrieval_time + rag_gen["e2el"],
            "rag_ctx_prec": rag_ctx_prec,
            
            "wiki_ttft": wiki_gen["ttft"],
            "wiki_tpot": wiki_gen["tpot"],
            "wiki_retrieval_latency": wiki_retrieval_time,
            "wiki_e2el": wiki_retrieval_time + wiki_gen["e2el"],
            "wiki_ctx_prec": wiki_ctx_prec,
        }

async def run_advanced_benchmark():
    project_root = Path(__file__).parent
    wiki_dir = project_root / "data" / "04_wiki"
    rag_db_dir = project_root / "data" / "03_rag_db"

    rag = AdvancedRAG(db_path=rag_db_dir)
    synth = SynthesizerAgent()

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

    LIMIT = 120
    data_file = project_root / "hotpot_train_v1.1.json"
    if not data_file.exists():
        data_file = project_root / "benchmarks" / "hotpot_eval.json"
        
    hotpot_set = get_stratified_data(data_file, LIMIT)
        
    tasks = [benchmark_item(item, rag, synth, wiki_dir, i) for i, item in enumerate(hotpot_set)]
    results = await asyncio.gather(*tasks)
    
    with open(project_root / "benchmarks" / "advanced_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    indices = [r["query_index"] for r in results]
    
    # 1. Latency Over Time Plot
    plt.figure(figsize=(10,6))
    plt.plot(indices, [r["rag_e2el"] for r in results], marker='o', label='RAG E2E Latency')
    plt.plot(indices, [r["wiki_e2el"] for r in results], marker='o', label='Wiki E2E Latency')
    plt.xlabel('Query Index (Over Time)')
    plt.ylabel('Latency (s)')
    plt.title('End-to-End Latency Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(project_root / "latency_over_time.pdf")
    
    # 2. TTFT Plot
    plt.figure(figsize=(10,6))
    plt.plot(indices, [r["rag_ttft"] for r in results], marker='s', label='RAG TTFT')
    plt.plot(indices, [r["wiki_ttft"] for r in results], marker='s', label='Wiki TTFT')
    plt.xlabel('Query Index')
    plt.ylabel('TTFT (s)')
    plt.title('Time To First Token (TTFT)')
    plt.legend()
    plt.grid(True)
    plt.savefig(project_root / "ttft_over_time.pdf")
    
    print("Done generating advanced metrics and plots.")

if __name__ == "__main__":
    asyncio.run(run_advanced_benchmark())
