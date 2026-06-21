import json
import numpy as np

with open('results_athena.json', 'r') as f:
    data = json.load(f)

rag_latencies = [item['rag_metrics']['latency'] for item in data]
wiki_latencies = [item['wiki_metrics']['latency'] for item in data]
factuality_scores = [item['judge_comparison']['factuality'] for item in data]
synthesis_scores = [item['judge_comparison']['synthesis'] for item in data]

print(f"Benchmark Summary (n={len(data)})")
print("-" * 30)
print(f"RAG Avg Latency: {np.mean(rag_latencies):.2f}s")
print(f"Wiki Avg Latency: {np.mean(wiki_latencies):.2f}s")
print(f"Avg Factuality: {np.mean(factuality_scores):.2f}/5")
print(f"Avg Synthesis:  {np.mean(synthesis_scores):.2f}/5")
print("-" * 30)

print("\nDetailed Findings:")
for i, item in enumerate(data, 1):
    print(f"\nQ{i}: {item['question']}")
    print(f"Judge: F={item['judge_comparison']['factuality']} S={item['judge_comparison']['synthesis']}")
    print(f"Reasoning: {item['judge_comparison']['reasoning']}")
