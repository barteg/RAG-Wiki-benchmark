import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path

def generate_visualizations(results_path: str, output_dir: str):
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    # Check if this is the summary format or raw list
    if isinstance(data, dict) and "raw_results" in data:
        df = pd.DataFrame(data["raw_results"])
        meta = data["benchmark_metadata"]
    else:
        df = pd.DataFrame(data)
        meta = {}

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. Latency Comparison
    plt.figure(figsize=(10, 6))
    rag_lat = [r['rag_metrics']['latency'] for r in df.to_dict('records')]
    wiki_lat = [r['wiki_metrics']['latency'] for r in df.to_dict('records')]
    
    lat_df = pd.DataFrame({
        'Question': range(len(df)),
        'RAG': rag_lat,
        'Wiki': wiki_lat
    }).melt('Question', var_name='Method', value_name='Latency (s)')
    
    sns.lineplot(data=lat_df, x='Question', y='Latency (s)', hue='Method', marker='o')
    plt.title('Time per Question (Latency)')
    plt.savefig(f"{output_dir}/latency_comparison.png")
    plt.close()

    # 2. Token Usage Efficiency
    plt.figure(figsize=(10, 6))
    rag_tokens = [r['rag_metrics'].get('tokens', 0) for r in df.to_dict('records')]
    wiki_tokens = [r['wiki_metrics'].get('tokens', 0) for r in df.to_dict('records')]
    
    token_df = pd.DataFrame({
        'Question': range(len(df)),
        'RAG': rag_tokens,
        'Wiki': wiki_tokens
    }).melt('Question', var_name='Method', value_name='Tokens')
    
    sns.barplot(data=token_df, x='Question', y='Tokens', hue='Method')
    plt.title('Token Usage per Query')
    plt.savefig(f"{output_dir}/token_usage.png")
    plt.close()

    # 3. Cumulative Cost (Economic Crossover)
    if meta:
        plt.figure(figsize=(10, 6))
        wiki_ingest = meta.get('wiki_ingest_tokens', 0)
        avg_wiki_q = sum(wiki_tokens) / len(wiki_tokens) if wiki_tokens else 0
        avg_rag_q = sum(rag_tokens) / len(rag_tokens) if rag_tokens else 0
        
        x = list(range(0, int(meta.get('economic_crossover_queries', 100) * 2)))
        y_rag = [avg_rag_q * i for i in x]
        y_wiki = [wiki_ingest + (avg_wiki_q * i) for i in x]
        
        plt.plot(x, y_rag, label='Total RAG Cost', color='red')
        plt.plot(x, y_wiki, label='Total Wiki Cost (Ingest + Query)', color='blue', linestyle='--')
        plt.axvline(x=meta.get('economic_crossover_queries', 0), color='green', linestyle=':', label='Crossover Point')
        
        plt.xlabel('Number of Queries')
        plt.ylabel('Total Tokens')
        plt.title('Economic Amortization: RAG vs Wiki')
        plt.legend()
        plt.savefig(f"{output_dir}/economic_crossover.png")
        plt.close()

    # 4. Judge Scores (Factuality & Synthesis)
    plt.figure(figsize=(10, 6))
    scores = []
    for r in df.to_dict('records'):
        comp = r.get('judge_comparison', {})
        # Extract explicitly to avoid non-numeric pollution
        scores.append({'Metric': 'Factuality', 'Score': float(comp.get('factuality', 0))})
        scores.append({'Metric': 'Synthesis', 'Score': float(comp.get('synthesis', 0))})
        scores.append({'Metric': 'Reasoning', 'Score': float(comp.get('reasoning_score', comp.get('reasoning_rating', 0)))})
    
    if scores:
        score_df = pd.DataFrame(scores)
        sns.boxplot(data=score_df, x='Metric', y='Score')
        plt.title('LLM Judge Quality Metrics (Wiki Path)')
        plt.ylim(0, 6)
        plt.savefig(f"{output_dir}/quality_scores.png")
    plt.close()

    print(f"Visualizations generated in: {output_dir}")

if __name__ == "__main__":
    import sys
    res_file = sys.argv[1] if len(sys.argv) > 1 else "benchmarks/results.json"
    generate_visualizations(res_file, "benchmarks/plots")
