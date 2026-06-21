import json
import numpy as np
from src.utils.metrics import calculate_metrics

def summarize():
    with open('benchmarks/comparative_results.json', 'r') as f:
        data = json.load(f)

    stats = {
        "rag": {"lat": [], "tokens": [], "f1": [], "em": []},
        "wiki_fast": {"lat": [], "tokens": [], "f1": [], "em": []},
        "wiki_smart": {"lat": [], "tokens": [], "f1": [], "em": []}
    }

    for item in data:
        for key in stats.keys():
            res = item[key]
            stats[key]["lat"].append(res["latency"])
            stats[key]["tokens"].append(res["tokens"])
            
            m = calculate_metrics(res["answer"], item["gold"])
            stats[key]["f1"].append(m["f1"])
            stats[key]["em"].append(1 if m["em"] else 0)

    print(f"{'Engine':<12} | {'Lat':<6} | {'Tok':<6} | {'F1':<6} | {'EM':<4}")
    print("-" * 45)
    for key, val in stats.items():
        print(f"{key:<12} | {np.mean(val['lat']):>5.1f}s | {np.mean(val['tokens']):>6.0f} | {np.mean(val['f1']):>6.2f} | {np.mean(val['em']):>4.2f}")

if __name__ == "__main__":
    summarize()
