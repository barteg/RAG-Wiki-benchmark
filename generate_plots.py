import json
import matplotlib.pyplot as plt
import numpy as np

# Load data
with open('results_athena.json', 'r') as f:
    data = json.load(f)

# Extract metrics
rag_latencies = []
wiki_fast_latencies = []
wiki_smart_latencies = []

rag_tokens = []
wiki_fast_tokens = []
wiki_smart_tokens = []

# For accuracy check
def check_correct(gold, ans_val):
    if isinstance(ans_val, dict):
        ans_str = str(ans_val.get('final_answer', ''))
    else:
        try:
            ans_json = json.loads(ans_val)
            if isinstance(ans_json, dict):
                ans_str = str(ans_json.get('final_answer', ''))
            else:
                ans_str = str(ans_val)
        except:
            ans_str = str(ans_val)
    
    ans_str = ans_str.lower().strip()
    gold_str = str(gold).lower().strip()
    
    return 1 if (gold_str in ans_str or ans_str in gold_str) and len(ans_str) > 0 else 0

rag_correct = 0
wiki_fast_correct = 0
wiki_smart_correct = 0

for item in data:
    gold = item.get('gold', '')
    
    # RAG
    if 'rag' in item:
        rag_latencies.append(item['rag'].get('latency', 0))
        rag_tokens.append(item['rag'].get('tokens', 0))
        rag_correct += check_correct(gold, item['rag'].get('answer', ''))
    
    # Wiki Fast
    if 'wiki_fast' in item:
        wiki_fast_latencies.append(item['wiki_fast'].get('latency', 0))
        wiki_fast_tokens.append(item['wiki_fast'].get('tokens', 0))
        wiki_fast_correct += check_correct(gold, item['wiki_fast'].get('answer', ''))
        
    # Wiki Smart
    if 'wiki_smart' in item:
        wiki_smart_latencies.append(item['wiki_smart'].get('latency', 0))
        wiki_smart_tokens.append(item['wiki_smart'].get('tokens', 0))
        wiki_smart_correct += check_correct(gold, item['wiki_smart'].get('answer', ''))

total = len(data)
rag_acc = (rag_correct / total) * 100 if total else 0
wiki_fast_acc = (wiki_fast_correct / total) * 100 if total else 0
wiki_smart_acc = (wiki_smart_correct / total) * 100 if total else 0

# 1. Latency Comparison Plot
plt.figure(figsize=(8, 6))
plt.boxplot([rag_latencies, wiki_fast_latencies, wiki_smart_latencies], labels=['Advanced RAG', 'Wiki Fast', 'Wiki Smart'])
plt.ylabel('Latency (seconds)')
plt.title('Latency Comparison')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('latency_comparison.pdf')
plt.close()

# 2. Judge Scores (Accuracy) Plot
plt.figure(figsize=(8, 6))
bars = plt.bar(['Advanced RAG', 'Wiki Fast', 'Wiki Smart'], [rag_acc, wiki_fast_acc, wiki_smart_acc], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
plt.ylabel('Accuracy (%)')
plt.title('Accuracy Comparison')
plt.ylim(0, 100)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 2, f'{yval:.1f}%', ha='center', va='bottom')
plt.savefig('judge_scores.pdf')
plt.close()

# 3. Latency Over Time Plot
plt.figure(figsize=(10, 5))
plt.plot(rag_latencies, label='Advanced RAG', alpha=0.8, marker='o', markersize=3)
plt.plot(wiki_fast_latencies, label='Wiki Fast', alpha=0.8, marker='s', markersize=3)
plt.plot(wiki_smart_latencies, label='Wiki Smart', alpha=0.8, marker='^', markersize=3)
plt.xlabel('Query Index')
plt.ylabel('End-to-End Latency (s)')
plt.title('Latency Over Time')
plt.legend()
plt.grid(linestyle='--', alpha=0.6)
plt.savefig('latency_over_time.pdf')
plt.close()

# 4. Tokens Over Time Plot
plt.figure(figsize=(10, 5))
plt.plot(rag_tokens, label='Advanced RAG', alpha=0.8, marker='o', markersize=3)
plt.plot(wiki_fast_tokens, label='Wiki Fast', alpha=0.8, marker='s', markersize=3)
plt.plot(wiki_smart_tokens, label='Wiki Smart', alpha=0.8, marker='^', markersize=3)
plt.xlabel('Query Index')
plt.ylabel('Tokens Generated/Processed')
plt.title('Tokens Over Time')
plt.legend()
plt.grid(linestyle='--', alpha=0.6)
plt.savefig('tokens_over_time.pdf')
plt.close()

print("Plots generated successfully.")
