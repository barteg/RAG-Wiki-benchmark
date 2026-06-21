import json

with open('results_athena.json', 'r') as f:
    data = json.load(f)

stats = {
    'rag': {'correct': 0, 'latency': 0.0, 'tokens': 0},
    'wiki_fast': {'correct': 0, 'latency': 0.0, 'tokens': 0},
    'wiki_smart': {'correct': 0, 'latency': 0.0, 'tokens': 0}
}

total = len(data)

def extract_answer(ans_val):
    if isinstance(ans_val, dict):
        return str(ans_val.get('final_answer', ''))
    try:
        ans_json = json.loads(ans_val)
        if isinstance(ans_json, dict):
            return str(ans_json.get('final_answer', ''))
    except:
        pass
    return str(ans_val)

for item in data:
    gold = str(item.get('gold', '')).lower().strip()
    
    for method in ['rag', 'wiki_fast', 'wiki_smart']:
        if method in item:
            ans = extract_answer(item[method].get('answer', '')).lower().strip()
            
            if gold in ans or ans in gold and len(ans)>0:
                stats[method]['correct'] += 1
            
            stats[method]['latency'] += item[method].get('latency', 0)
            stats[method]['tokens'] += item[method].get('tokens', 0)

for method in stats:
    print(f"--- {method.upper()} ---")
    print(f"Accuracy: {stats[method]['correct'] / total * 100:.2f}%")
    print(f"Average Latency: {stats[method]['latency'] / total:.4f}s")
    print(f"Average Tokens: {stats[method]['tokens'] / total:.2f}")
    print()
