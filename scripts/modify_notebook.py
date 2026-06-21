import json

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

with open('results_athena.json', 'r') as f:
    data = json.load(f)

rag_fails_smart_wins = []
smart_fails_rag_wins = []

for item in data:
    gold = item.get('gold', '')
    rag_ans = item.get('rag', {}).get('answer', '')
    smart_ans = item.get('wiki_smart', {}).get('answer', '')
    
    rag_correct = check_correct(gold, rag_ans)
    smart_correct = check_correct(gold, smart_ans)
    
    if rag_correct == 0 and smart_correct == 1:
        rag_fails_smart_wins.append(item)
    elif rag_correct == 1 and smart_correct == 0:
        smart_fails_rag_wins.append(item)

# Format the text
md_content = "### Qualitative Examples: Where One Pipeline Outperforms the Other\n\n"

md_content += "#### Case 1: Advanced RAG Fails, Wiki Smart Wins\n"
md_content += "In these scenarios, the dynamic retrieval of RAG fails to properly connect the multi-hop entities, while the pre-compiled graph of Wiki Smart correctly navigates the links.\n\n"

for i, item in enumerate(rag_fails_smart_wins[:2]):
    md_content += f"**Example {i+1}**\n"
    md_content += f"- **Question**: {item.get('question')}\n"
    md_content += f"- **Gold Answer**: {item.get('gold')}\n"
    
    rag = item.get('rag', {}).get('answer', '')
    try: rag_final = json.loads(rag).get('final_answer', '')
    except: rag_final = str(rag)
    
    smart = item.get('wiki_smart', {}).get('answer', '')
    try: smart_final = json.loads(smart).get('final_answer', '')
    except: smart_final = str(smart)

    md_content += f"- **RAG Answer**: {rag_final} *(Incorrect)*\n"
    md_content += f"- **Wiki Smart Answer**: {smart_final} *(Correct)*\n\n"

md_content += "#### Case 2: Wiki Smart Fails, Advanced RAG Wins\n"
md_content += "In these scenarios, the guided walk of Wiki Smart might miss a crucial link or hallucinate a connection that wasn't properly compiled, whereas the brute-force semantic search of RAG successfully finds the raw context.\n\n"

for i, item in enumerate(smart_fails_rag_wins[:2]):
    md_content += f"**Example {i+1}**\n"
    md_content += f"- **Question**: {item.get('question')}\n"
    md_content += f"- **Gold Answer**: {item.get('gold')}\n"
    
    rag = item.get('rag', {}).get('answer', '')
    try: rag_final = json.loads(rag).get('final_answer', '')
    except: rag_final = str(rag)
    
    smart = item.get('wiki_smart', {}).get('answer', '')
    try: smart_final = json.loads(smart).get('final_answer', '')
    except: smart_final = str(smart)

    md_content += f"- **RAG Answer**: {rag_final} *(Correct)*\n"
    md_content += f"- **Wiki Smart Answer**: {smart_final} *(Incorrect)*\n\n"

# Modify the notebook
with open('benchmark_report.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [line + '\n' for line in md_content.split('\n')][:-1] # Add back newlines as required by jupyter
}

# Ensure no trailing newline in the last element just to be clean
if new_cell["source"] and new_cell["source"][-1].endswith('\n'):
    new_cell["source"][-1] = new_cell["source"][-1][:-1]

nb['cells'].append(new_cell)

with open('benchmark_report.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully.")
