import json
import string
import re
from collections import Counter
from typing import Dict, Optional

def extract_final_answer_deterministic(text: str) -> str:
    """
    Parses the strict JSON output from the model to get the exact final answer.
    Zero secondary LLM calls. Zero regex.
    """
    try:
        data = json.loads(text)
        return str(data.get("final_answer", "")).strip()
    except json.JSONDecodeError:
        # Fallback if the model included extra text around the JSON block
        try:
            # Try to find JSON block
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end != -1:
                data = json.loads(text[start:end+1])
                return str(data.get("final_answer", "")).strip()
        except:
            pass
        return text.strip()

def normalize_answer(s: str) -> str:
    """Standard SQuAD-style normalization."""
    s = s.lower().strip()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    exclude = set(string.punctuation)
    s = ''.join(ch for ch in s if ch not in exclude)
    return ' '.join(s.split())

def calculate_metrics(prediction: str, ground_truth: str) -> Dict:
    pred_final = extract_final_answer_deterministic(prediction)
    
    norm_pred = normalize_answer(pred_final)
    norm_gold = normalize_answer(ground_truth)
    
    # EM
    em = (norm_pred == norm_gold)
    
    # F1
    pred_tokens = norm_pred.split()
    gold_tokens = norm_gold.split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        f1 = 1.0 if norm_pred == norm_gold else 0.0
    else:
        precision = 1.0 * num_same / len(pred_tokens)
        recall = 1.0 * num_same / len(gold_tokens)
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
    return {
        "f1": f1,
        "em": em,
        "extracted_answer": pred_final
    }
