import json
import os
import argparse
from pathlib import Path

def preprocess_hotpot(file_path: str, output_raw_dir: Path, output_eval_path: Path, num_examples: int):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Filter for comparison questions
    comparison_data = [item for item in data if item.get("type") == "comparison"][:num_examples]
    
    eval_set = []
    output_raw_dir.mkdir(parents=True, exist_ok=True)
    
    for i, item in enumerate(comparison_data):
        question = item["question"]
        answer = item["answer"]
        
        for title, sentences in item["context"]:
            content = "".join(sentences)
            filename = f"hotpot_{i}_{title.replace(' ', '_').replace('/', '_')}.txt"
            with open(output_raw_dir / filename, "w", encoding="utf-8") as f_out:
                f_out.write(content)
        
        eval_set.append({
            "question": question,
            "answer": answer,
            "id": item["_id"]
        })
    
    with open(output_eval_path, "w", encoding="utf-8") as f_eval:
        json.dump(eval_set, f_eval, indent=4)
    
    print(f"Preprocessed {len(comparison_data)} HotpotQA examples.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_examples", type=int, default=10)
    args = parser.parse_args()
    
    preprocess_hotpot(
        "hotpot_train_v1.1.json", 
        Path("data/01_raw"), 
        Path("benchmarks/hotpot_eval.json"),
        args.num_examples
    )
