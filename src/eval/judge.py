import os
import json
import time
from pathlib import Path
from typing import List, Literal, Optional
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv
from tenacity import retry, wait_exponential, stop_after_attempt

from src.utils.rag_baseline import RAGBaseline

class EvalScore(BaseModel):
    factuality: int = Field(description="Score from 1-5 on how factually accurate the answer is.")
    synthesis: int = Field(description="Score from 1-5 on how well the answer connects multiple facts.")
    hallucination_present: bool = Field(description="Whether the answer contains facts not in the context.")
    reasoning: str = Field(description="Brief explanation of the scores.")

class BenchmarkResult(BaseModel):
    question: str
    rag_answer: str
    wiki_answer: str
    rag_metrics: dict
    wiki_metrics: dict
    judge_comparison: EvalScore

class EvalJudge:
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, model: Optional[str] = None):
        load_dotenv()
        self.client = OpenAI(
            base_url=base_url or os.getenv("NVIDIA_BASE_URL"),
            api_key=api_key or os.getenv("NVIDIA_API_KEY")
        )
        self.model = model or os.getenv("NVIDIA_MODEL")

    @retry(wait=wait_exponential(multiplier=1, min=4, max=120), stop=stop_after_attempt(10))
    def query_wiki(self, question: str, wiki_dir: Path) -> dict:
        index_path = wiki_dir / "index.json"
        with open(index_path, "r") as f:
            index = json.load(f)
        
        pages = list(index["pages"].keys())
        
        prompt = (
            f"Given this list of wiki pages: {pages}\n"
            f"And this multi-hop question: {question}\n"
            "Pick the 1 or 2 most relevant pages needed to answer this question. "
            "Return ONLY the page names separated by commas."
        )
        res = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        total_tokens = res.usage.total_tokens
        selected_pages = [p.strip().replace(".md", "") for p in res.choices[0].message.content.split(",")]
        
        wiki_context = ""
        for page in selected_pages:
            page_path = wiki_dir / f"{page}.md"
            if page_path.exists():
                with open(page_path, "r") as f:
                    wiki_context += f"\n--- PAGE: {page} ---\n" + f.read()
        
        if not wiki_context:
            return {"answer": "Relevant wiki pages not found.", "tokens": total_tokens}

        prompt = f"WIKI CONTEXT:\n{wiki_context}\n\nQUESTION: {question}\nAnswer using the wiki context."
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        total_tokens += response.usage.total_tokens
        return {"answer": response.choices[0].message.content, "tokens": total_tokens}

    @retry(wait=wait_exponential(multiplier=1, min=4, max=120), stop=stop_after_attempt(10))
    def evaluate(self, question: str, rag: RAGBaseline, wiki_dir: Path) -> BenchmarkResult:
        # 1. RAG Query
        start = time.time()
        rag_res = rag.query(question)
        rag_latency = time.time() - start

        # 2. Wiki Query
        start = time.time()
        wiki_res = self.query_wiki(question, wiki_dir)
        wiki_latency = time.time() - start

        # 3. Judge
        system_instruction = (
            "You are a Senior LLM Evaluation Judge.\n"
            "Compare two answers (RAG vs Wiki) against a 'Golden Standard' of reasoning.\n"
            "CRITICAL: You must return a JSON object with EXACTLY these fields:\n"
            "- \"factuality\": integer 1-5\n"
            "- \"synthesis\": integer 1-5\n"
            "- \"hallucination_present\": boolean\n"
            "- \"reasoning\": string"
        )
        
        prompt = (
            f"QUESTION: {question}\n\n"
            f"--- RAG ANSWER ---\n{rag_res['answer']}\n\n"
            f"--- WIKI ANSWER ---\n{wiki_res['answer']}\n\n"
            "Evaluate both. Provide structured scores."
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        
        judge_score = EvalScore(**json.loads(response.choices[0].message.content))

        return BenchmarkResult(
            question=question,
            rag_answer=rag_res['answer'],
            wiki_answer=wiki_res['answer'],
            rag_metrics={"latency": rag_latency, "tokens": rag_res['tokens']},
            wiki_metrics={"latency": wiki_latency, "tokens": wiki_res['tokens']},
            judge_comparison=judge_score
        )
