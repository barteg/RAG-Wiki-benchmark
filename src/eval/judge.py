import os
import json
import time
import asyncio
from typing import Dict, List, Optional
from pydantic import BaseModel
from openai import AsyncOpenAI
from dotenv import load_dotenv
from src.utils.metrics import calculate_metrics

class Comparison(BaseModel):
    synthesis: int
    factuality: int
    reasoning: int
    f1: float
    em: bool
    extracted_answer: str

class EvalResult(BaseModel):
    question: str
    gold_answer: str
    rag_answer: str
    wiki_answer: str
    rag_metrics: Dict
    wiki_metrics: Dict
    judge_comparison: Comparison
    judge_model: str

class EvalJudge:
    def __init__(self):
        load_dotenv()
        self.client = AsyncOpenAI(
            api_key=os.getenv("JUDGE_API_KEY") or os.getenv("NVIDIA_API_KEY"),
            base_url=os.getenv("JUDGE_BASE_URL") or os.getenv("NVIDIA_BASE_URL")
        )
        self.judge_model = os.getenv("JUDGE_MODEL", "gpt-4o")

    async def evaluate(self, question: str, gold_answer: str, rag_engine, wiki_dir: str) -> EvalResult:
        # Parallel Engine Queries
        rag_task = asyncio.create_task(rag_engine.query(question))
        from src.agents.synthesizer import SynthesizerAgent
        synth = SynthesizerAgent()
        wiki_task = asyncio.create_task(synth.answer_from_wiki(question, wiki_dir))
        
        start_time = time.time()
        rag_res, wiki_res = await asyncio.gather(rag_task, wiki_task)
        latency = time.time() - start_time

        # FIX 5: Deterministic Metrics with Extraction Prompt
        # We judge the Wiki answer (the one we care about for the compiler)
        metrics = calculate_metrics(wiki_res["answer"], gold_answer)

        # Impartial Reasoning Scores
        scores = await self._compare_answers(question, rag_res["answer"], wiki_res["answer"])
        
        comparison = Comparison(
            **scores,
            f1=metrics["f1"],
            em=metrics["em"],
            extracted_answer=metrics["extracted_answer"]
        )

        return EvalResult(
            question=question,
            gold_answer=gold_answer,
            rag_answer=rag_res["answer"],
            wiki_answer=wiki_res["answer"],
            rag_metrics={"latency": latency, "tokens": rag_res["tokens"]},
            wiki_metrics={"latency": latency, "tokens": wiki_res["tokens"]},
            judge_comparison=comparison,
            judge_model=self.judge_model
        )

    async def _compare_answers(self, question: str, rag_ans: str, wiki_ans: str) -> dict:
        prompt = f"Judge AI answers to: {question}\n\nA: {rag_ans}\nB: {wiki_ans}\nRate 1-5 for synthesis, factuality, reasoning. Return JSON."
        res = await self.client.chat.completions.create(
            model=self.judge_model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        return json.loads(res.choices[0].message.content)
