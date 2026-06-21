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
    hallucination_present: bool
    f1: float
    em: bool
    extracted_answer: str

class EvalResult(BaseModel):
    question: str
    gold_answer: str
    rag_answer: str
    wiki_answer: str
    zero_shot_answer: str
    rag_metrics: Dict
    wiki_metrics: Dict
    zero_shot_metrics: Dict
    judge_comparison: Comparison
    judge_model: str

class EvalJudge:
    def __init__(self):
        load_dotenv()
        self.client = AsyncOpenAI(
            api_key=os.getenv("JUDGE_API_KEY") or os.getenv("LLM_API_KEY") or os.getenv("NVIDIA_API_KEY"),
            base_url=os.getenv("JUDGE_BASE_URL") or os.getenv("LLM_BASE_URL") or os.getenv("NVIDIA_BASE_URL")
        )
        self.judge_model = os.getenv("JUDGE_MODEL", "gpt-4o")

    async def _zero_shot_query(self, question: str) -> dict:
        prompt = f"Answer concisely using JSON: {{\"reasoning\": \"...\", \"final_answer\": \"...\"}}\n\nQ: {question}"
        res = await self.client.chat.completions.create(
            model=os.getenv("LLM_MODEL") or os.getenv("NVIDIA_MODEL"), 
            messages=[{"role": "user", "content": prompt}], 
            response_format={"type": "json_object"}
        )
        return {"answer": res.choices[0].message.content, "tokens": res.usage.total_tokens}

    async def evaluate(self, question: str, gold_answer: str, rag_engine, wiki_dir: str) -> EvalResult:
        # Parallel Engine Queries + Zero Shot Baseline
        start_rag = time.perf_counter()
        rag_task = asyncio.create_task(rag_engine.query(question))
        
        from src.agents.synthesizer import SynthesizerAgent
        synth = SynthesizerAgent()
        start_wiki = time.perf_counter()
        wiki_task = asyncio.create_task(synth.answer_from_wiki(question, wiki_dir))
        
        start_zs = time.perf_counter()
        zero_shot_task = asyncio.create_task(self._zero_shot_query(question))
        
        rag_res, wiki_res, zero_shot_res = await asyncio.gather(rag_task, wiki_task, zero_shot_task)
        
        # Calculate specific latencies
        # Note: Since they are tasks on same loop, these will be similar if parallel, 
        # but the engines themselves should return their internal timing if possible.
        # For now, let's use the total time they took to complete.
        latency_rag = time.perf_counter() - start_rag
        latency_wiki = time.perf_counter() - start_wiki
        latency_zs = time.perf_counter() - start_zs

        # Calculate Metrics for all three
        wiki_metrics = calculate_metrics(wiki_res["answer"], gold_answer)
        rag_metrics = calculate_metrics(rag_res["answer"], gold_answer)
        zero_shot_metrics = calculate_metrics(zero_shot_res["answer"], gold_answer)

        # Impartial Reasoning Scores
        scores = await self._compare_answers(question, rag_res["answer"], wiki_res["answer"])
        
        comparison = Comparison(
            **scores,
            f1=wiki_metrics["f1"],
            em=wiki_metrics["em"],
            extracted_answer=wiki_metrics["extracted_answer"]
        )

        return EvalResult(
            question=question,
            gold_answer=gold_answer,
            rag_answer=rag_res["answer"],
            wiki_answer=wiki_res["answer"],
            zero_shot_answer=zero_shot_res["answer"],
            rag_metrics={"latency": latency_rag, "tokens": rag_res["tokens"], "f1": rag_metrics["f1"], "em": rag_metrics["em"]},
            wiki_metrics={"latency": latency_wiki, "tokens": wiki_res["tokens"], "f1": wiki_metrics["f1"], "em": wiki_metrics["em"]},
            zero_shot_metrics={"latency": latency_zs, "tokens": zero_shot_res["tokens"], "f1": zero_shot_metrics["f1"], "em": zero_shot_metrics["em"]},
            judge_comparison=comparison,
            judge_model=self.judge_model
        )

    async def _compare_answers(self, question: str, rag_ans: str, wiki_ans: str) -> dict:
        prompt = f"""Judge the AI answers to the following question.
Question: {question}

System A (RAG): {rag_ans}
System B (Wiki): {wiki_ans}

Evaluate both answers. Provide a JSON response with the following exact structure:
{{
  "synthesis": <1-5 score for how well information was combined>,
  "factuality": <1-5 score for accuracy compared to the golden standard>,
  "reasoning": <1-5 score for logical deduction>,
  "hallucination_present": <true/false, set to true if EITHER answer includes fabricated details, fake entities, or claims not logically supported by the question>
}}"""
        res = await self.client.chat.completions.create(
            model=self.judge_model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        return json.loads(res.choices[0].message.content)
