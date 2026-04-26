import os
import json
from pathlib import Path
from typing import List, Literal, Optional
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv

class Contradiction(BaseModel):
    is_contradiction: bool = Field(description="Whether a logical contradiction was found.")
    severity: Literal["none", "low", "high"] = Field(description="The severity of the conflict.")
    conflict_description: Optional[str] = Field(None, description="Detailed description of what contradicts.")
    reconciliation_suggestion: Optional[str] = Field(None, description="How the LLM suggests fixing the conflict.")

from tenacity import retry, wait_exponential, stop_after_attempt

class LinterAgent:
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, model: Optional[str] = None):
        load_dotenv()
        self.client = OpenAI(
            base_url=base_url or os.getenv("NVIDIA_BASE_URL"),
            api_key=api_key or os.getenv("NVIDIA_API_KEY")
        )
        self.model = model or os.getenv("NVIDIA_MODEL")

    @retry(wait=wait_exponential(multiplier=1, min=4, max=120), stop=stop_after_attempt(10))
    def lint(self, target_page: str, new_text: str, wiki_dir: Path) -> Contradiction:
        page_path = wiki_dir / f"{target_page}.md"
        if not page_path.exists():
            return Contradiction(is_contradiction=False, severity="none")

        with open(page_path, "r", encoding="utf-8") as f:
            current_content = f.read()

        system_instruction = (
            "You are the Linter agent. Your job is to identify logical contradictions.\n"
            "Compare the 'NEW SOURCE TEXT' against the 'EXISTING WIKI PAGE'.\n"
            "CRITICAL: You must return a JSON object with EXACTLY these fields:\n"
            "- \"is_contradiction\": boolean\n"
            "- \"severity\": \"none\", \"low\", or \"high\"\n"
            "- \"conflict_description\": string or null\n"
            "- \"reconciliation_suggestion\": string or null"
        )

        prompt = (
            f"Wiki Page: {target_page}.md\n\n"
            f"--- EXISTING WIKI CONTENT ---\n{current_content}\n\n"
            f"--- NEW SOURCE TEXT ---\n{new_text}\n\n"
            "Audit the new text against the wiki content."
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
        
        return Contradiction(**json.loads(response.choices[0].message.content))
