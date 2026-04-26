import os
import json
import time
from pathlib import Path
from typing import List, Literal, Optional
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv
from tenacity import retry, wait_exponential, stop_after_attempt

class RoutingDecision(BaseModel):
    action: Literal["create_new_page", "update_existing_page"]
    target_page: str
    related_pages: List[str] = []
    reasoning: str
    tokens: dict = {}
    active_time: float = 0.0

class LibrarianAgent:
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, model: Optional[str] = None):
        load_dotenv()
        self.client = OpenAI(
            base_url=base_url or os.getenv("NVIDIA_BASE_URL"),
            api_key=api_key or os.getenv("NVIDIA_API_KEY")
        )
        self.model = model or os.getenv("NVIDIA_MODEL")

    def read_index(self, wiki_dir: Path) -> dict:
        index_path = wiki_dir / "index.json"
        if not index_path.exists():
            return {"pages": {}}
        with open(index_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @retry(wait=wait_exponential(multiplier=1, min=4, max=120), stop=stop_after_attempt(10))
    def route_text(self, text: str, source_id: str, wiki_dir: Path) -> RoutingDecision:
        start_time = time.time()
        index_data = self.read_index(wiki_dir)
        existing_pages = list(index_data["pages"].keys())
        
        system_instruction = (
            "You are the Librarian agent. Decide if text belongs to an existing page or a new one.\n"
            f"Existing: {existing_pages}\n"
            "Return JSON: action, target_page, related_pages, reasoning."
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": f"Source: {source_id}\nText: {text}"}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        
        active_time = time.time() - start_time
        usage = response.usage
        decision_dict = json.loads(response.choices[0].message.content)
        decision_dict["tokens"] = {
            "prompt": usage.prompt_tokens,
            "completion": usage.completion_tokens,
            "total": usage.total_tokens
        }
        decision_dict["active_time"] = active_time
        return RoutingDecision(**decision_dict)
