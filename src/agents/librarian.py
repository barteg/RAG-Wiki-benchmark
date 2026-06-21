import os
import json
import time
import asyncio
from pathlib import Path
from typing import List, Literal, Optional
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
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
        self.client = AsyncOpenAI(
            base_url=base_url or os.getenv("LLM_BASE_URL") or os.getenv("NVIDIA_BASE_URL"),
            api_key=api_key or os.getenv("LLM_API_KEY") or os.getenv("NVIDIA_API_KEY")
        )
        self.model = model or os.getenv("LLM_MODEL") or os.getenv("NVIDIA_MODEL")

    async def read_index(self, wiki_dir: Path) -> dict:
        index_path = wiki_dir / "index.json"
        if not index_path.exists():
            return {"pages": {}}
        import aiofiles
        async with aiofiles.open(index_path, mode='r', encoding="utf-8") as f:
            return json.loads(await f.read())

    @retry(wait=wait_exponential(multiplier=1, min=4, max=120), stop=stop_after_attempt(10))
    async def route_text(self, text: str, source_id: str, wiki_dir: Path) -> RoutingDecision:
        start_time = time.time()
        print(f"    [Librarian DEBUG] Reading index for {source_id}...")
        index_data = await self.read_index(wiki_dir)
        existing_pages = list(index_data["pages"].keys())
        
        import aiofiles
        schema_path = wiki_dir.parent.parent / "WIKI_SCHEMA.md"
        schema_text = ""
        if schema_path.exists():
            async with aiofiles.open(schema_path, mode='r') as f:
                schema_text = await f.read()

        system_instruction = (
            f"{schema_text}\n\n"
            "--- TASK SPECIFIC CONTEXT ---\n"
            f"Existing Pages: {existing_pages[:50]} {'...' if len(existing_pages)>50 else ''}\n"
        )

        print(f"    [Librarian DEBUG] Sending request for {source_id}...")
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": f"Source: {source_id}\nText: {text[:1000]}"} # Truncate for stability
                ],
                # response_format={"type": "json_object"}, # Disabled for wider compatibility
                temperature=0.1,
                timeout=60.0
            )
            print(f"    [Librarian DEBUG] Response received for {source_id}.")
            raw_content = response.choices[0].message.content
            print(f"    [Librarian DEBUG] Raw Content: {raw_content[:200]}...")
            
            # Robust JSON Extract
            import re
            content = raw_content.strip()
            if "```" in content:
                match = re.search(r"\{.*\}", content, re.DOTALL)
                content = match.group(0) if match else raw_content

            try:
                decision_dict = json.loads(content)
            except json.JSONDecodeError:
                print(f"    [Librarian DEBUG] JSON parse failed, attempting regex fallback...")
                decision_dict = {}
                # Fallback regex parsing for plaintext responses
                action_match = re.search(r'(?i)action\s*:\s*"?([^"\n]+)"?', raw_content)
                target_match = re.search(r'(?i)target_?page\s*:\s*"?([^"\n]+)"?', raw_content)
                if action_match: decision_dict["action"] = action_match.group(1).strip()
                if target_match: decision_dict["target_page"] = target_match.group(1).strip()

            try:
                # Ensure required fields
                if "action" not in decision_dict or decision_dict["action"] not in ["create_new_page", "update_existing_page"]: 
                    decision_dict["action"] = "create_new_page"
                if "target_page" not in decision_dict: 
                    decision_dict["target_page"] = source_id.replace(".txt", "")
                
                res = RoutingDecision(**decision_dict)
                res.active_time = time.time() - start_time
                res.tokens = {
                    "prompt": response.usage.prompt_tokens,
                    "completion": response.usage.completion_tokens,
                    "total": response.usage.total_tokens
                }
                print(f"    [Librarian DEBUG] Routing Success: {res.target_page}")
                return res
            except Exception as inner_e:
                print(f"    [Librarian DEBUG] Processing Error: {inner_e}")
                raise inner_e
                
        except Exception as e:
            print(f"    [Librarian DEBUG] Request/API Error for {source_id}: {e}")
            raise e
