import os
import json
import time
from pathlib import Path
from typing import Optional
from openai import OpenAI
from dotenv import load_dotenv
from tenacity import retry, wait_exponential, stop_after_attempt

class SynthesizerAgent:
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, model: Optional[str] = None):
        load_dotenv()
        self.client = OpenAI(
            base_url=base_url or os.getenv("NVIDIA_BASE_URL"),
            api_key=api_key or os.getenv("NVIDIA_API_KEY")
        )
        self.model = model or os.getenv("NVIDIA_MODEL")

    @retry(wait=wait_exponential(multiplier=1, min=4, max=120), stop=stop_after_attempt(10))
    def synthesize(self, target_page: str, new_text: str, source_id: str, wiki_dir: Path) -> dict:
        start_time = time.time()
        page_path = wiki_dir / f"{target_page}.md"
        existing_content = ""
        if page_path.exists():
            with open(page_path, "r", encoding="utf-8") as f:
                existing_content = f.read()

        system_instruction = "Merge new facts into Markdown wiki page. Use headings and citations [^source_id]."
        prompt = f"Page: {target_page}\nExisting: {existing_content}\nNew: {new_text}"

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        
        active_time = time.time() - start_time
        usage = response.usage
        updated_content = response.choices[0].message.content
        
        with open(page_path, "w", encoding="utf-8") as f:
            f.write(updated_content)
            
        return {
            "content_len": len(updated_content),
            "active_time": active_time,
            "tokens": {
                "prompt": usage.prompt_tokens,
                "completion": usage.completion_tokens,
                "total": usage.total_tokens
            }
        }
