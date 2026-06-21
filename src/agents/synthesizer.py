import os
import json
import time
import re
import sqlite3
import asyncio
import aiofiles
import chromadb
from pathlib import Path
from typing import Optional, List, Set, Dict, Tuple
from openai import AsyncOpenAI
from dotenv import load_dotenv
from tenacity import retry, wait_exponential, stop_after_attempt

from src.utils.rag_baseline import sanitize_fts_query

_SENTENCE_MODEL = None

def get_sentence_model():
    global _SENTENCE_MODEL
    if _SENTENCE_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _SENTENCE_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    return _SENTENCE_MODEL

class SynthesizerAgent:
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, model: Optional[str] = None):
        load_dotenv()
        self.client = AsyncOpenAI(
            base_url=base_url or os.getenv("LLM_BASE_URL") or os.getenv("NVIDIA_BASE_URL"),
            api_key=api_key or os.getenv("LLM_API_KEY") or os.getenv("NVIDIA_API_KEY")
        )
        self.model = model or os.getenv("LLM_MODEL") or os.getenv("NVIDIA_MODEL")
        
        try:
            self.chroma_client = chromadb.HttpClient(host=os.getenv("CHROMA_HOST", "localhost"), port=8000)
        except:
            self.chroma_client = chromadb.PersistentClient(path=os.getenv("WIKI_CHROMA_PATH", "data/04_wiki/chroma"))
        self.title_collection = self.chroma_client.get_or_create_collection(name="wiki_titles")

    def _get_db(self, wiki_dir: Path) -> sqlite3.Connection:
        conn = sqlite3.connect(wiki_dir / "wiki_index.sqlite")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS pages_fts USING fts5(title, norm_title, content)")
        return conn

    async def _retrieve_wiki_graph(self, question: str, wiki_dir: Path) -> str:
        with self._get_db(wiki_dir) as db:
            q_norm = re.sub(r'[\W_]+', ' ', question).lower()
            
            # 1. Anchor Search
            keywords = [w for w in q_norm.split() if len(w) >= 4]
            match_query = sanitize_fts_query(" ".join(keywords))
            cursor = db.execute("SELECT title FROM pages_fts WHERE pages_fts MATCH ? ORDER BY rank LIMIT 5", (match_query,))
            anchors = [r[0] for r in cursor.fetchall()]
            
            context = []
            seen = set()
            queue = anchors
            model = get_sentence_model()
            q_emb = model.encode(question)
            
            for depth in range(2):
                next_candidates = []
                for page in queue:
                    if page in seen or len(seen) > 12: continue
                    seen.add(page)
                    path = wiki_dir / f"{page}.md"
                    if not path.exists(): continue
                    
                    async with aiofiles.open(path, mode='r') as f:
                        content = await f.read()
                        context.append(f"PAGE: {page}\n{content}")
                        # Extract link context
                        sentences = re.split(r'(?<=[.!?])\s+', content)
                        for s in sentences:
                            for l in re.findall(r"\[\[(.*?)\]\]", s):
                                if l not in seen: next_candidates.append((l, s))

                # FIX 2: GLOBAL FALLBACK (if walk hits dead end or insufficient data)
                if len(context) < 3:
                    # Trigger a global search for "orphan" pages
                    cursor = db.execute("SELECT title, content FROM pages_fts WHERE content MATCH ? LIMIT 3", (match_query,))
                    for t, c in cursor.fetchall():
                        if t not in seen:
                            context.append(f"PAGE: {t} (GLOBAL FALLBACK)\n{c}")
                            seen.add(t)

                if not next_candidates: break
                
                # Scored Guided Walk...
                cand_titles = [c[0] for c in next_candidates]
                cand_texts = [f"{c[0]}: {c[1]}" for c in next_candidates]
                scores = cos_sim(q_emb, model.encode(cand_texts))[0]
                queue = [cand_titles[i] for i in sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)[:3]]

            return "\n\n---\n\n".join(context)

    @retry(wait=wait_exponential(multiplier=1, min=4, max=120), stop=stop_after_attempt(10))
    async def answer_from_wiki(self, question: str, wiki_dir: Path, fast_mode: bool = False) -> dict:
        # Prompt includes "If you cannot answer from context, return 'NOT_FOUND'" for Negative Set testing
        if fast_mode:
            # Fast Mode: FTS only, no semantic walk
            with self._get_db(wiki_dir) as db:
                q_norm = re.sub(r'[\W_]+', ' ', question).lower()
                keywords = [w for w in q_norm.split() if len(w) >= 4]
                match_query = sanitize_fts_query(" ".join(keywords))
                cursor = db.execute("SELECT content FROM pages_fts WHERE content MATCH ? LIMIT 10", (match_query,))
                ctx = "\n\n---\n\n".join([r[0] for r in cursor.fetchall()])
        else:
            ctx = await self._retrieve_wiki_graph(question, wiki_dir)
            
        prompt = f"Answer using JSON: {{\"reasoning\": \"...\", \"final_answer\": \"...\"}}\nIf the answer is not in the context, set final_answer to 'NOT_FOUND'.\n\nCONTEXT:\n{ctx}\n\nQ: {question}"
        res = await self.client.chat.completions.create(model=self.model, messages=[{"role": "user", "content": prompt}], response_format={"type": "json_object"}, timeout=60.0)
        return {"answer": res.choices[0].message.content, "tokens": res.usage.total_tokens}

    @retry(wait=wait_exponential(multiplier=1, min=4, max=120), stop=stop_after_attempt(10))
    async def synthesize(self, target_page: str, new_text: str, source_id: str, wiki_dir: Path) -> dict:
        start_time = time.time()
        page_path = wiki_dir / f"{target_page}.md"
        
        current_content = ""
        if page_path.exists():
            async with aiofiles.open(page_path, mode='r') as f:
                current_content = await f.read()

        schema_path = wiki_dir.parent.parent / "WIKI_SCHEMA.md"
        schema_text = ""
        if schema_path.exists():
            async with aiofiles.open(schema_path, mode='r') as f:
                schema_text = await f.read()

        system_instruction = (
            f"{schema_text}\n\n"
            "--- TASK SPECIFIC CONTEXT ---\n"
            f"You are operating on the Wiki page: {target_page}.\n"
        )

        prompt = (
            f"--- EXISTING CONTENT ---\n{current_content}\n\n"
            f"--- NEW TEXT FROM SOURCE: {source_id} ---\n{new_text}"
        )

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            timeout=60.0
        )
        
        updated_content = response.choices[0].message.content
        # Strip potential markdown blocks
        if updated_content.startswith("```"):
            updated_content = re.sub(r"^```markdown\n|^```\n|```$", "", updated_content, flags=re.MULTILINE).strip()

        # Atomic Write
        async with aiofiles.open(page_path, mode='w') as f:
            await f.write(updated_content)

        # Update FTS5 Index for parity retrieval
        with self._get_db(wiki_dir) as db:
            db.execute("INSERT OR REPLACE INTO pages_fts(title, norm_title, content) VALUES (?, ?, ?)", 
                       (target_page, target_page.lower(), updated_content))
            db.commit()

        return {
            "active_time": time.time() - start_time,
            "tokens": response.usage.total_tokens
        }

from sentence_transformers.util import cos_sim
