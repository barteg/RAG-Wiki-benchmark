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
            base_url=base_url or os.getenv("NVIDIA_BASE_URL"),
            api_key=api_key or os.getenv("NVIDIA_API_KEY")
        )
        self.model = model or os.getenv("NVIDIA_MODEL")
        
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
        db = self._get_db(wiki_dir)
        q_norm = re.sub(r'[\W_]+', ' ', question).lower()
        
        # 1. Anchor Search
        keywords = [w for w in q_norm.split() if len(w) >= 4]
        match_query = " OR ".join(keywords)
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
            scores = cos_sim(q_emb, model.encode(candidate_texts))[0] # Simplified logic
            queue = [cand_titles[i] for i in sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)[:3]]

        return "\n\n---\n\n".join(context)

    async def answer_from_wiki(self, question: str, wiki_dir: Path) -> dict:
        # Prompt includes "If you cannot answer from context, return 'NOT_FOUND'" for Negative Set testing
        ctx = await self._retrieve_wiki_graph(question, wiki_dir)
        prompt = f"Answer using JSON: {{\"reasoning\": \"...\", \"final_answer\": \"...\"}}\nIf the answer is not in the context, set final_answer to 'NOT_FOUND'.\n\nCONTEXT:\n{ctx}\n\nQ: {question}"
        res = await self.client.chat.completions.create(model=self.model, messages=[{"role": "user", "content": prompt}], response_format={"type": "json_object"})
        return {"answer": res.choices[0].message.content, "tokens": res.usage.total_tokens}

    async def synthesize(self, target_page: str, new_text: str, source_id: str, wiki_dir: Path) -> dict:
        # Standard synthesis logic remains...
        pass

from sentence_transformers.util import cos_sim
