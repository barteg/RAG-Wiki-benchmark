import os
import json
import asyncio
import sqlite3
import re
import httpx
from pathlib import Path
from typing import List, Optional, Dict
from openai import AsyncOpenAI
from dotenv import load_dotenv
from tenacity import retry, wait_exponential, stop_after_attempt
from langchain_text_splitters import RecursiveCharacterTextSplitter

def sanitize_fts_query(query: str) -> str:
    # Remove FTS special characters and punctuation
    clean = re.sub(r'[^a-zA-Z0-9\s]', ' ', query)
    words = [w for w in clean.split() if len(w) > 3]
    return " OR ".join(words)

class AdvancedRAG:
    def __init__(self, db_path: Path, api_key: Optional[str] = None, base_url: Optional[str] = None, model: Optional[str] = None):
        load_dotenv()
        self.client = AsyncOpenAI(
            base_url=base_url or os.getenv("LLM_BASE_URL") or os.getenv("NVIDIA_BASE_URL"),
            api_key=api_key or os.getenv("LLM_API_KEY") or os.getenv("NVIDIA_API_KEY")
        )
        self.model = model or os.getenv("LLM_MODEL") or os.getenv("NVIDIA_MODEL")
        self.db_path = db_path
        
        # SYMMETRY FIX: RAG also gets SQLite FTS5 for entity parity
        self.sqlite_path = db_path / "rag_index.sqlite"
        self._init_sqlite()
        
        try:
            self.chroma_client = chromadb.HttpClient(host=os.getenv("CHROMA_HOST", "localhost"), port=8000)
        except:
            # Persistent Client with minimal threading to avoid affinity errors
            self.chroma_client = chromadb.PersistentClient(
                path=str(db_path / "chroma")
            )
        self.collection = self.chroma_client.get_or_create_collection(name="advanced_rag_v5")

        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=400, chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def _init_sqlite(self):
        conn = sqlite3.connect(self.sqlite_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(content, source_id)")
        conn.commit()

    def ingest(self, text: str, source_id: str):
        chunks = self.text_splitter.split_text(text)
        ids = [f"{source_id}_{i}" for i in range(len(chunks))]
        # 1. Vector Ingest
        try:
            import onnxruntime
            onnxruntime.set_default_logger_severity(3)
        except: pass
        self.collection.add(documents=chunks, ids=ids, metadatas=[{"source": source_id}]*len(chunks))
        # 2. FTS5 Ingest (For Parity)
        with sqlite3.connect(self.sqlite_path) as conn:
            data = [(c, source_id) for c in chunks]
            conn.executemany("INSERT INTO chunks_fts(content, source_id) VALUES (?, ?)", data)
            conn.commit()

    async def _hybrid_retrieve(self, user_query: str, expanded_query: str) -> List[str]:
        """Parallel FTS5 + Vector search for Tool Parity."""
        # a. FTS5 Search
        def fts_search():
            conn = sqlite3.connect(self.sqlite_path)
            # FIX 2: Use the original user_query for FTS5 to ensure keywords aren't lost by HyDE hallucination
            words = sanitize_fts_query(user_query)
            if not words: return []
            cursor = conn.execute("SELECT content FROM chunks_fts WHERE content MATCH ? ORDER BY rank LIMIT 10", (words,))
            return [r[0] for r in cursor.fetchall()]

        fts_task = asyncio.to_thread(fts_search)
        # b. Vector Search
        # Vector search can use the expanded query for semantic matching
        vec_task = asyncio.to_thread(self.collection.query, query_texts=[expanded_query], n_results=10)
        
        fts_docs, vec_res = await asyncio.gather(fts_task, vec_task)
        vec_docs = vec_res['documents'][0] if vec_res.get('documents') else []
        
        # Simple RRF-style merge
        return list(dict.fromkeys(fts_docs[:5] + vec_docs[:5]))

    @retry(wait=wait_exponential(multiplier=1, min=4, max=60), stop=stop_after_attempt(5))
    async def query(self, user_query: str) -> dict:
        # FIX 1: Proper HyDE Query Expansion
        hyde_prompt = f"Write a brief, hypothetical answer to the following question. This will be used for semantic search, so include relevant keywords and entities. Do not include any introductory text.\n\nQuestion: {user_query}"
        
        try:
            expanded = (await self.client.chat.completions.create(
                model=self.model, messages=[{"role": "user", "content": hyde_prompt}], temperature=0.3, timeout=30.0
            )).choices[0].message.content
        except Exception:
            expanded = user_query # Fallback if HyDE fails
            
        docs = await self._hybrid_retrieve(user_query, expanded)
        
        # FIX 4: Proper Context Formatting
        context_str = "\n\n---\n\n".join(docs)
        
        # FIX 3: Symmetric Prompting with NOT_FOUND instruction
        prompt = f"Answer using JSON: {{\"reasoning\": \"...\", \"final_answer\": \"...\"}}\nIf the answer is not in the context, set final_answer to 'NOT_FOUND'.\n\nCONTEXT:\n{context_str}\n\nQ: {user_query}"
        
        try:
            ans = await self.client.chat.completions.create(model=self.model, messages=[{"role": "user", "content": prompt}], timeout=60.0)
            
            # Robust JSON extraction similar to Librarian
            import re
            content = ans.choices[0].message.content.strip()
            if "```" in content:
                match = re.search(r"\{.*\}", content, re.DOTALL)
                content = match.group(0) if match else content
                
            return {"answer": content, "tokens": ans.usage.total_tokens}
        except Exception as e:
            return {"answer": f"{{\"reasoning\": \"Error parsing or timeout\", \"final_answer\": \"NOT_FOUND\"}}", "tokens": 0}

import chromadb
