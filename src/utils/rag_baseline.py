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

class AdvancedRAG:
    def __init__(self, db_path: Path, api_key: Optional[str] = None, base_url: Optional[str] = None, model: Optional[str] = None):
        load_dotenv()
        self.client = AsyncOpenAI(
            base_url=base_url or os.getenv("NVIDIA_BASE_URL"),
            api_key=api_key or os.getenv("NVIDIA_API_KEY")
        )
        self.model = model or os.getenv("NVIDIA_MODEL")
        self.db_path = db_path
        
        # SYMMETRY FIX: RAG also gets SQLite FTS5 for entity parity
        self.sqlite_path = db_path / "rag_index.sqlite"
        self._init_sqlite()
        
        try:
            self.chroma_client = chromadb.HttpClient(host=os.getenv("CHROMA_HOST", "localhost"), port=8000)
        except:
            self.chroma_client = chromadb.PersistentClient(path=str(db_path / "chroma"))
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
        self.collection.add(documents=chunks, ids=ids, metadatas=[{"source": source_id}]*len(chunks))
        # 2. FTS5 Ingest (For Parity)
        conn = sqlite3.connect(self.sqlite_path)
        data = [(c, source_id) for c in chunks]
        conn.executemany("INSERT INTO chunks_fts(content, source_id) VALUES (?, ?)", data)
        conn.commit()

    async def _hybrid_retrieve(self, query: str) -> List[str]:
        """Parallel FTS5 + Vector search for Tool Parity."""
        # a. FTS5 Search
        def fts_search():
            conn = sqlite3.connect(self.sqlite_path)
            words = " OR ".join([w for w in query.split() if len(w) > 4])
            if not words: return []
            cursor = conn.execute("SELECT content FROM chunks_fts WHERE content MATCH ? ORDER BY rank LIMIT 10", (words,))
            return [r[0] for r in cursor.fetchall()]

        fts_task = asyncio.to_thread(fts_search)
        # b. Vector Search
        vec_task = asyncio.to_thread(self.collection.query, query_texts=[query], n_results=10)
        
        fts_docs, vec_res = await asyncio.gather(fts_task, vec_task)
        vec_docs = vec_res['documents'][0] if vec_res.get('documents') else []
        
        # Simple RRF-style merge
        return list(dict.fromkeys(fts_docs[:5] + vec_docs[:5]))

    async def query(self, user_query: str) -> dict:
        expanded = (await self.client.chat.completions.create(
            model=self.model, messages=[{"role": "user", "content": f"Query: {user_query}"}]
        )).choices[0].message.content
        
        docs = await self._hybrid_retrieve(expanded)
        
        # Re-ranker and generation logic follows...
        prompt = f"Answer using JSON: {{\"reasoning\": \"...\", \"final_answer\": \"...\"}}\n\nCONTEXT:\n{docs}\n\nQ: {user_query}"
        ans = await self.client.chat.completions.create(model=self.model, messages=[{"role": "user", "content": prompt}], response_format={"type": "json_object"})
        return {"answer": ans.choices[0].message.content, "tokens": ans.usage.total_tokens}

import chromadb
