import os
from pathlib import Path
from typing import List, Optional
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

from tenacity import retry, wait_exponential, stop_after_attempt

class RAGBaseline:
    def __init__(self, db_path: Path, api_key: Optional[str] = None, base_url: Optional[str] = None, model: Optional[str] = None):
        load_dotenv()
        self.client = OpenAI(
            base_url=base_url or os.getenv("NVIDIA_BASE_URL"),
            api_key=api_key or os.getenv("NVIDIA_API_KEY")
        )
        self.model = model or os.getenv("NVIDIA_MODEL")
        self.chroma_client = chromadb.PersistentClient(path=str(db_path))
        self.collection = self.chroma_client.get_or_create_collection(name="rag_baseline")

    def ingest(self, text: str, source_id: str):
        # Simple chunking by paragraph for the baseline
        chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
        
        ids = [f"{source_id}_{i}" for i in range(len(chunks))]
        metadatas = [{"source": source_id} for _ in range(len(chunks))]
        
        self.collection.add(
            documents=chunks,
            ids=ids,
            metadatas=metadatas
        )

    @retry(wait=wait_exponential(multiplier=1, min=4, max=120), stop=stop_after_attempt(10))
    def query(self, user_query: str, n_results: int = 3) -> dict:
        # 1. Retrieve
        results = self.collection.query(
            query_texts=[user_query],
            n_results=n_results
        )
        
        context_chunks = results['documents'][0]
        context_text = "\n\n".join(context_chunks)
        
        # 2. Generate
        system_instruction = (
            "You are a Traditional RAG Assistant.\n"
            "Answer the question ONLY using the provided context chunks.\n"
            "If the answer is not in the context, say you don't know."
        )
        
        prompt = f"CONTEXT:\n{context_text}\n\nQUESTION: {user_query}"
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        
        return {
            "answer": response.choices[0].message.content,
            "tokens": response.usage.total_tokens
        }
