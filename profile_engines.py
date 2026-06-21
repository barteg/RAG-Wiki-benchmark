import asyncio
import time
import os
from pathlib import Path
from src.agents.synthesizer import SynthesizerAgent
from src.utils.rag_baseline import AdvancedRAG
from dotenv import load_dotenv

async def profile_query():
    load_dotenv()
    project_root = Path(__file__).parent
    wiki_dir = project_root / "data" / "04_wiki"
    rag_db_dir = project_root / "data" / "03_rag_db"
    
    rag = AdvancedRAG(db_path=rag_db_dir)
    synth = SynthesizerAgent()
    
    question = "Which magazine was started first Arthur's Magazine or First for Women?"
    
    print("Profiling RAG...")
    start = time.time()
    rag_res = await rag.query(question)
    print(f"RAG took {time.time()-start:.2f}s")
    
    print("\nProfiling Wiki...")
    start = time.time()
    wiki_res = await synth.answer_from_wiki(question, wiki_dir)
    print(f"Wiki took {time.time()-start:.2f}s")

if __name__ == "__main__":
    asyncio.run(profile_query())
