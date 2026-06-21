import os
import asyncio
import time
from pathlib import Path
from openai import AsyncOpenAI
from dotenv import load_dotenv

async def diag():
    load_dotenv()
    api_key = os.getenv("LLM_API_KEY") or os.getenv("NVIDIA_API_KEY")
    base_url = os.getenv("LLM_BASE_URL") or os.getenv("NVIDIA_BASE_URL")
    model = os.getenv("LLM_MODEL") or os.getenv("NVIDIA_MODEL")
    
    print(f"DIAG: URL={base_url}")
    print(f"DIAG: Model={model}")

    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    
    print("DIAG: Sending request...")
    start = time.time()
    try:
        res = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Hello, diagnostic test."}],
            timeout=30
        )
        print(f"DIAG: Success in {time.time()-start:.2f}s")
        print(f"DIAG: Response: {res.choices[0].message.content}")
    except Exception as e:
        print(f"DIAG: Error: {e}")

if __name__ == "__main__":
    asyncio.run(diag())
