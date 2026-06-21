import asyncio
import os
from openai import AsyncOpenAI
from dotenv import load_dotenv

async def main():
    load_dotenv()
    client = AsyncOpenAI(
        base_url=os.getenv("LLM_BASE_URL"),
        api_key=os.getenv("LLM_API_KEY")
    )
    model = os.getenv("LLM_MODEL")
    user_query = "Which tennis player won more Grand Slam titles, Henri Leconte or Jonathan Stark?"
    res = await client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": f"Query: {user_query}"}]
    )
    print("HyDE Output:", res.choices[0].message.content)

asyncio.run(main())
