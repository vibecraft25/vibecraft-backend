__author__ = "Se Hoon Kim(sehoon787@korea.ac.kr)"

# Standard imports
import asyncio

# Third-party imports
from dotenv import load_dotenv

# Custom imports
from mcp_agent.client import VibeCraftClient

load_dotenv()


async def main():
    print("✅ 사용할 AI 모델을 선택하세요: claude / gemini / gpt (기본: claude)")
    engine = "gemini"
    client = VibeCraftClient(engine)

    try:
        topic = "피자 일매출을 시각화하는 페이지를 제작할거야"
        await client.run_pipeline(
            topic,
            r"C:\Users\Administrator\Desktop\proj\vibecraft\samples\sample.csv")
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
