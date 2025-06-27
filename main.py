__author__ = "Se Hoon Kim(sehoon787@korea.ac.kr)"

# Standard imports
import asyncio

# Third-party imports
from dotenv import load_dotenv

# Custom imports
from engine.claude_engine import ClaudeEngine
from engine.gemini_engine import GeminiEngine
from engine.openai_engine import OpenAIEngine
from client.vibe_craft_client import VibeCraftClient

load_dotenv()


async def main():
    print("✅ 사용할 AI 모델을 선택하세요: claude / gemini / gpt (기본: claude)")
    model_choice = input("모델: ").strip().lower() or "claude"

    engine_map = {
        "claude": ClaudeEngine,
        "gemini": GeminiEngine,
        "gpt": OpenAIEngine
    }

    if model_choice not in engine_map:
        print("❌ 유효하지 않은 모델입니다.")
        return

    engine = engine_map[model_choice]()
    client = VibeCraftClient(engine)

    try:
        topic = input("🎤 주제를 입력하세요: ").strip()
        await client.run_pipeline(topic)
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
