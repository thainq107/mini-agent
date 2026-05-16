import os
import sys

from dotenv import load_dotenv

from agent import ReActAgent
from llm import OpenAIClient


def main() -> None:
  """Main entry point for ReAct agent CLI."""
  load_dotenv()

  if len(sys.argv) < 2:
    print("Usage: python main.py \"Your question here\"")
    sys.exit(1)

  question = " ".join(sys.argv[1:])

  api_key = os.getenv("OPENAI_API_KEY")
  if not api_key:
    print("Error: OPENAI_API_KEY not set in .env")
    sys.exit(1)

  model = os.getenv("OPENAI_MODEL", "gpt-4o")

  client = OpenAIClient(api_key=api_key, model=model)
  agent = ReActAgent(client, max_iterations=10)

  result = agent.run(question)
  print(f"\nFinal Result: {result}")


if __name__ == "__main__":
  main()
