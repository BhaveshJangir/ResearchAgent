"""
main.py
-------
CLI entry point. Run:

    python main.py "How does the James Webb telescope cool its instruments?"

Reads OPENAI_API_KEY from .env (or the environment).
"""

from __future__ import annotations

import sys

from dotenv import load_dotenv

from agent import ResearchAgent


def main() -> int:
    load_dotenv()

    if len(sys.argv) < 2:
        print('Usage: python main.py "your research question"')
        return 1

    question = " ".join(sys.argv[1:])
    agent = ResearchAgent(verbose=True)
    result = agent.run(question)

    print("\n" + "=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)
    print(result.final_answer)
    print(
        f"\n(Used {result.iterations} iterations, "
        f"{len(result.tool_calls)} tool calls.)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
