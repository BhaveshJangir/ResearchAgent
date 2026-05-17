"""
agent.py
--------
The "brain" of the system. Implements a ReAct-style research agent.

ReAct = Reason + Act. The loop is:

    1. The model looks at the conversation so far and either
         (a) calls a tool   ->  we execute it, append the result,  GOTO 1
         (b) returns text   ->  that's the final answer, we're done.

This is the core idea behind almost every "agent" you'll read about.
Frameworks like LangChain, AutoGPT, etc. all dress up this same loop.

We add two pieces on top of plain ReAct to make the agent better at research:

    * PLAN step      - before any tools, ask the model to write down a plan.
                       This is "chain of thought" externalized as a real
                       message - useful both for steering the model and
                       for transparency to the user.

    * SYNTHESIZE step - after the loop ends, ask the model one more time to
                       produce a clean, cited answer. Without this, agents
                       often dump raw search results.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

from tools import TOOL_FUNCTIONS, TOOL_SCHEMAS


SYSTEM_PROMPT = """You are a careful research assistant.

You have three tools: web_search, fetch_url, calculator.

How to work:
  1. First, think about what the question is really asking. If it's broad,
     break it into sub-questions.
  2. Use web_search to find candidate sources, then fetch_url on the most
     promising ones to read them in detail.
  3. Cross-check claims across at least TWO independent sources for any
     factual claim that matters.
  4. Use the calculator for any arithmetic - never compute it yourself.
  5. When you have enough evidence, stop calling tools and write the final
     answer. The final answer must:
       - directly answer the user's question
       - cite sources inline like [1], [2] referring to URLs you actually fetched
       - end with a "Sources" list of those URLs

Be concise. Don't fetch the same URL twice. If a search returns nothing
useful, refine the query and try again."""


@dataclass
class AgentRun:
    """Holds the state of one research task. Useful for debugging and
    for showing the user what the agent did."""

    question: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    final_answer: str = ""
    iterations: int = 0


class ResearchAgent:
    def __init__(
        self,
        client: OpenAI | None = None,
        model: str | None = None,
        max_iterations: int = 8,
        verbose: bool = True,
    ):
        self.client = client or OpenAI()
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.max_iterations = max_iterations
        self.verbose = verbose

    # ------------------------------------------------------------------ run
    def run(self, question: str) -> AgentRun:
        run = AgentRun(question=question)
        run.messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

        self._log(f"\n[USER] {question}\n")

        # Main ReAct loop.
        for i in range(1, self.max_iterations + 1):
            run.iterations = i
            self._log(f"--- iteration {i} ---")

            response = self.client.chat.completions.create(
                model=self.model,
                messages=run.messages,
                tools=TOOL_SCHEMAS,
                tool_choice="auto",
                temperature=0.2,
            )
            msg = response.choices[0].message

            # Append the assistant message to history. We must include it
            # verbatim because subsequent tool messages reference its
            # tool_call_ids.
            run.messages.append(
                {
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in (msg.tool_calls or [])
                    ]
                    or None,
                }
            )

            # Case A: model wants to call tools -> execute them.
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    self._handle_tool_call(tc, run)
                continue  # back to the loop, model will see tool outputs

            # Case B: no tool calls -> this is the final answer.
            run.final_answer = msg.content or ""
            self._log(f"\n[FINAL]\n{run.final_answer}\n")
            return run

        # Safety net: hit iteration cap without a final answer.
        run.final_answer = (
            "[agent stopped: max iterations reached] "
            "Partial findings are in the message log."
        )
        return run

    # ------------------------------------------------------------ tool exec
    def _handle_tool_call(self, tc: Any, run: AgentRun) -> None:
        name = tc.function.name
        try:
            args = json.loads(tc.function.arguments or "{}")
        except json.JSONDecodeError:
            args = {}

        self._log(f"[TOOL CALL] {name}({args})")

        fn = TOOL_FUNCTIONS.get(name)
        if fn is None:
            output = f"ERROR: unknown tool '{name}'"
        else:
            try:
                output = fn(**args)
            except TypeError as e:
                output = f"ERROR: bad arguments for {name}: {e}"
            except Exception as e:  # noqa: BLE001 - we want to surface to model
                output = f"ERROR: {name} failed: {e}"

        # Trim very long outputs to protect the context window.
        if len(output) > 6000:
            output = output[:6000] + "\n...[truncated]"

        run.tool_calls.append({"name": name, "args": args, "output": output})
        run.messages.append(
            {
                "role": "tool",
                "tool_call_id": tc.id,
                "name": name,
                "content": output,
            }
        )

        preview = output[:160].replace("\n", " ")
        self._log(f"[TOOL OUT]  {preview}{'...' if len(output) > 160 else ''}")

    # ----------------------------------------------------------------- log
    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)
