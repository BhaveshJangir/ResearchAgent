# Smart Research Agent — a learning project

A tiny but real **agentic AI** application in ~250 lines of Python.

You give it a question. It plans, searches the web, reads pages, optionally does math, and returns a cited answer. The point isn't the answer — it's to see *how the agent thinks*. Run it with `verbose=True` and you'll watch every reasoning step.

## What makes something "agentic"?

A regular LLM call is one-shot: prompt in, text out. An **agent** is an LLM running inside a loop where it can:

1. **Reason** about what to do next.
2. **Act** by calling tools (functions you expose to it).
3. **Observe** the tool's output.
4. Decide whether to keep going or finish.

That four-step cycle is called **ReAct** (Reason + Act). Almost every "AI agent" you've heard of — AutoGPT, LangChain agents, Cursor's editor agent, deep-research products — is a variation on this loop. Strip the marketing and you're left with: *LLM in a while-loop with function calling*.

## Architecture

```
        ┌────────────────────┐
user ──▶│  ResearchAgent     │
        │                    │
        │   ┌────────────┐   │      ┌──────────────────┐
        │   │  LLM call  │◀──┼──────│ system + history │
        │   └─────┬──────┘   │      └──────────────────┘
        │         │          │
        │   tool_calls?      │
        │     /     \        │
        │   yes      no ─────┼─▶ final answer
        │    │               │
        │    ▼               │
        │  ┌──────────────┐  │
        │  │ run tool(s)  │──┼──▶ web_search / fetch_url / calculator
        │  └──────┬───────┘  │
        │         │          │
        │  append result,    │
        │  loop back ────────┘
        └────────────────────┘
```

## Files

| File | What it does |
|------|--------------|
| `tools.py` | Defines the three tools (`web_search`, `fetch_url`, `calculator`) and their JSON schemas — the "API surface" the LLM sees. |
| `agent.py` | The ReAct loop. Talks to OpenAI, dispatches tool calls, collects the final answer. |
| `main.py`  | CLI entry point. `python main.py "your question"`. |
| `requirements.txt` | Python dependencies. |
| `.env.example` | Copy to `.env` and add your `OPENAI_API_KEY`. |

## Run it

```bash
# 1. install
pip install -r requirements.txt

# 2. configure
cp .env.example .env
# edit .env and paste your OpenAI API key

# 3. ask
python main.py "How much electricity does training GPT-4 use, roughly?"
```

You'll see something like:

```
[USER] How much electricity does training GPT-4 use, roughly?
--- iteration 1 ---
[TOOL CALL] web_search({'query': 'GPT-4 training electricity consumption estimate'})
[TOOL OUT]  [{"title": "...", "url": "...", ...}, ...]
--- iteration 2 ---
[TOOL CALL] fetch_url({'url': 'https://...'})
[TOOL OUT]  Researchers estimate that training GPT-4 required ...
--- iteration 3 ---
[TOOL CALL] calculator({'expression': '50_000_000 / 11000'})
[TOOL OUT]  4545.45...
--- iteration 4 ---
[FINAL]
GPT-4's training is estimated at roughly 50 GWh ... [1][2]
Sources:
[1] https://...
[2] https://...
```

## Concepts to notice while reading the code

### 1. Tools = (function, schema)

Every tool has two halves: the Python function that runs, and a JSON schema in `TOOL_SCHEMAS` that tells the model **what the tool is for and how to call it**. The model never sees your code — it only sees the schema. Sharp descriptions = better tool selection.

### 2. The loop is dumb on purpose

Look at `ResearchAgent.run()`. It's a `for` loop with an `if msg.tool_calls`. There's no clever state machine. *All* the intelligence is in the prompt + the model's choice of what tool to call next. This is the whole trick of modern agents: push intelligence into the model and keep your scaffolding boring.

### 3. The system prompt is the agent's "policy"

Read `SYSTEM_PROMPT` carefully. Lines like *"Cross-check claims across at least TWO independent sources"* and *"Don't fetch the same URL twice"* aren't decoration — they're behavioral rules. Tweaking the system prompt is usually the highest-leverage way to change agent behavior, ahead of changing the code.

### 4. Memory = the message list

We don't have a separate "memory" module. The agent remembers the run by appending every assistant message and every tool result onto `run.messages`, which we re-send each iteration. That's it. Long-running agents add summarization or a vector store on top, but conceptually it's still "the conversation so far."

### 5. Guards: iteration cap, output truncation, safe `eval`

Real agents fail in real ways. `max_iterations` stops infinite loops. Truncating tool output protects the context window. Restricting `eval`'s globals prevents the LLM from running arbitrary code via `calculator`. Treat the LLM as untrusted input.

## Things to try next (great learning exercises)

1. **Add a new tool.** A `wikipedia_summary(title)` tool, or a `today()` tool that returns the current date. Add it to `TOOL_FUNCTIONS` and `TOOL_SCHEMAS` — that's it.
2. **Switch models.** Try `gpt-4o` vs `gpt-4o-mini`. Watch reasoning quality vs cost.
3. **Add a planning step.** Before the loop, make one extra LLM call asking *"Write a 3-step plan, no tools."* Append it as an assistant message. See if quality improves.
4. **Add reflection.** After the final answer, run a critic prompt that scores the answer; if it scores low, loop again.
5. **Switch to multi-agent.** Make a `Researcher` agent and a `Critic` agent that exchange messages. Now you have a multi-agent system.
6. **Persist runs.** Save `AgentRun` to JSON so you can replay or compare runs.

## Common gotchas

- **`tool_calls` must be echoed back verbatim.** When you append the assistant message, you must include the original `tool_calls` array. The follow-up `tool` messages reference the `tool_call_id`s; if they don't match, the API errors.
- **The model can call multiple tools in one turn.** Loop over `msg.tool_calls`, don't assume there's only one.
- **Searches can return junk.** That's why we instruct the model to cross-check and refine queries — it's not optional.

That's the whole project. Read `agent.py` end-to-end once, then `tools.py`. After that you can build basically any agent: it's the same skeleton with different tools and a different system prompt.
