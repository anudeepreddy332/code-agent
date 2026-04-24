# Code Agent – Self-Correcting Python Fixer

[![Phase 3](https://img.shields.io/badge/Phase-3-blue)](https://github.com/anudeepreddy332)
[![LangGraph](https://img.shields.io/badge/Framework-LangGraph-orange)](https://github.com/langchain-ai/langgraph)
[![DeepSeek](https://img.shields.io/badge/LLM-DeepSeek-green)](https://deepseek.com)

An agent that takes a broken Python script, executes it in a sandbox, reads the traceback, diagnoses the error, patches the code, and re-runs—up to 5 iterations.

Previous phases: [CLI Research Agent](https://github.com/anudeepreddy332/cli-research-agent) | [Knowledge Agent (RAG)](https://github.com/anudeepreddy332/knowledge-agent)

---

## Why This Exists

Phase 2 gave the agent memory and retrieval. Phase 3 teaches it **self-correction**.  
Instead of "question → answer," the loop becomes **execute → evaluate → retry** until the goal is met or a cost ceiling is hit.

Code fixing is the sharpest use case: failure is unambiguous (the script either runs or it doesn't).

---

## Architecture

    User provides broken script  
        ↓  
    [LangGraph State Machine]  
        ↓  
    1. Execute (sandbox subprocess)  
    2. Diagnose (LLM reads error + code)  
    3. Patch (LLM rewrites code)  
    4. Repeat (max 5 iterations)  
    5. Done (success or give up)  

LangGraph replaces the raw while loop from earlier phases. Every state is typed, every transition is explicit, and the graph is inspectable.

---

## Tech Stack

| Component       | Tool |
|-----------------|------|
| LLM             | DeepSeek (deepseek-chat) |
| Agent Framework | LangGraph |
| Observability   | LangSmith (free tier) |
| Package Manager | uv |
| Code Execution  | subprocess with hard timeout |

---

## Project Structure

    code-agent/  
    ├── graph.py           # LangGraph state machine definition  
    ├── nodes.py           # Node functions (execute, diagnose, patch, evaluate)  
    ├── tools.py           # Sandboxed code execution tool  
    ├── main.py            # Entry point / CLI  
    ├── src/code_agent/  
    │   └── config.py      # Constants, cost limits, prompt version  
    └── tests/  
        └── broken_scripts/  # Sample broken scripts for testing  

---

## Getting Started

1. Clone the repo  
```
git clone git@github.com:anudeepreddy332/code-agent.git  
cd code-agent
```

2. Install dependencies with uv  
```
uv sync  
```

3. Set environment variables (.env)  
DEEPSEEK_API_KEY=your_key  
LANGSMITH_API_KEY=your_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=code-agent
4. 
4. Run the agent on a broken script  
```
python main.py --file tests/broken_scripts/syntax_error.py  
```
---

## Progress & Roadmap

- [x] Day 15: Rebuild Phase 1 agent in LangGraph (familiarization)
- [x] Day 16: Sandboxed code execution tool
- [x] Day 17: Reflexion loop (diagnose + patch)
- [x] Day 18: Evaluator node (1–10 score)
- [x] Day 19: LangSmith tracing
- [ ] Day 20: Human-in-the-loop checkpoint
- [ ] Day 21: Benchmarking (fix rate, cost per fix)


---

## Author

Anudeep – Building agentic systems from scratch, one phase at a time.  
LinkedIn: https://linkedin.com/in/anudeep-reddy-mutyala  
GitHub: https://github.com/anudeepreddy332
Portfolio: themachinist.org