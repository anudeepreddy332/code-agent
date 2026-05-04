# Code Agent – Self‑Correcting Python Fixer

[![Phase 3 Complete](https://img.shields.io/badge/Phase-3%20Complete-brightgreen)](https://github.com/anudeepreddy332)
[![LangGraph](https://img.shields.io/badge/Framework-LangGraph-orange)](https://github.com/langchain-ai/langgraph)
[![DeepSeek](https://img.shields.io/badge/LLM-DeepSeek-green)](https://deepseek.com)
[![Benchmark 95%](https://img.shields.io/badge/Benchmark-95%25-success)]()

An agent that takes a broken Python script, executes it in a sandbox, reads the traceback,
diagnoses the error, proposes a patch, asks for human approval (optional), and re‑runs –
up to five iterations.

It **doesn’t just run code** – it reflects on why fixes fail, evaluates its own output,
and degrades gracefully when things go wrong.

---

## Why this exists

Phase 2 gave the agent memory and hybrid RAG.
Phase 3 teaches it **self‑correction**.

The loop is no longer “question in, answer out”.
It’s **execute → evaluate → retry** until the code works *correctly* or a cost ceiling is hit.

Code fixing is the sharpest use‑case: failure is unambiguous (the script either runs or it doesn’t).

---

## Architecture

```
User provides broken script
        │
        ▼
┌─────────────────────────────────┐
│        LangGraph State Machine  │
│                                 │
│  execute ──► should_continue?   │
│               │                 │
│               ├─ diagnose       │
│               ├─ evaluate       │
│               └─ end            │
│                                 │
│  diagnose → patch → human_approval → execute (loop)
│                                 │
│  evaluate → should_evaluate_continue?
│               ├─ diagnose (score < 7)
│               └─ end (score ≥ 7)
└─────────────────────────────────┘
```

Every transition is an explicit, typed edge.  The whole state is visible and checkpointed.

---

## Key Features

- **LangGraph state machine** – No more implicit `while` loops.  Every step is a node, every branch is a conditional edge.
- **Reflexion loop** – If a fix fails, the agent sees its previous attempts and tries a different strategy.
- **Human‑in‑the‑loop checkpoint** – Before applying a patch, the agent pauses and shows a diff.  You approve, reject (with a reason that feeds back into diagnosis), or bypass with `--auto`.
- **LLM Evaluator** – After a successful run, another LLM call scores the fix (1‑10).  Only scores ≥ 7 pass; otherwise the agent retries with feedback.
- **Cost & token tracking** – Every LLM call is logged, accumulated, and gated.  The agent will stop if the cost ceiling is hit.
- **Resilience testing** – Automated tests for timeouts, cost‑ceiling, max‑iterations, and even invalid API keys.
- **Benchmark harness** – Runs 20 broken scripts, reports fix rate, iterations, cost, and logs everything.

---

## Benchmark Results (20 scripts)

| Metric            | Value       |
|-------------------|-------------|
| Fix rate          | 95% (19/20) |
| Blocked / Error   | 0%          |
| Mean iterations   | 2.4         |
| Mean cost per run | $0.0006     |
| Total cost (20)   | $0.0128     |
| Mean evaluator score  | 9.7 / 10    |
The only script that didn’t pass (`import_error.py`) requires NumPy, which isn’t installed in the sandbox.  That’s an environment limitation, not an agent failure.

---

## Project Structure

```
code-agent/
├── graph.py                  # LangGraph wiring (AgentState, build_graph)
├── nodes.py                  # Node & edge functions (execute, diagnose, patch, evaluate, HITL)
├── tools.py                  # Sandboxed code execution (subprocess + timeout)
├── main.py                   # CLI entry point, HITL interrupt loop, run logging
├── src/code_agent/
│   └── config.py             # Keys, model, cost limits, prompt version
├── scripts/
│   ├── benchmark.py          # Full batch evaluation (20 scripts)
│   ├── analyze_failures.py   # Failure extraction & regression stubs
│   ├── regression_check.py   # Baseline vs current comparison
│   └── test_resilience.py    # Timeout, cost ceiling, max iter, bad API key tests
├── tests/
│   └── broken_scripts/       # 20 deliberately broken Python files
├── outputs/
│   ├── benchmark_results/    # JSON per benchmark run
│   ├── run_logs/             # Per‑invocation structured logs
│   └── langsmith_trace_review.md
└── pyproject.toml
```

---

## Getting Started

### 1. Clone and install
```bash
git clone git@github.com:anudeepreddy332/code-agent.git
cd code-agent
uv sync
```

### 2. Environment variables
Create a `.env` file with:
```
DEEPSEEK_API_KEY=your_deepseek_key
LANGSMITH_API_KEY=your_langsmith_key   # optional – enables tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=code-agent
```

### 3. Run the agent on a broken script
```bash
# Interactive mode (will ask for patch approval)
python -m main tests/broken_scripts/attribute_error.py

# Fully automatic (bypasses human approval)
python -m main tests/broken_scripts/attribute_error.py --auto
```

### 4. Run the full benchmark
```bash
python -m scripts.benchmark
```
This will execute all 20 scripts, print a summary, and save results to `outputs/benchmark_results/`.

### 5. Check for regressions
After a benchmark run, use:
```bash
python -m scripts.analyze_failures
python -m scripts.regression_check
```

### 6. Resilience tests
```bash
python -m scripts.test_resilience
```

All resilience scenarios validated: timeouts, cost ceiling breaches, max iteration limits, and invalid API keys — all handled gracefully with no system crashes.

---

## Tech Stack

| Layer         | Tool                           |
|---------------|--------------------------------|
| LLM           | DeepSeek (`deepseek-chat`)     |
| Agent engine  | LangGraph                      |
| Observability | LangSmith                      |
| Package mgmt | `uv`                           |
| Code execution| `subprocess` with hard timeout |

---

## Author
**Anudeep** – Building agentic systems from scratch.  
[LinkedIn](https://linkedin.com/in/anudeep-reddy-mutyala) | [GitHub](https://github.com/anudeepreddy332) | [Portfolio](https://themachinist.org)