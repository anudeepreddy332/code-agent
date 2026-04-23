"""
Code-fix agent entry point.

Reads a broken Python filem runs it through the LangGraph fix loop,
prints a structured report.

Usage:
    python -m main path/to/broken_script.py

The graph runs until the code executes successfully, cost is exceeded,
or max iterations are reached. Final state is printed as a report.
"""
import sys
import uuid
import time
from graph import build_graph
from src.code_agent.config import PROMPT_VERSION, DEEPSEEK_MODEL

def main():
    if len(sys.argv) < 2:
        print("Usage: python -m main path/to/script.py")
        sys.exit(1)

    script_path = sys.argv[1]
    try:
        code = open(script_path).read()
    except FileNotFoundError:
        print(f"File not found: {script_path}")
        sys.exit(1)

    graph = build_graph()

    initial_state = {
        "code": code,
        "original_code": code,
        "error": None,
        "stdout": "",
        "exit_code": -1,
        "iterations": 0,
        "total_cost_usd": 0,
        "total_tokens": 0,
        "status": "running",
        "diagnosis": None,
        "patch_explanation": None,
        "evaluator_score": None,
        "run_id": str(uuid.uuid4())[:8],
    }
    print(f"\nCode-Fix Agent | model={DEEPSEEK_MODEL} | prompt={PROMPT_VERSION}")
    print(f"Script: {script_path}")
    print("=" * 60)

    start = time.time()
    final_state = graph.invoke(initial_state)
    elapsed = time.time() - start

    print("\n" + "=" * 60)
    print(f"STATUS        : {final_state['status']}")
    print(f"ITERATIONS    : {final_state['iterations']}")
    print(f"TOTAL TOKENS  : {final_state['total_tokens']}")
    print(f"TOTAL COST    : ${final_state['total_cost_usd']:.4f}")
    print(f"ELAPSED       : {elapsed:.1f}s")
    if final_state["status"] == "done":
        print("\nFIXED CODE:")
        print(final_state["code"])


if __name__ == "__main__":
    main()