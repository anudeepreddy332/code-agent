"""
Code-fix agent entry point.

Reads a broken Python file, runs it through the LangGraph fix loop,
handles human approval interrupts, prints a structured report.

Usage:
    python -m main tests/broken_scripts/type_error.py

The graph pauses at node_human_approval and waits for y/n input.
On approval, it resumes from exactly where it paused.
On rejection, the agent re-diagnoses using the rejection reason.

Pass --auto to bypass human approval (non-interactive mode):
    python -m main tests/broken_scripts/logic_error.py --auto
"""

import sys
import uuid
import time
from pathlib import Path
from langgraph.types import Command
from graph import build_graph
from src.code_agent.config import PROMPT_VERSION, DEEPSEEK_MODEL
import os
from datetime import datetime as dt

def main():
    if len(sys.argv) < 2:
        print("Usage: python -m main path/to/script.py [--auto]")
        sys.exit(1)

    script_path = sys.argv[1]
    auto_mode = "--auto" in sys.argv

    try:
        code = Path(script_path).read_text()
    except FileNotFoundError:
        print(f"File not found: {script_path}")
        sys.exit(1)

    graph = build_graph()
    script_name = Path(script_path).stem
    run_name = f"{script_name}_{dt.now().strftime('%H%M%S')}"

    initial_state = {
        "code": code,
        "original_code": code,
        "error": None,
        "stdout": "",
        "exit_code": -1,
        "iterations": 0,
        "total_cost_usd": 0.0,
        "total_tokens": 0,
        "status": "running",
        "diagnosis": None,
        "patch_explanation": None,
        "evaluator_score": None,
        "evaluator_feedback": None,
        "expected_output": None,
        "human_approval": None,
        "rejection_reason": None,
        "attempt_history": [],
        "run_id": str(uuid.uuid4())[:8],
        "bypass_hitl": auto_mode,
    }

    thread_id = initial_state["run_id"]
    config = {
        "configurable": {"thread_id": thread_id},
        "run_name": run_name,
    }

    os.environ["LANGCHAIN_TAGS"] = f"phase3, {script_name}"

    print(f"\nCode-Fix Agent | model={DEEPSEEK_MODEL} | prompt={PROMPT_VERSION}")
    print(f"Script: {script_path}")

    if auto_mode:
        print("Mode: auto (HITL bypassed)")

    print("=" * 60)

    start = time.time()
    current_input = initial_state

    while True:
        result = graph.invoke(current_input, config)

        # Detect interrupt from node_human_approval
        interrupt_data = result.get("__interrupt__")
        if interrupt_data:
            payload = interrupt_data[0].value if interrupt_data else {}
            diff = payload.get("diff", "(no diff available)")
            iteration = payload.get("iteration", "?")
            eval_score = payload.get("evaluator_score")

            print(f"\n{'=' * 60}")
            print(f"PATCH READY — Iteration {iteration}")
            if eval_score is not None:
                print(f"Pre-evaluation score: {eval_score}/10")
            print("\nDiff (current → patched):\n")
            print(diff.strip() if diff.strip() else "(no visible changes)")
            print(f"\n{'=' * 60}")

            approval = input("Apply this patch? [y/n]: ").strip().lower()
            reason = ""
            if approval != "y":
                reason = input("Rejection reason (helps the agent): ").strip()

            # Resume the graph — Command(resume=...) passes data back to interrupt()
            current_input = Command(resume={"approved": approval, "reason": reason})

        else:
            final_state = result
            break

    elapsed = time.time() - start

    print("\n" + "=" * 60)
    print(f"STATUS        : {final_state['status']}")
    print(f"ITERATIONS    : {final_state['iterations']}")
    print(f"TOTAL TOKENS  : {final_state['total_tokens']}")
    print(f"TOTAL COST    : ${final_state['total_cost_usd']:.4f}")
    print(f"ELAPSED       : {elapsed:.1f}s")

    if final_state.get("evaluator_score") is not None:
        print(f"EVAL SCORE    : {final_state['evaluator_score']}/10")

    if final_state.get("evaluator_feedback"):
        print(f"EVAL FEEDBACK : {final_state['evaluator_feedback'][:150]}")

    if final_state["status"] == "done":
        print("\nFIXED CODE:")
        print(final_state["code"])


if __name__ == "__main__":
    main()