"""
Benchmark harness for the code-fix agent.

Runs all broken scripts in tests/broken_scripts through the agent,
collects structured metrics, saves JSON logs, prints a report.

Usage:
    python -m scripts.benchmark

Why separate from main.py: main.py is interactive (HITL prompts).
benchmark.py uses bypass_hitl=True, runs all scripts in sequence,
and captures structured output for analysis.

Metrics collected per run:
    - status: done / blocked / cost_exceeded / error
    - fixed: bool - status==done AND evaluator_score >= MIN_PASS_SCORE
    - iterations: how many execute cycles
    - evaluator_score: final LLM score (None if never evaluated)
    - cost_usd: API cost for this run
    - tokens: total tokens used
    - elapsed_s: wall clock time
    - error_type: first line of stderr from iteration 1 ( for categorization)

Aggregate metrics:
    - fix_rate, timeout_rate, error_rate
    - mean_iterations, mean_cost_usd, total_cost_usd
    - evaluator_rejection_rate
"""

import json
import uuid
from pathlib import Path
import time
from graph import build_graph
from src.code_agent.config import MIN_PASS_SCORE, DEEPSEEK_MODEL, PROMPT_VERSION

BROKEN_SCRIPTS_DIR = Path("tests/broken_scripts")
RESULTS_DIR = Path("outputs/benchmark_results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def run_single(script_path: Path, graph) -> dict:
    """
    Run one broken script through the full agent loop.

    bypass_hitl=True: node_human_approval auto-approves every patch.
    expected_output=None: evaluator infers correctness from code intent.
      For known-output scripts, set this to the expected stdout string.

    Returns a flat dict of metrics for this run. Catches all exceptions
    so one failing script doesn't abort the benchmark.
    """
    code = script_path.read_text()
    run_id = str(uuid.uuid4())[:8]

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
        "run_id": run_id,
        "bypass_hitl": True,
    }

    config = {
        "configurable": {"thread_id": run_id},
        "run_name": f"benchmark_{script_path.stem}_{run_id}",
    }

    start = time.time()
    try:
        final_state = graph.invoke(initial_state, config)
    except Exception as e:
        return {
            "script": script_path.name,
            "status": "error",
            "fixed": False,
            "iterations": 0,
            "evaluator_score": None,
            "evaluator_feedback": None,
            "cost_usd": 0.0,
            "tokens": 0,
            "elapsed_s": round(time.time() - start, 2),
            "error_type": f"benchmark harness error: {e}",
            "run_id": run_id,
        }
    elapsed = round(time.time() - start, 2)

    # Extract error type from first execution for categorization
    history = final_state.get("attempt_history", [])
    first_error = history[0]["error"] if history else final_state.get("error", "")
    error_type = first_error.strip().splitlines()[-1] if first_error else "none"

    fixed = (
        final_state["status"] == "done"
        and (final_state.get("evaluator_score") or 0) >= MIN_PASS_SCORE
    )

    return {
        "script": script_path.name,
        "status": final_state["status"],
        "fixed": fixed,
        "iterations": final_state["iterations"],
        "evaluator_score": final_state.get("evaluator_score"),
        "evaluator_feedback": (final_state.get("evaluator_feedback") or "")[:200],
        "cost_usd": round(final_state["total_cost_usd"], 6),
        "tokens": final_state["total_tokens"],
        "elapsed_s": elapsed,
        "error_type": error_type[:120],
        "run_id": run_id,
    }


def print_report(results: list[dict]):
    """
    Print per-script table and aggregate metrics summary.
    """
    print("\n" + "=" * 90)
    print(f"{'Script':<28} {'Status':<12} {'Iter':>4} {'Score':>6} {'Cost':>9} {'Time':>7}")
    print("-" * 90)
    for r in results:
        score_str = f"{r['evaluator_score']}/10" if r["evaluator_score"] is not None else "  N/A"
        mark = "✓" if r["fixed"] else "✗"
        print(
            f"{mark} {r['script']:<26} "
            f"{r['status']:<12} "
            f"{r['iterations']:>4} "
            f"{score_str:>6} "
            f"${r['cost_usd']:.4f}   "
            f"{r['elapsed_s']:>6.1f}s"
        )
    print("=" * 90)

    n = len(results)
    fixed = [r for r in results if r["fixed"]]
    blocked = [r for r in results if r["status"] == "blocked"]
    errored = [r for r in results if r["status"] == "error"]
    cost_exceeded = [r for r in results if r["status"] == "cost_exceeded"]

    mean_iter = sum(r["iterations"] for r in results) / n
    mean_cost = sum(r["cost_usd"] for r in results) / n
    total_cost = sum(r["cost_usd"] for r in results)

    print(f"\nBenchmark Summary  ({n} scripts | model={DEEPSEEK_MODEL} | prompt={PROMPT_VERSION})")
    print(f"  Fix rate           : {len(fixed) / n:.0%}  ({len(fixed)}/{n})")
    print(f"  Blocked rate       : {len(blocked) / n:.0%}  ({len(blocked)}/{n})")
    print(f"  Error rate         : {len(errored) / n:.0%}  ({len(errored)}/{n})")
    print(f"  Cost exceeded      : {len(cost_exceeded) / n:.0%}  ({len(cost_exceeded)}/{n})")
    print(f"  Mean iterations    : {mean_iter:.1f}")
    print(f"  Mean cost per run  : ${mean_cost:.4f}")
    print(f"  Total cost         : ${total_cost:.4f}")

    if blocked or errored:
        print(f"\nFailed scripts:")
        for r in blocked + errored:
            print(f"  {r['script']}: {r['error_type'][:80]}")
    print()


def main():
    scripts = sorted(BROKEN_SCRIPTS_DIR.glob("*.py"))
    if not scripts:
        print(f"No scripts found in {BROKEN_SCRIPTS_DIR}")
        return

    print(f"\nCode-Fix Agent Benchmark")
    print(f"Model: {DEEPSEEK_MODEL} | Prompt: {PROMPT_VERSION}")
    print(f"Running {len(scripts)} scripts...\n")

    graph = build_graph()
    results = []

    for script_path in scripts:
        print(f"  [{script_path.stem:<25}]", end=" ", flush=True)
        result = run_single(script_path, graph)
        results.append(result)
        mark = "✓" if result["fixed"] else "✗"
        print(f"{mark}  iter={result['iterations']}  score={result['evaluator_score']}  ${result['cost_usd']:.4f}")

    print_report(results)

    # Save structured JSON - one file per benchmark run
    timestamp = int(time.time())
    output_path = RESULTS_DIR / f"benchmark_{timestamp}.json"
    output_path.write_text(json.dumps({
        "meta": {
            "model": DEEPSEEK_MODEL,
            "prompt_version": PROMPT_VERSION,
            "timestamp": timestamp,
            "n_scripts": len(results),
        },
        "results": results,
    }, indent=2))
    print(f"Results saved → {output_path}\n")


if __name__ == "__main__":
    main()








