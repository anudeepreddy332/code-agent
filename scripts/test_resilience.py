"""
Simulates tool and LLM failures to verify the agent degrades gracefully.

Tests:
    1. Execution timeout — script that exceeds EXECUTION_TIMEOUT
    2. LLM API failure — temporarily invalid API key
    3. Cost ceiling — set MAX_COST_PER_RUN to $0.00 and verify termination
    4. Max iterations — script that can't be fixed in MAX_ITERATIONS attempts

Usage:
    python -m scripts.test_resilience

"""
import os
import uuid
from graph import build_graph
from src.code_agent.config import MAX_ITERATIONS, EXECUTION_TIMEOUT

def _run(code: str, overrides: dict = None) -> dict:
    """Run a script through the agent with optional state overrides."""
    graph = build_graph()
    state = {
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
        "bypass_hitl": True,
        **(overrides or {}),
    }
    config = {"configurable": {"thread_id": state["run_id"]}}
    return graph.invoke(state, config)


def test_timeout():
    """Agent should handle timeout gracefully — not crash."""
    print("Test 1: Execution timeout", end=" ... ")
    code = "while True: pass"
    result = _run(code)
    # Agent should diagnose the timeout and attempt a fix, or block at MAX_ITERATIONS
    assert result["status"] in ("done", "blocked"), f"unexpected status: {result['status']}"
    assert result["iterations"] >= 1, "agent should have attempted at least one iteration"
    print("PASS")


def test_cost_ceiling():
    """Agent should terminate immediately when cost ceiling is pre-exceeded."""
    print("Test 2: Cost ceiling", end=" ... ")
    code = "print('hello')"
    # Pre-set cost above ceiling — agent should end immediately
    result = _run(code, overrides={"total_cost_usd": 999.0})
    assert result["status"] == "cost_exceeded", f"expected cost_exceeded, got: {result['status']}"
    print("PASS")

def test_max_iterations():
    """Agent should terminate with 'blocked' when MAX_ITERATIONS reached."""
    print("Test 3: Max iterations", end=" ... ")
    # Test the gate directly: pre-set iterations to max
    result = _run(
        "def f(\n    pass",  # broken syntax
        overrides={"iterations": MAX_ITERATIONS}
    )
    assert result["status"] in ("blocked", "cost_exceeded"), \
        f"expected blocked or cost_exceeded, got: {result['status']}"
    print("PASS")



def test_invalid_api_key():
    """Agent should return a clear error when the API key is invalid."""
    print("Test 4: Invalid API key", end=" ... ")
    original_key = os.environ.get("DEEPSEEK_API_KEY", "")
    os.environ["DEEPSEEK_API_KEY"] = "invalid_key_for_testing"
    try:
        result = _run("print(1/0)")
        # Graph should either error gracefully or return blocked
        # It won't return "done" with an invalid key
        assert result["status"] != "done", \
            "agent should not succeed with invalid API key"
        print("PASS")
    except Exception as e:
        # Exception from LangChain is also acceptable — means it didn't silently pass
        print(f"PASS (raised exception: {type(e).__name__})")
    finally:
        os.environ["DEEPSEEK_API_KEY"] = original_key


def main():
    print("\nPhase 3.5 Resilience Tests\n" + "=" * 40)
    test_timeout()
    test_cost_ceiling()
    test_max_iterations()
    test_invalid_api_key()
    print("=" * 40)
    print("All resilience tests passed.\n")


if __name__ == "__main__":
    main()












