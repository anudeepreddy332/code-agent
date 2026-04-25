"""
Node and edge functions for the code-fix agent LangGraph state machine.

Each node is a pure function: takes AgentState, returns a dict of fields to update.
LangGraph merges the returned dict into the current state automatically.

Each edge function returns a string that LangGraph maps to the next node name.

Imports state schema from graph.py to avoid circular imports — AgentState
is defined there because build_graph() needs it at compile time.

Run: imported by graph.py. No standalone execution.
"""

import json
import re
import difflib
from typing import Literal

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.types import interrupt

from src.code_agent.config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    COST_PER_1M_INPUT_TOKENS,
    COST_PER_1M_OUTPUT_TOKENS,
    MAX_COST_PER_RUN,
    MAX_ITERATIONS,
    MIN_PASS_SCORE,
)
from tools import execute_python

def _make_client() -> ChatOpenAI:
    """
    Build LangChain ChatOpenAI client pointed at DeepSeek.

    Why ChatOpenAI and not raw openai client: LangGraph nodes work
    naturally with LangChain message objects. DeepSeek's OpenAI-compatible
    API works by overriding base_url and api_key.

    Client is created per-call in each node — slightly wasteful but
    avoids shared mutable state across nodes.
    """
    return ChatOpenAI(
        model=DEEPSEEK_MODEL,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        temperature=0.3,
    )

def _track_cost(state: dict, response) -> dict:
    """
    Extract token usage from LangChain response, return accumulated totals.

    Returns new running totals (not deltas) — LangGraph state merge
    replaces fields, not adds to them, so we must return the full new value.

    Why track here and not in the client: LangGraph state is the single source of truth.
    Centralizing cost tracking in state means the cost gate in should_continue() sees
    the real accumulated cost, not an estimate.
    """

    usage = response.usage_metadata
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    cost = (
        (input_tokens / 1e6 * COST_PER_1M_INPUT_TOKENS)
        + (output_tokens / 1e6 * COST_PER_1M_OUTPUT_TOKENS)
    )
    return {
        "total_cost_usd": state["total_cost_usd"] + cost,
        "total_tokens": state["total_tokens"] + input_tokens + output_tokens,
    }


def _show_diff(original: str, patched: str) -> str:
    """
    Produce a unified diff between original and patched code.
    Same format as git diff - added lines start with _, removed with -.
    """
    diff = difflib.unified_diff(
        original.splitlines(keepends=True),
        patched.splitlines(keepends=True),
        fromfile="current_code",
        tofile="patched_code",
        lineterm="",
    )
    return "".join(diff)



# -----------------------------------------------------------------------
# Node 1: execute
# -----------------------------------------------------------------------

def node_execute(state: dict) -> dict:
    """
    Execute current code in subprocess. Archive previous attempt before
    overwriting error state.

    Why archive before overwriting: after this node runs, state fields
    (error, exit_code, stdout) are replaced with new execution results.
    node_diagnose needs the full history of what was tried — if we don't
    save it here, it's gone.

    Archival only happens when a prior diagnosis exists (skips first run).

    Vulnerability: attempt_history grows with each iteration. We cap
    history injection at 3 entries in node_diagnose — the list itself
    can grow longer without harm.
    """
    print(f"\n[execute] iteration {state['iterations'] + 1}")
    result = execute_python(state["code"])
    print(f"  exit_code={result['exit_code']} | timed_out={result['timed_out']}")
    if result["stderr"]:
        print(f"  stderr: {result['stderr'][:200]}")

    new_history = list(state.get("attempt_history", []))
    if state.get("diagnosis") is not None:
        new_history.append({
            "code": state["code"],
            "error": state.get("error", ""),
            "diagnosis": state.get("diagnosis", ""),
            "patch_explanation": state.get("patch_explanation", ""),
        })

    return {
        "stdout": result["stdout"],
        "error": result["stderr"] if not result["success"] else None,
        "exit_code": result["exit_code"],
        "iterations": state["iterations"] + 1,
        "attempt_history": new_history,
    }


# -----------------------------------------------------------------------
# Node 2: diagnose
# -----------------------------------------------------------------------

def node_diagnose(state: dict) -> dict:
    """
    Reflexion-enhanced diagnose node.

    Iteration 1: standard diagnosis — what's wrong?
    Iteration 2+: reflexion — what was tried, why did it fail,
    what's a different strategy?

    Also injects evaluator feedback (if evaluator rejected a fix)
    and human rejection reason (if human rejected a patch).

    Why cap history at 3: older attempts are less relevant and long
    history pushes the prompt toward the context limit.
    """
    client = _make_client()

    history = state.get("attempt_history", [])
    is_reflexion = len(history) > 0

    # Build evaluator feedback context
    eval_feedback = state.get("evaluator_feedback", "")
    eval_score = state.get("evaluator_score")
    evaluator_context = ""
    if eval_score is not None and eval_score < MIN_PASS_SCORE:
        evaluator_context = (
            f"\n\nIMPORTANT: A code evaluator reviewed your previous fix "
            f"and scored it {eval_score}/10 with this feedback:\n"
            f"{eval_feedback}\n"
            f"Your next diagnosis must address this evaluator feedback specifically."
        )

    # Build human rejection context
    rejection_context = ""
    if state.get("human_approval") == "rejected" and state.get("rejected_reason"):
        rejection_context = (
            f"\n\nA human reviewer rejected the previous patch with this reason:\n"
            f"{state['rejection_reason']}\n"
            f"Your next diagnosis must address this rejection reason specifically."
        )

    if not is_reflexion:
        system_content = (
            "You are a Python debugging expert. "
            "You will be given a Python script and its error output. "
            "Diagnose the root cause of the error in 2-3 sentences. "
            "Be specific — identify the exact line or construct causing the failure. "
            "Do not suggest fixes yet. Just diagnose."
        )
        user_content = (
            f"CODE:\n```python\n{state['code']}\n```\n\n"
            f"ERROR:\n{state['error']}"
            f"{evaluator_context}{rejection_context}"
        )
    else:
        # Reflexion — show previous attempts, ask for new strategy
        recent = history[-3:]
        attempts_str = ""
        for i, attempt in enumerate(recent, 1):
            attempts_str += (
                f"\n--- Attempt {i} ---\n"
                f"Code tried:\n```python\n{attempt['code']}\n```\n"
                f"Error produced:\n{attempt['error']}\n"
                f"Diagnosis made:\n{attempt['diagnosis']}\n"
                f"Patch applied:\n{attempt['patch_explanation'][:300]}\n"
            )

        system_content = (
            "You are a Python debugging expert running a reflexion loop. "
            "Previous fix attempts have failed. "
            "Reflect on why previous patches failed, then diagnose with a NEW strategy. "
            "Do not repeat a diagnosis that already failed. "
            "Be specific about what was wrong with the previous approach."
        )
        user_content = (
            f"ORIGINAL CODE:\n```python\n{state['original_code']}\n```\n\n"
            f"PREVIOUS FAILED ATTEMPTS:{attempts_str}\n"
            f"CURRENT CODE:\n```python\n{state['code']}\n```\n\n"
            f"CURRENT ERROR:\n{state['error']}\n\n"
            "What is wrong with the current approach, and what different strategy should be tried?"
            f"{evaluator_context}{rejection_context}"
        )

    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=user_content),
    ]

    response = client.invoke(messages)
    cost_update = _track_cost(state, response)
    label = "(reflexion)" if is_reflexion else ""
    print(f"[diagnose] {label} {response.content[:150]}")

    return {
        "diagnosis": response.content,
        **cost_update,
    }

# -----------------------------------------------------------------------
# Node 3: patch
# -----------------------------------------------------------------------

def node_patch(state: dict) -> dict:
    """
    Produce corrected code based on the diagnosis.

    Returns full corrected file, not a diff — LLMs are more reliable
    producing complete files than patch hunks for small scripts.

    Parsing: strips ```python fences. If parsing fails, uses raw
    response and lets the next execute() surface the error (fail forward).

    Vulnerability: LLM may wrap code in prose. Guard strips everything
    outside the first ```python...``` block.
    """
    client = _make_client()
    messages = [
        SystemMessage(content=(
            "You are a Python debugging expert. "
            "You will be given a Python script, its error, and a diagnosis. "
            "Return ONLY the corrected Python code inside a ```python block. "
            "No explanation before or after the code block. "
            "Fix only what is broken — do not rewrite unrelated parts."
        )),
        HumanMessage(content=(
            f"CODE:\n```python\n{state['code']}\n```\n\n"
            f"ERROR:\n{state['error']}\n\n"
            f"DIAGNOSIS:\n{state['diagnosis']}"
        )),
    ]

    response = client.invoke(messages)
    cost_update = _track_cost(state, response)

    raw = response.content
    if "```python" in raw:
        patched = raw.split("```python")[1].split("```")[0].strip()
    elif "```" in raw:
        patched = raw.split("```")[1].split("```")[0].strip()
    else:
        patched = raw.strip()

    print(f"[patch] code updated ({len(patched)} chars)")

    return {
        "code": patched,
        "patch_explanation": response.content,
        **cost_update,
    }


def node_human_approval(state: dict) -> dict:
    """
    Interrupt the graph and wait for human approval of the patch.

    bypass_hit1=True skips the interrupt - used by benchmark.py for
    non-interactive runs. Without bypass, the graph suspends here via
    interrupt() and resumes when main.py calls graph.invoke(Command(resume=...))

    How LangGraph interrupt works:
        1. Interrupt (value) suspends the graph, returning the value to the caller
            as part of the interrupted state.
        2. main.py detects __interrupt__ in the result, displays the diff,
            gets human input, then calls graph.invoke(Command(resume=decision)).
        3. interrupt() returns decision here and execution continues.

    Vulnerability: no cap on rejection count. A human can reject
    indefinitely. Production fix: add rejection_count to state and
    end as "blocked" after N rejections.
    """

    if state.get("bypass_hit1"):
        print("[human] Auto-approved (benchmark mode)")
        return {"human_approval": "approved", "rejection_reason": None}

    diff_str = _show_diff(state["original_code"], state["code"])

    decision = interrupt({
        "message": "Review the proposed patch and approve or reject.",
        "diff": diff_str,
        "iteration": state["iterations"],
        "evaluator_score": state.get("evaluator_score")
    })

    approved = str(decision.get("approved", "")).lower() == "y"
    reason = str(decision.get("reason", ""))

    if approved:
        print("[human] Patch approved.")
        return {"human_approval": "approved", "rejection_reason": None}
    else:
        print(f"[human] Patch rejected: {reason}")
        return {"human_approval": "rejected", "rejection_reason": reason}



# -----------------------------------------------------------------------
# Node 5: evaluate
# -----------------------------------------------------------------------

def node_evaluate(state: dict) -> dict:
    """
    LLM evaluator: scores the fixed code's correctness on a 1-10 scale.

    Catches logic errors that produce wrong output despite exit_code==0.
    Score < MIN_PASS_SCORE sends the agent back to diagnose with feedback.

    Score guide given to LLM:
        9-10: correct output, clean minimal fix
        7-8:  correct output, minor style issues
        5-6:  partially correct, edge cases missed
        3-4:  wrong output but no exception
        1-2:  fix made things worse

    Vulnerability: LLM evaluator can misjudge correctness, especially
    for numerical outputs. expected_output field provides ground truth
    when the caller knows what correct stdout should look like.
    MIN_PASS_SCORE=7 is a tuning knob.

    Parse failure guard: if LLM doesn't return valid JSON, default to
    score=3 (conservative — don't pass something we can't verify).
    """

    client = _make_client()

    expected_hint = (
        f"\nExpected output (ground truth):\n{state['expected_output']}"
        if state.get("expected_output")
        else "\n(No expected output provided — infer correctness from code intent.)"
    )
    prior_feedback = (
        f"\nPrior evaluation feedback:\n{state['evaluator_feedback']}"
        if state.get("evaluator_feedback")
        else ""
    )

    messages = [
        SystemMessage(content=(
            "You are a code correctness evaluator. "
            "Score the fix from 1 to 10. "
            "Return ONLY a JSON object in this exact format, nothing else:\n"
            '{"score": <int 1-10>, "feedback": "<one paragraph explanation>"}\n'
            "Score guide:\n"
            "9-10: correct output, clean minimal fix\n"
            "7-8:  correct output, minor issues\n"
            "5-6:  partially correct, edge cases missed\n"
            "3-4:  wrong output but no exception\n"
            "1-2:  fix made things worse"
        )),
        HumanMessage(content=(
            f"ORIGINAL CODE:\n```python\n{state['original_code']}\n```\n\n"
            f"FIXED CODE:\n```python\n{state['code']}\n```\n\n"
            f"ACTUAL OUTPUT (stdout):\n{state['stdout'] or '(no output)'}\n"
            f"STDERR:\n{state['error'] or '(none)'}"
            f"{expected_hint}{prior_feedback}"
        )),
    ]

    response = client.invoke(messages)
    cost_update = _track_cost(state, response)

    raw = re.sub(r"```json|```", "", response.content.strip()).strip()
    try:
        parsed = json.loads(raw)
        score = int(parsed["score"])
        feedback = str(parsed["feedback"])
    except (json.JSONDecodeError, KeyError, ValueError):
        score = 3
        feedback = f"[evaluator parse error] Raw: {raw[:300]}"

    passed = score >= MIN_PASS_SCORE
    print(f"[evaluate] score={score}/10 | {'PASS' if passed else 'FAIL'} | {feedback[:100]}")

    return {
        "evaluator_score": score,
        "evaluator_feedback": feedback,
        "status": "done" if passed else "running",
        **cost_update,
    }


# -----------------------------------------------------------------------
# Node 6: end
# -----------------------------------------------------------------------

def node_end(state: dict) -> dict:
    """
    Set terminal status based on why we ended.
    Called when cost exceeded or max iterations hit without fixing.
    """
    if state["total_cost_usd"] > MAX_COST_PER_RUN:
        return {"status": "cost_exceeded"}
    if state["iterations"] >= MAX_ITERATIONS and state["exit_code"] != 0:
        return {"status": "blocked"}
    return {"status": "done"}


# -----------------------------------------------------------------------
# Edge functions
# -----------------------------------------------------------------------

def should_continue(state: dict) -> Literal["diagnose", "evaluate", "end"]:
    """
    Called after execute. Routes based on execution result and limits.

    Priority order:
        1. Cost gate — money ceiling, always checked first
        2. Success — route to evaluate (not directly to end — exit_code==0
           doesn't mean correct output)
        3. Max iterations — give up
        4. Default — diagnose (keep trying)
    """
    if state["total_cost_usd"] > MAX_COST_PER_RUN:
        print(f"[gate] cost exceeded: ${state['total_cost_usd']:.4f}")
        return "end"
    if state["exit_code"] == 0:
        print("[gate] code runs successfully → evaluate")
        return "evaluate"
    if state["iterations"] >= MAX_ITERATIONS:
        print(f"[gate] max iterations ({MAX_ITERATIONS}) reached → blocked")
        return "end"
    return "diagnose"


def should_evaluate_continue(state: dict) -> Literal["diagnose", "end"]:
    """
    Called after evaluate. Routes based on score.

    If total cost exceeds max cost per run → end as blocked.
    If score passed → end.
    If score failed but iterations remain → diagnose with feedback.
    If score failed and iterations exhausted → end as blocked.
    """
    if state["total_cost_usd"] > MAX_COST_PER_RUN:
        print(f"[gate] cost exceeded during evaluation: ${state['total_cost_usd']:.4f}\n"
              f"→ blocked")
        return "end"
    if state["status"] == "done":
        return "end"
    if state["iterations"] >= MAX_ITERATIONS:
        print("[gate] evaluation failed but max iterations reached → blocked")
        return "end"
    print(f"[gate] evaluation failed (score={state['evaluator_score']}) → diagnose")
    return "diagnose"


def should_approve_continue(state: dict) -> Literal["execute", "diagnose"]:
    """
    Called after human_approval.
    Approved → execute the patched code.
    Rejected → back to diagnose with rejection reason as context.
    """
    if state.get("human_approval") == "approved":
        return "execute"
    return "diagnose"

