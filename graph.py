"""
Code-fix agent built on LangGraph

State machine:
    START → execute → should_continue?  → diagnose → patch → execute (loop)
                                        ↓ (success or limits hit)
                                        END

Each node is a pure function: takes AgentState, returns dict of fields to update.
LangGraph merges the returned dict into the current state automatically

Why TypeDict for state: LangGraph requires a typed state schema.
Every field that any node reads or writes must be declared here.
This is the contract between nodes - if a node writes a field not declared here,
LangGraph raises at graph compile time, not at runtime. Fail fast.

Run: python -m main
"""

from langchain_core.messages import SystemMessage, HumanMessage
from typing_extensions import TypedDict
from typing import Literal
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from src.code_agent.config import (DEEPSEEK_API_KEY, DEEPSEEK_MODEL, DEEPSEEK_BASE_URL,
                                   COST_PER_1M_INPUT_TOKENS, COST_PER_1M_OUTPUT_TOKENS,
                                   MAX_COST_PER_RUN, MAX_ITERATIONS)
from tools import execute_python



# Agent state schema
# Every field that flows through the graph must be declared here.
# LangGraph passes this dict between nodes - nodes read what they need, return only the fields they're updating.

class AgentState(TypedDict):
    code: str                    # current version of the code being fixed
    original_code: str           # never mutated — used for diffing in reports
    error: str | None            # most recent stderr from execution
    stdout: str                  # most recent stdout
    exit_code: int               # most recent exit code
    iterations: int              # how many fix attempts so far
    total_cost_usd: float        # accumulated cost across all LLM calls this run
    total_tokens: int            # accumulated tokens
    status: Literal["running", "done", "blocked", "cost_exceeded"]
    diagnosis: str | None        # LLM's explanation of what's wrong
    patch_explanation: str | None  # LLM's explanation of what it changed
    evaluator_score: int | None  # 1-10 score from evaluator node
    run_id: str                  # unique ID for this run (for logging)
    attempt_history: list[dict]  # [{code, error, diagnosis, patch_explanation}, ...]


def _make_client() -> ChatOpenAI:
    """
    Build LangChain ChatOpenAI client pointed at DeepSeek.

    Why ChatOpenAI and not raw openai client: LangGraph nodes receive and return LangChain message objects.
    ChatOpenAI handles the conversion. Deepseek's OpenAI-compatible API works with this client by overriding
    base_url and api_key.

    Vulnerability: client is created per-call in each node that needs it.
    Slightly wasteful but avoids shared state across nodes. Acceptable here.
    """
    return ChatOpenAI(
        model=DEEPSEEK_MODEL,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        temperature=0.3,
    )

def _track_cost(state: AgentState, response) -> dict:
    """
    Extract token usage from LangChain response and compute cost delta.
    Returns dict of updated cost fields to merge into state.

    Why track here and not in the client: LangGraph state is the single source of truth.
    Centralizing cost tracking in state means the cost gate in should_continue() sees
    the real accumulated cost, not an estimate.
    """

    usage = response.usage_metadata
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    cost = (input_tokens / 1e6 * COST_PER_1M_INPUT_TOKENS) + (output_tokens / 1e6 * COST_PER_1M_OUTPUT_TOKENS)
    return {
        "total_cost_usd": state["total_cost_usd"] + cost,
        "total_tokens": state["total_tokens"] + input_tokens + output_tokens,
    }

# Node 1: execute
# Runs the current code. Updates error, stdout, exit_code in state
# This node has no LLM call - it's pure execution.

def node_execute(state: AgentState) -> dict:
    """
    Execute current code in subprocess. Archive previous attempt before overwriting error state.

    Why archive here: after execute runs, the state fields (error, exit_code)
    are about to be overwritten with the new execution result.

    Why this is a separate node and not part of diagnose:
    Separation of concerns. Execute is deterministic - same code, same result.
    Diagnose is an LLM call - expensive and non-deterministic.
    Keeping them separate means you can re-execute without re-diagnosing
    (useful for the evaluator node later).

    Vulnerability: attempt_history grows unboundedly. At MAX_ITERATIONS=5,
    max 5 entries — acceptable. If MAX_ITERATIONS is raised significantly,
    the history injected into the prompt will eventually exceed context limits.
    Mitigation: only inject the last 3 attempts into the prompt.
    """
    print(f"\n[execute] iteration {state['iterations'] + 1}")
    result = execute_python(state["code"])
    print(f"  exit_code={result['exit_code']} | timed_out={result['timed_out']}")
    if result["stderr"]:
        print(f"  stderr: {result['stderr'][:200]}")

    # Archive the current attempt before state is overwritten
    # Only archive if there was a previous diagnosis (skip the very first execution)

    new_history = list(state.get("attempt_history", []))
    if state.get("diagnosis") is not None:
        new_history.append({
            "code": state["code"],
            "error": state.get("error", ""),
            "diagnosis": state.get("diagnosis", ""),
            "patch_explanation": state.get("patch_explanation" ""),
        })

    return {
        "stdout": result["stdout"],
        "error": result["stderr"] if not result["success"] else None,
        "exit_code": result["exit_code"],
        "iterations": state["iterations"] + 1,
        "attempt_history": new_history,
    }


# Conditional edge: should_continue
# Called after execute. Decides next node based on state.
# Returns a string that LangGraph maps to the next node name.

def should_continue(state: AgentState) -> Literal["diagnose", "evaluate", "end"]:
    """
    Routing function - not a node, an edge condition.

    Decision tree:
        1. Cost exceeded? end immediately (money gate)
        2. Code ran successfully? evaluate (did we actually fix it?)
        3. Max iterations reached? end (blocked)
        4. Otherwise -> diagnose (keep trying)

    Why evaluate on success instead of ending directly:
    The code ran without error, but that doesn't mean that it's correct.
    A script can succeed (exit 0) and produce wrong output.
    The evaluator scores correctness, not just execution success.
    For this phase, we accept exit_code == 0 as "done" and
    skip the evaluator node - we add it in the later phase.
    """
    if state["total_cost_usd"] > MAX_COST_PER_RUN:
        print(f"[gate] cost exceeded: ${state['total_cost_usd']:.4f}")
        return "end"

    if state["exit_code"] == 0:
        print("[gate] code runs successfully → done")
        return "evaluate"

    if state["iterations"] >= MAX_ITERATIONS:
        print(f"[gate] max iterations ({MAX_ITERATIONS}) reached → blocked")
        return "end"

    return "diagnose"


# Node 2: diagnose
# LLM reads the error and explains what's wrong. No code changes here.

def node_diagnose(state: AgentState) -> dict:
    """
    Reflexion-enhanced diagnose node. Ask DeepSeek to diagnose the error. Store explanation in state.

    On iteration 1: standard diagnosis — what's wrong with this code?
    On iteration 2+: reflexion — here's what was tried, why did it fail,
    what should be tried differently?

    Why separate from patch: forcing the LLM to first articulate what's wrong
    before patching improves patch quality. This is the Reflexion pattern -
    explicit diagnosis step reduces hallucinated fixes.

    Input to LLM: current code + stderr.
    Output: plain English diagnosis stored in state["diagnosis"].

    Vulnerability: if error is a timeout (exit_code == -1),
    the stderr is our synthetic "[execution timeout]" message.
    The LLM can still diagnose this - infinite loop, blocking call, etc.
    """
    client = _make_client()

    history = state.get("attempt_history", [])
    is_reflexion = len(history) > 0

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
        )
    else:
        # Reflexion — show previous attempts, ask for new strategy
        recent_attempts = history[-3:]  # last 3 only
        attempts_str = ""
        for i, attempt in enumerate(recent_attempts, 1):
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
            "You will be given the history of attempts and the current error. "
            "Reflect on why previous patches failed, then diagnose the root cause "
            "with a NEW strategy. Do not repeat a diagnosis that already failed. "
            "Be specific about what was wrong with the previous approach."
        )
        user_content = (
            f"ORIGINAL CODE:\n```python\n{state['original_code']}\n```\n\n"
            f"PREVIOUS FAILED ATTEMPTS:{attempts_str}\n"
            f"CURRENT CODE:\n```python\n{state['code']}\n```\n\n"
            f"CURRENT ERROR:\n{state['error']}\n\n"
            "What is wrong with the current approach, and what different strategy should be tried?"
        )

    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=user_content),
    ]

    response = client.invoke(messages)
    cost_update = _track_cost(state, response)
    print(f"[diagnose] {'(reflexion)' if is_reflexion else ''} {response.content[:150]}")

    return {
        "diagnosis": response.content,
        **cost_update,
    }


# Node 3: patch
# LLM applies the fix. Returns corrected code only.

def node_patch(state: AgentState) -> dict:
    """
    Ask DeepSeek to patch the code based on the diagnosis.

    Input: current code + diagnosis
    Output: corrected Python code (full file, not a diff).

    Why full file and not a diff: the LLM is more reliable producing
    complete corrected files than patch hunks. For small scripts this is fine.
    At large file sizes (> 500 lines), you'd switch to targeted diffs.

    Parsing: we expect the LLM to return code in a ```python block.
    We extract it. If parsing fails, we return the raw content and let the next execute() surface the error - fail forward.

    Vulnerability: LLM sometimes wraps the code in prose ("Here's the fixed version:").
    The parser strips the ```pythin fence but may include trailing prose.
    Guard: strip everything outside the first ```python...```block.
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

    # Extract code from ```python block
    raw = response.content
    if "```python" in raw:
        patched = raw.split("```python")[1].split("```")[0].strip()
    elif "```" in raw:
        patched = raw.split("```")[1].split("```")[0].strip()
    else:
        patched = raw.strip() # fallback - take full response

    print(f"[patch] code updated ({len(patched)} chars)")

    return {
        "code": patched,
        "patch_explanation": response.content,
        **cost_update,
    }

# Node 4: evaluate (stub - full implementation in a later stage)
# For now: if code ran successfully, mark done.

def node_evaluate(state: AgentState) -> dict:
    """
    Evaluate whether the fix is correct. Will add evaluator node later.
    For now: execution success = done.
    """
    print("[evaluate] code executed successfully → marking done")
    return {"status": "done", "evaluator_score": 10}


# Node 5: end
# Terminal node - sets final status based on state.

def node_end(state: AgentState) -> dict:
    """
    Set terminal status. Called when cost exceeded or max iterations hit.
    """
    if state["total_cost_usd"] > MAX_COST_PER_RUN:
        return {"status": "cost_exceeded"}
    if state["iterations"] >= MAX_ITERATIONS and state["exit_code"] != 0:
        return {"status": "blocked"}
    return {"status": "done"}

# Graph assembly

def build_graph() -> StateGraph:
    """
    Wire nodes and edges into a compiled LangGraph.

    add_node: registers a function as a graph node.
    add_edge: unconditional transition (A always goes to B).
    add_conditional_edges: function decides which node to go to next.

    Compile: validates the graph (no orphan nodes, no missing edges),
    returns a runnable object. Fails fast at import time if wiring is wrong.
    """
    builder = StateGraph(AgentState)

    builder.add_node("execute", node_execute)
    builder.add_node("diagnose", node_diagnose)
    builder.add_node("patch", node_patch)
    builder.add_node("evaluate", node_evaluate)
    builder.add_node("end", node_end)

    builder.add_edge(START, "execute")
    builder.add_conditional_edges(
        "execute",
        should_continue,
        {
            "diagnose": "diagnose",
            "evaluate": "evaluate",
            "end": "end",
        }
    )
    builder.add_edge("diagnose", "patch")
    builder.add_edge("patch", "execute")       # patch → re-execute → check again
    builder.add_edge("evaluate", END)
    builder.add_edge("end", END)

    return builder.compile()























