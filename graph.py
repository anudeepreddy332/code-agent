"""
LangGraph state machine for the code-fix agent.

Defines AgentState and build_graph() only.
All node and edge logic lives in nodes.py.

State machine:
    START → execute → should_continue?
                        ├─ diagnose → patch → human_approval → execute (loop)
                        ├─ evaluate → should_evaluate_continue?
                        │               ├─ diagnose (score failed)
                        │               └─ end (score passed)
                        └─ end

Run: imported by main.py. No standalone execution.
"""

from typing import Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from nodes import (
    node_execute, node_diagnose, node_patch,
    node_human_approval, node_evaluate, node_end,
    should_continue, should_evaluate_continue, should_approve_continue,
)


# Agent state schema
# Every field that flows through the graph must be declared here.
# LangGraph passes this dict between nodes - nodes read what they need, return only the fields they're updating.

class AgentState(TypedDict):
    code: str                      # current version of the code being fixed
    original_code: str             # never mutated — used for diffing
    error: str | None              # most recent stderr
    stdout: str                    # most recent stdout
    exit_code: int                 # most recent exit code
    iterations: int                # execute→diagnose→patch cycles so far
    total_cost_usd: float          # accumulated API cost
    total_tokens: int              # accumulated tokens
    status: Literal["running", "done", "blocked", "cost_exceeded"]
    diagnosis: str | None          # LLM's diagnosis of current error
    patch_explanation: str | None  # LLM's explanation of what it changed
    evaluator_score: int | None    # 1-10 score from node_evaluate
    evaluator_feedback: str | None # evaluator's explanation of score
    expected_output: str | None    # optional ground truth for evaluator
    human_approval: str | None     # "approved" or "rejected"
    rejection_reason: str | None   # human's reason if rejected
    attempt_history: list[dict]    # [{code, error, diagnosis, patch_explanation}, ...]
    run_id: str                    # unique ID for this run
    bypass_hitl: bool              # True = auto-approve (benchmark mode)


# Graph assembly

def build_graph() -> StateGraph:
    """
    Wire all nodes and edges into a compiled LangGraph.

    add_node: registers a function as a named node.
    add_edge: unconditional A → B transition.
    add_conditional_edges: routing function picks next node.

    MemorySaver checkpointer: required for interrupt()/resume in
    node_human_approval. Stores full state in memory between
    invoke() calls so the graph can resume exactly where it paused.

    Compile validates the graph at build time — missing edges or
    orphan nodes raise immediately, not at runtime.
    """

    builder = StateGraph(AgentState)

    builder.add_node("execute", node_execute)
    builder.add_node("diagnose", node_diagnose)
    builder.add_node("patch", node_patch)
    builder.add_node("human_approval", node_human_approval)
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
    builder.add_edge("patch", "human_approval")
    builder.add_conditional_edges(
        "human_approval",
        should_approve_continue,
        {"execute": "execute", "diagnose": "diagnose"},
    )

    builder.add_conditional_edges(
        "evaluate",
        should_evaluate_continue,
        {
            "diagnose": "diagnose", # score failed → retry with feedback
            "end": "end",           # score passed → done
        }
    )
    builder.add_edge("end", END)

    memory = MemorySaver()

    return builder.compile(checkpointer=memory)

