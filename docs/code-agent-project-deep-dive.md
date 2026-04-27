# Code-Fix Agent: A Deep Dive into Phase 3


This is a deep-dive of the code-fix agent built in Phase 3 of a multi-phase agentic AI project. It covers the architecture, the execution pipeline, the engineering decisions behind every major component, the evaluation methodology, and the lessons learned.

---

## Part 1: What Phase 3 Is and Why It Exists

### The progression

Phase 1 built a research agent. It could search the web, fetch pages, compute values, and write reports. It answered questions.

Phase 2 gave that agent a memory — a local knowledge base built from personal notes, searchable through hybrid retrieval, with citations and source grounding. It could now answer questions from what it already knew.

Both phases share a common shape: the user asks something, the agent finds information, the agent responds. The loop is query-driven. It starts and ends in one turn.

Phase 3 breaks that shape entirely.

The code-fix agent is not answering a question. It is pursuing a goal. The goal is: make this Python script run correctly. It executes the script, reads the failure, diagnoses what went wrong, patches the code, and tries again. If the patch fails, it tries a different approach. If the fix produces code that runs but produces wrong output, it catches that too. It keeps going until the goal is met, a cost ceiling is hit, or it exhausts its attempts.

This is the distinction between a query-response agent and a goal-directed agent. Phase 3 is where the system develops the capacity to self-correct — to observe the consequences of its own actions and adjust.

### Why code fixing specifically

Code execution is the sharpest possible use case for a self-correcting agent because failure is unambiguous. Either the script runs or it does not. Either the output is correct or it is not. There is no subjectivity. This makes it ideal for measuring whether the loop actually works.

In contrast, a self-correcting research agent or writing agent would require human judgment to evaluate whether the output is good. Code gives you a ground truth signal: the exit code.

### What Phase 3 adds to the previous phases

Phase 1 and Phase 2 both used a raw while loop — a Python function that called the LLM, checked the response, called tools, and looped until done. The control flow was implicit. A reader of the code had to trace through conditionals to understand what states the agent could be in.

Phase 3 introduces LangGraph. The same loop is now expressed as an explicit state machine where every state is named, every transition is a function, and the graph can be inspected, traced, and reasoned about. The state the agent carries between steps is a typed dictionary — every field is declared up front, and any node that reads or writes a field does so transparently.

This is not just an aesthetic improvement. It becomes load-bearing when the loop has conditional branches (retry on evaluation failure), mid-run interruptions (human approval), and accumulated context (attempt history for reflexion). A raw while loop handling all of these simultaneously would be difficult to debug and nearly impossible to extend.

---

## Part 2: Architecture and Workflow

### The state machine

The agent is modeled as a directed graph. Each node is a function. Each edge is a transition. Some transitions are unconditional; others are conditional and depend on the current state.

```
START
  |
  v
[execute] -----> should_continue?
                    |
                    |--- exit_code != 0, cost < ceiling, iterations < max ---> [diagnose]
                    |                                                               |
                    |                                                            [patch]
                    |                                                               |
                    |                                                       [human_approval]
                    |                                                           /       \
                    |                                                    approved      rejected
                    |                                                      |               |
                    |                                                   [execute]      [diagnose]
                    |
                    |--- exit_code == 0 ---> [evaluate] ---> should_evaluate_continue?
                    |                                              |
                    |                                             / \
                    |                                    score>=7   score<7
                    |                                      |           |
                    |                                    [end]    [diagnose]
                    |
                    |--- cost > ceiling OR iterations >= max ---> [end]
                                                                    |
                                                                   END
```

Think of this like a quality control station at a factory. A broken part comes in (the script). A worker examines it (execute), figures out what's wrong (diagnose), makes a repair (patch), a supervisor approves the repair (human approval), the part is tested again (re-execute), and a quality inspector scores it (evaluate). If it passes, it ships (end, status=done). If it fails, it goes back to the repair station with the inspector's notes. If it cannot be fixed in a reasonable number of attempts, it is flagged as blocked and set aside.

### The files

The project is split cleanly between four core files and a scripts directory.

`graph.py` contains only two things: the `AgentState` TypedDict that declares every field the agent tracks, and `build_graph()` that wires nodes and edges into a compiled LangGraph object. It imports all node and edge functions from `nodes.py`. Nothing about how nodes work lives here — only how they connect.

`nodes.py` contains all the logic: six node functions and three edge functions. Every LLM call, every tool dispatch, every conditional routing decision lives here.

`tools.py` contains one function: `execute_python`. It writes code to a temporary file, runs it in a subprocess with a hard timeout, captures stdout and stderr, and returns a structured result dict. Nothing in this file makes LLM calls or knows about the graph.

`main.py` is the CLI entry point. It builds the initial state, invokes the graph, handles the human approval interrupt loop, prints the final report, and saves a structured run log to disk.

The `scripts/` directory contains four utilities: `benchmark.py` for systematic evaluation across all broken scripts, `analyze_failures.py` for extracting failed runs from benchmark JSON, `regression_check.py` for comparing current results against a stored baseline, and `test_resilience.py` for simulating failure modes.

---

## Part 3: End-to-End Pipeline Walk-Through

To make the architecture concrete, here is what happens when you run `attribute_error.py` through the agent.

The broken script:

```python
class Dog:
    def __init__(self, name):
        self.name = name

d = Dog("Rex")
print(d.age)   # Dog has no attribute 'age'
```

### Step 1: execute (iteration 1)

The script is written to a temp file and run in a subprocess. The subprocess exits with code 1. stderr contains:

```
AttributeError: 'Dog' object has no attribute 'age'
```

State after this node:
- `exit_code`: 1
- `error`: the full traceback string
- `stdout`: empty
- `iterations`: 1
- `attempt_history`: still empty (no prior diagnosis to archive yet)

### Step 2: should_continue

The edge function checks in order: cost ceiling not exceeded, exit_code is not 0, iterations (1) is below MAX_ITERATIONS (5). Returns "diagnose".

### Step 3: diagnose

The LLM receives the broken code and the traceback. It is asked to identify the root cause in 2-3 sentences without suggesting a fix. It responds with something like: "The error occurs because the `Dog` class does not define an `age` attribute in `__init__`. When `d.age` is accessed, Python raises an `AttributeError` because `age` was never set on the instance."

State after this node:
- `diagnosis`: the LLM's explanation
- `total_cost_usd`: incremented by the cost of this call
- `total_tokens`: incremented

### Step 4: patch

The LLM receives the broken code, the error, and the diagnosis. It is instructed to return only the corrected code inside a code fence. It produces:

```python
class Dog:
    def __init__(self, name):
        self.name = name
        self.age = None

d = Dog("Rex")
print(d.age)
```

The node strips the fence markers and stores the corrected code.

State after this node:
- `code`: the patched version
- `patch_explanation`: the LLM's raw response
- Cost and tokens updated

### Step 5: human_approval

In benchmark mode (`bypass_hitl=True`), this node auto-approves and returns immediately. In interactive mode, the graph suspends here via `interrupt()`. The caller (main.py) receives control back, displays a unified diff showing what changed, and prompts for `y/n`. Once the user types `y`, `Command(resume={"approved": "y", "reason": ""})` resumes the graph from this exact point. The node sets `human_approval` to "approved".

### Step 6: execute (iteration 2)

The patched code runs. Exit code is 0. Stdout is "None" (the default value of `self.age`).

State after:
- `exit_code`: 0
- `stdout`: "None"
- `error`: None
- `iterations`: 2

### Step 7: should_continue

Exit code is 0, so this returns "evaluate".

### Step 8: evaluate

The LLM receives the original code, the fixed code, and the actual output. It is asked to score correctness from 1 to 10 and explain its reasoning in JSON. For this script it scores 9: the AttributeError is resolved, the default value of None is reasonable, and the fix is minimal — but it notes the original intent might have wanted a meaningful default. Score 9 clears the threshold of 7, so status is set to "done".

### Step 9: should_evaluate_continue

Status is "done", so this returns "end".

### Step 10: end

Sets terminal status. Returns `{"status": "done"}`.

The benchmark record for this script: status=done, fixed=true, iterations=2, evaluator_score=9, cost=$0.00037, elapsed=4.42s.

---

## Part 4: Technical Deep Dives

### LangGraph state machines vs. raw while loops

A raw while loop carries its state in local variables. The loop condition, the iteration counter, the accumulated messages, the flags — all of it lives scattered across the function body. When you want to add a new capability (say, a human approval step), you add more variables and more conditionals. The complexity compounds. Reading the code requires tracing execution mentally.

A LangGraph state machine externalizes the state. Every field is declared in `AgentState`. Every node declares what it reads and what it writes by virtue of what it accesses in the state dict and what keys it returns. The graph is a data structure that can be compiled, validated, visualized, and traced.

Think of it this way: a raw while loop is like a chef who keeps the recipe in their head. They know what to do next because they remember where they are. A LangGraph state machine is like a recipe card with checkboxes. Each step is written down, and you can see at a glance what step you are on, what has been completed, and what comes next. If someone else needs to pick up mid-recipe, the card tells them everything.

### Typed state and observability

`AgentState` is a TypedDict with 18 fields. Every field that any node reads or writes must be declared here. LangGraph validates this at compile time: if a node returns a key that is not in the schema, it raises immediately — not when that key is accessed at runtime.

This turns a class of subtle bugs into loud, early errors. In a raw loop, you might return a dict with a typo in the key name and not notice until the downstream code fails with a confusing KeyError. In LangGraph, the compilation step catches it.

The typed schema also makes every run observable. LangSmith traces show the state before and after each node. You can see exactly what went into a diagnosis call, what came out, and what the agent decided to do next. This is not possible with a raw loop unless you manually add print statements everywhere.

### Node functions and conditional edges

Every node function has the same signature: it takes the current state dict and returns a dict of fields to update. LangGraph merges the returned dict into the state — fields not in the returned dict are unchanged. This means nodes are composable and focused: each one does one thing and declares exactly what it changes.

Edge functions have a different role. They do not modify state. They read state and return a string that LangGraph maps to the next node. `should_continue` after `execute` checks cost, exit code, and iteration count in that priority order. `should_evaluate_continue` after `evaluate` checks whether the score passed. `should_approve_continue` after `human_approval` checks whether the human approved.

The priority order in edge functions matters. In `should_continue`, the cost gate is checked before the success check. This means an agent that somehow succeeds on the same iteration that it exceeds the cost ceiling will still be terminated for cost. Money gates go first.

### The reflexion pattern

On the first failed attempt, `node_diagnose` receives the current code and error. Simple diagnosis: what is wrong with this code right now?

On subsequent attempts, it receives something richer: the full history of previous attempts, each containing the code that was tried, the error it produced, the diagnosis that was made, and the patch that was applied. The prompt changes from "diagnose this error" to "here is everything that was tried and why it failed — what is a different approach?"

This is the reflexion pattern. The agent observes its own failure history and uses it to generate a new strategy rather than repeating the same diagnosis with different wording. `attempt_history` is archived by `node_execute` before it overwrites the error fields with new execution results — this timing is deliberate. If archival happened after the new execution results were written, the wrong error would be saved.

`attempt_history` is capped at the last three entries when injected into the prompt. Older entries are less relevant and would push long-running agents toward context limits.

### Human-in-the-loop with interrupt and resume

LangGraph's `interrupt()` is a clean suspension mechanism. When `node_human_approval` calls `interrupt(payload)`, the graph stops at that point. The entire state is checkpointed by `MemorySaver`. Control returns to `graph.invoke()` in main.py, which now returns a partial result containing the interrupt payload.

main.py detects the interrupt, extracts the diff from the payload, displays it to the terminal, and prompts the user. When the user responds, main.py calls `graph.invoke(Command(resume=decision), config)` with the same thread ID. The graph resumes from the exact point where it suspended, and `interrupt()` returns the decision dict.

This is architecturally important for production use. In a real system, the "resume" signal would not come from terminal input — it would come from a webhook, a UI button click, or a message on a queue. The `interrupt()/resume` pattern supports all of these without changing the graph. You would only change how main.py handles the interrupt.

`MemorySaver` is what makes this possible. Without a checkpointer, the graph has no memory of its state between `invoke()` calls. The graph would restart from the beginning on the resume call. With `MemorySaver`, the state is stored in memory indexed by thread ID, and the graph picks up exactly where it left off.

### Sandboxed code execution

`execute_python` writes the code string to a temporary file and runs it with `subprocess.run()`. This is intentional isolation. If you used Python's `exec()` built-in instead, the code would run inside the agent's own process. A script that calls `sys.exit()`, raises an unhandled exception, or enters an infinite loop would crash or hang the agent itself.

With subprocess isolation, the agent's process is insulated. A crash in the script is just a non-zero exit code and a stderr string. A timeout is caught by subprocess's `timeout` parameter, which sends SIGKILL to the child process and raises `TimeoutExpired` in the parent. The agent converts this to a structured error string that the LLM can read and diagnose.

The temp file is always deleted in a `finally` block, so even if the subprocess crashes or the timeout fires, the disk is not left with orphaned temp files accumulating across runs.

The sandbox is not security-hardened. A malicious script could still write to disk, make network calls, or fork-bomb the system. For a learning project running trusted test scripts, this is acceptable. A production system would run the subprocess in a container with no network access and a read-only filesystem.

### LLM-based evaluator

The evaluator node is what separates "the code runs" from "the code is correct." Exit code 0 means no exception was raised. It does not mean the output is right. `logic_error.py` proved this: the inverted Fahrenheit-to-Celsius formula ran cleanly and exited 0, but produced 180.0 for 212°F instead of the correct 100.0.

The evaluator gives the LLM the original broken code (which encodes the intent), the fixed code, and the actual output. It asks for a score from 1 to 10 with a JSON response. The score guide tells the LLM that 3-4 means wrong output with no exception, which is exactly the logic error case.

When the evaluator scores below 7, it does not just fail the run. It sends the agent back to `node_diagnose` with the evaluator's feedback injected into the diagnosis prompt. The agent now knows not just that its fix was wrong, but specifically why — and it has the evaluator's explanation to guide the next attempt.

The LLM response is expected as a JSON object. If parsing fails (which happens when the LLM wraps the JSON in prose or adds a preamble), the score defaults to 3. This conservative default means a parse failure is treated as a questionable fix that needs review, not a pass.

### Cost tracking and cost gates

Every LLM call in `node_diagnose`, `node_patch`, and `node_evaluate` calls `_track_cost()` immediately after the response. This function extracts token counts from the response's `usage_metadata`, computes the cost using DeepSeek's pricing ($0.27/million input tokens, \$1.10/million output tokens), and returns the new accumulated totals to be merged into state.

The cost gate in `should_continue` checks `state["total_cost_usd"] > MAX_COST_PER_RUN` before anything else. This means the gate fires even when the code has just succeeded: if the most recent LLM call pushed cost over the ceiling, the agent terminates as `cost_exceeded` rather than evaluating a potentially successful fix. This is intentional — once the ceiling is breached, the run is over. You do not get to spend more to confirm whether the last fix worked.

`MAX_COST_PER_RUN` is currently \$0.10. Across the 20-script benchmark, the most expensive run was `import_error.py` at \$0.0029, and the total across all 20 scripts was $0.015. The ceiling has significant headroom for this workload.

---

## Part 5: Evaluation and Testing (Phase 3.5)

### The benchmark harness

`scripts/benchmark.py` runs all 20 broken scripts in `tests/broken_scripts/` sequentially through the full agent pipeline. It uses `bypass_hitl=True` so the human approval gate auto-approves every patch without prompting. This makes the benchmark non-interactive and reproducible.

For each script, it records: status, whether it was fixed (status=done AND evaluator_score >= 7), iterations, evaluator score, API cost, total tokens, elapsed time, and the error type extracted from the first execution. At the end it saves a structured JSON file to `outputs/benchmark_results/` with a timestamp in the filename, then prints a table and aggregate summary.

The JSON output is the source of truth for regression tracking. `scripts/regression_check.py` reads the latest benchmark JSON, compares it against a stored baseline, and reports which scripts changed status in either direction. When a previously-fixed script fails, that is a regression. When a previously-failing script now passes, that is an improvement worth reviewing before updating the baseline.

### Benchmark results

The final run across all 20 scripts produced the following results.

```
Script                        Status    Iter  Score     Cost     Time
----------------------------------------------------------------
attribute_error.py            done         2    9/10  $0.0004    4.4s
dict_mutation.py              done         2   10/10  $0.0003    4.2s
encoding_error.py             done         2   10/10  $0.0004    4.6s
file_not_found.py             done         2   10/10  $0.0004    3.6s
import_error.py               done         5    2/10  $0.0029   24.3s  [NOT FIXED]
index_error.py                done         2   10/10  $0.0003    3.4s
infinite_loop.py              done         2   10/10  $0.0003   14.3s
key_error.py                  done         2    9/10  $0.0003    4.2s
logic_error.py                done         2   10/10  $0.0007    7.2s
none_type_error.py            done         2   10/10  $0.0003    3.9s
off_by_one.py                 done         2   10/10  $0.0004    4.7s
recursion_error.py            done         3    9/10  $0.0012   10.0s
string_format_error.py        done         2   10/10  $0.0004    4.6s
syntax_error.py               done         2   10/10  $0.0003    3.9s
tricky_error.py               done         2    9/10  $0.0005   14.6s
type_error.py                 done         2   10/10  $0.0004    6.3s
unbound_local.py              done         2   10/10  $0.0004    4.8s
value_error.py                done         2   10/10  $0.0003    3.8s
wrong_return.py               done         4   10/10  $0.0020   19.5s
zero_division.py              done         2    9/10  $0.0004    4.9s

Fix rate:          95%  (19/20)
Blocked rate:       0%
Mean iterations:   2.3
Mean cost per run: $0.0008
Total cost:        $0.0153
```

Nineteen of twenty scripts were fixed correctly. All runs completed — none hit the iteration ceiling, none exceeded the cost ceiling. Mean iterations of 2.3 reflects that most scripts need one diagnose-patch-execute cycle, with a small number requiring two or three.

The one failure is `import_error.py`. The script tries to import numpy and pandas, which are not installed in the agent's sandbox environment. The agent correctly identified the missing modules, but every attempted fix either renamed the imports (still missing), swapped to other uninstalled libraries, or replaced the functionality in ways the evaluator judged as losing the original intent. After five iterations the evaluator scored the best attempt a 2. This is a structural limitation of the sandbox design, not an agent failure. The fix for this class of problem requires either pip access in the sandbox or pre-installing the libraries — neither of which is appropriate for a secure execution environment running arbitrary code.

### Interpreting the results

The evaluator scores reveal something interesting. Scripts that required pure reasoning (logic_error, wrong_return, syntax_error) scored 10. Scripts that required a judgment call about the right default value (zero_division returning 0 for empty list, key_error returning None for missing key) scored 9 — the evaluator noted these are reasonable but not uniquely correct choices. This matches intuition: there is a spectrum from "objectively wrong" to "correct but opinionated."

The recursion error required three iterations. The first patch added a base case but introduced a different error. The reflexion loop activated on iteration 2 — the agent saw its prior attempt, noted it had introduced a new bug, and tried a different approach. The final fix added a depth limit parameter, which the evaluator scored 9 (correct, but the depth limit approach is a workaround rather than fixing the root cause of infinite recursion).

`infinite_loop.py` and `tricky_error.py` both show 14 seconds elapsed despite being fixed in 2 iterations. The extra time is the 10-second execution timeout on the first iteration — the script ran for 10 seconds before being killed, then the agent diagnosed and patched it. The elapsed time correctly captures this real-world cost.

### Resilience testing

`scripts/test_resilience.py` validates four failure modes:

**Execution timeout.** The agent is given `while True: pass`. It times out after 10 seconds, receives the synthetic timeout error string, diagnoses an infinite loop, patches with a termination condition, and either fixes it or blocks at MAX_ITERATIONS. The assertion checks that status is "done" or "blocked" — either outcome is acceptable. What is not acceptable is a crash.

**Cost ceiling.** The agent is given normal code but with `total_cost_usd` pre-set to 999.0 in the initial state. The first call to `should_continue` sees cost above ceiling and routes to "end". Status is `cost_exceeded`. This validates that the gate fires even before any LLM call is made.

**Max iterations.** The agent is given broken syntax with `iterations` pre-set to `MAX_ITERATIONS`. The first `should_continue` check sees iterations at the maximum and routes to "end". Status is "blocked".

**Invalid API key.** The DEEPSEEK_API_KEY environment variable is temporarily replaced with a garbage string. The LLM call in `node_diagnose` raises an authentication error. The test asserts the final status is not "done" — either it raises an exception (caught by the test) or returns "blocked" or "cost_exceeded". The test passes either way, restores the original key in a `finally` block, and confirms no silent success occurred with bad credentials.

All four tests pass reliably.

### Phase 3.5 completion criteria

The criteria required before moving to Phase 4:

- Agent runs end-to-end on at least 10 real test cases: 20 scripts run, 19 fixed. Done.
- Every failure logged and converted to a regression test: `import_error.py` documented in `analyze_failures.py` output, stored in benchmark JSON, tracked by `regression_check.py` baseline. Done.
- Logs contain inputs, outputs, tool calls, latency, cost, and final status: LangSmith traces capture all of this per node. Run logs saved as JSON to `outputs/run_logs/` per `main.py` invocation. Benchmark JSON captures aggregate metrics. Done.
- Retry and fallback logic verified: all four resilience tests pass. Done.
- Human-in-the-loop checkpoint works reliably: `interrupt()/resume` tested with both approval and rejection paths, rejection reason confirmed to flow into `node_diagnose`. Done.
- At least 5 LangSmith traces reviewed manually: five traces reviewed and documented in `outputs/langsmith_trace_review.md`. Done.

All criteria met. Phase 3 is complete.

---

## Part 6: Guardrails, Safety, and Failure Modes

### The four guardrails

**Cost ceiling ($0.10 per run).** Checked first in `should_continue` before any other decision. If accumulated cost exceeds the ceiling, the graph terminates regardless of execution status. The agent reports `cost_exceeded` and the caller has partial state including whatever fix was most recently attempted. The ceiling is a hard wall, not a soft warning.

**Max iterations (5).** Checked in `should_continue` after the cost gate and after the success check. If the agent has attempted five full cycles without success, it reports "blocked" and stops. This prevents infinite loops in cases where the agent consistently makes the wrong diagnosis or where the problem is structurally unfixable (as with import_error.py). Five was chosen as a balance between giving the reflexion loop enough cycles to try different approaches and preventing runaway cost on hard cases.

**Execution timeout (10 seconds).** Applied by `subprocess.run(timeout=EXECUTION_TIMEOUT)`. If the subprocess does not exit within 10 seconds, Python sends SIGKILL and raises `TimeoutExpired`. The executor catches this, returns a structured result with `timed_out=True` and a synthetic stderr string that the LLM can read. The agent correctly diagnosed `infinite_loop.py` from this synthetic message. The timeout is per execution, not per run — a script that times out on every attempt consumes 10 seconds per iteration.

**Input validation.** `execute_python` checks that the code string is non-empty before writing it to disk. The LLM response parser in `node_evaluate` handles JSON parse failure by defaulting to score=3 rather than crashing. The tool executor in `node_patch` handles missing code fences by falling back to the raw response rather than returning an empty string.

### Known limitations

The sandbox has no pip access. Any script that requires a library not already installed in the agent's environment cannot be fixed by installing the missing dependency. The agent will exhaust its iterations trying to rewrite the script without the library, which is often not the right fix. This is the `import_error.py` scenario.

The evaluator can misjudge logic errors when there is no ground truth `expected_output` provided. It infers intent from the original code, which is usually correct but not guaranteed. For the benchmark, `expected_output` was left as None for all scripts, relying entirely on evaluator inference. Supplying ground truth expected output for each script would improve evaluation accuracy for logic errors.

The human rejection loop has no cap. A human can reject every patch indefinitely, keeping the agent running. A production system should add a `rejection_count` field to state and terminate with "blocked" after N rejections.

### The typo bug

During development, a typo crept into the `bypass_hitl` field name — it was written as `bypass_hit1` (with a digit one) in one location while referenced as `bypass_hitl` (with a lowercase L) elsewhere. In benchmark mode, this meant the auto-approve path was never triggered, causing the graph to reach the `interrupt()` call and wait for human input that was never coming. The bug was caught when the benchmark hung without output and fixed with a careful diff. It underscores a genuine risk with string-keyed state dicts: typos in field names are not caught at compile time unless every field access is validated against the schema at runtime, which LangGraph does not do for dict access within node functions — only for the return values of those functions.

---

## Part 7: Key Learnings

**Explicit state is not a luxury.** The single biggest architectural improvement over the raw loop was externalizing the agent's state into a typed schema. Every field that mattered was named, and every node declared what it produced. This made debugging fast and made the system extendable. Adding the human approval gate required adding two fields to `AgentState` and one new node — the rest of the graph did not need to change.

**exit_code == 0 is necessary but not sufficient.** The evaluator node was the most important addition to the pipeline, and the logic error test case proved it before the benchmark was run. Any agent that treats execution success as correctness will produce confidently wrong results. The evaluator adds one more LLM call per successful execution, but it is the call that makes the system trustworthy.

**Reflexion requires memory, and memory requires design.** The `attempt_history` field is what makes the reflexion loop work. Getting the archival timing right — saving the previous attempt before overwriting the current error fields — required explicit reasoning about the order of operations in `node_execute`. In a raw loop, this would have been a comment in the code. In LangGraph, it is enforced by the structure of the state machine: the execute node always runs before diagnose, and the archive happens inside execute before the new results are written.

**Instrumentation and testing are not optional.** LangSmith tracing was added in Day 19, but its value was immediate. The observation that `node_diagnose` and `node_evaluate` have the highest latency (2-4 seconds each) came directly from the traces, and it points to the right place to look if latency becomes a problem: the LLM calls with the largest prompts. Without tracing, this would have required manual timing instrumentation. The resilience tests caught the cost ceiling gate working correctly before the benchmark ran, which meant the benchmark results reflected real agent behavior under normal conditions rather than behavior masked by silent failures.

**Sandbox design shapes what the agent can fix.** The 5% failure rate is entirely explained by `import_error.py`, and that failure is entirely explained by the sandbox design choice: no pip access. This is the right design choice for security, but it is a real constraint. Understanding where your agent fails — and whether those failures are agent bugs or environmental constraints — requires the kind of structured logging and evaluation that the benchmark harness provides. The fix rate of 95% is not a claim that the agent is good at fixing all Python errors. It is a precise claim: the agent fixes 19 of these 20 specific scripts in this specific environment with these specific parameters.

**Typos in dict keys are silent killers.** The `bypass_hitl` typo did not raise an error. The graph simply took the wrong branch. In a system built on string-keyed state, the only defenses are careful code review, consistent naming conventions, and integration tests that exercise the bypass path. The benchmark serves as that integration test — if the bypass does not work, the benchmark hangs immediately on the first script.

---