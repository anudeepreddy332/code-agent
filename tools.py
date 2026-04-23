"""
Sandboxed Py execution tool for the code-fix agent.

Runs arbitrary Python code in a subprocess with a hard timeout.
Returns structured output the LLM can reason about.

Why subprocess and not exec(): exec() runs in the same process —
a crash, infinite loop, or sys.exit() in the code kills the agent.
subprocess isolates the risk completely.

Run: imported by graph.py and nodes.py. No standalone execution.
"""
import tempfile
import subprocess
import sys
import os
from src.code_agent.config import EXECUTION_TIMEOUT

def execute_python(code: str) -> dict:
    """
    Write code to a temp file and run it in subprocess.

    Input:
        code: str - Python source code to execute.

    Output dict:
        {
            "success": bool,    # True if exit code == 0
            "stdout": str,      # captured standard output
            "stderr": str,      # captured standard error (tracebacks live here)
            "exit_code": int,   # 0 = success, 1 = exception, 2 = syntax error
            "timed_out": bool   # True if execution exceeded EXECUTION_TIMEOUT
        }

    Vulnerabilities:
        - Malicious code can still do damage within the subprocess (disk writes,
          network calls, fork bombs). For this learning project, acceptable.
          Production would use a container with no network and read-only filesystem.
        - tempfile is cleaned up in finally block — safe even if execution crashes.
        - EXECUTION_TIMEOUT is a hard wall via subprocess timeout parameter.
          The process is killed (SIGKILL on Unix) if it exceeds the limit.

    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=EXECUTION_TIMEOUT,
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.returncode,
            "timed_out": False,
        }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "stdout": "",
            "stderr": f"[execution timeout] Script exceeded {EXECUTION_TIMEOUT}s limit.",
            "exit_code": -1,
            "timed_out": True,
        }

    except Exception as e:
        return {
            "success": False,
            "stdout": "",
            "stderr": f"[execution error] {e}",
            "exit_code": -1,
            "timed_out": False,
        }

    finally:
        os.unlink(tmp_path)         # always clean up temp file



















