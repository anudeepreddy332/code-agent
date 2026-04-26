"""
Reads latest benchmark JSON and extracts failed runs.
Prints a regression test stub for each failure.

Usage:
    python -m scripts.analyze_failures

A regression test for a failed script means:
    - The script is kept in tests/broken_scripts/ (already done)
    - Expected behavior is documented (what should the fix produce?)
    - Re-running the benchmark after any agent change should not
      cause a previously-fixed script to fail (that's a regression)

This script helps you track which scripts are reliably fixed vs flaky.

"""

import json
from pathlib import Path

RESULTS_DIR = Path("outputs/benchmark_results")

def main():
    result_files = sorted(RESULTS_DIR.glob("benchmark_*.json"))
    if not result_files:
        print("No benchmark results found. Run scripts/benchmark.py first.")
        return

    latest = result_files[-1]
    data = json.loads(latest.read_text())
    results = data["results"]

    failed = [r for r in results if not r["fixed"]]
    passing = [r for r in results if r["fixed"]]

    print(f"Benchmark: {latest.name}")
    print(f"Passing: {len(passing)} | Failed: {len(failed)}\n")

    if not failed:
        print("All scripts fixed. No regressions to track.")
        return

    print("Failed scripts — add to regression tracker:\n")
    for r in failed:
        print(f"  Script  : {r['script']}")
        print(f"  Status  : {r['status']}")
        print(f"  Iter    : {r['iterations']}")
        print(f"  Error   : {r['error_type'][:80]}")
        print(f"  Action  : Review {r['script']} — is this a hard case or an agent bug?")
        print()

    print("Regression rule: if a script in 'passing' fails on next run, that's a regression.")
    print("Track this by comparing benchmark JSONs across runs.\n")


if __name__ == "__main__":
    main()














