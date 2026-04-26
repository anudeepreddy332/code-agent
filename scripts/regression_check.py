"""
Regression checker: compare current benchmark results against a stored baseline.

Usage:
    python -m scripts.regression_check

Stores a baseline on first run, then checks for regressions on subsequent runs.

A regression is defined as: a script that was previously fixed (passing) now fails,
or a script that was previously failing now passes (which requires manual review).
"""

import json
from pathlib import Path

BASELINE_PATH = Path("outputs/benchmark_results/baseline.json")
RESULTS_DIR = Path("outputs/benchmark_results")


def main():
    result_files = sorted(RESULTS_DIR.glob("benchmark_*.json"))
    if not result_files:
        print("No benchmark results found. Run scripts/benchmark.py first.")
        return

    latest = result_files[-1]
    current = json.loads(latest.read_text())
    current_results = {r["script"]: r["fixed"] for r in current["results"]}

    if not BASELINE_PATH.exists():
        # No baseline – save current as baseline
        BASELINE_PATH.write_text(latest.read_text())
        print("Baseline saved. Run again after a code change to check for regressions.")
        return

    baseline = json.loads(BASELINE_PATH.read_text())
    baseline_results = {r["script"]: r["fixed"] for r in baseline["results"]}

    regressions = []
    improvements = []
    for script, was_fixed in baseline_results.items():
        is_fixed = current_results.get(script)
        if is_fixed is None:
            print(f"  ⚠ {script} present in baseline but missing from current run.")
            continue
        if was_fixed and not is_fixed:
            regressions.append(script)
        elif not was_fixed and is_fixed:
            improvements.append(script)

    if regressions:
        print(f"\nREGRESSIONS ({len(regressions)}): previously fixed scripts that now fail:")
        for s in regressions:
            print(f"  ✗ {s}")
    else:
        print("✓ No regressions detected.")

    if improvements:
        print(f"\nNEWLY FIXED ({len(improvements)}): scripts that now pass (were failing before):")
        for s in improvements:
            print(f"  ✓ {s} (update baseline after manual review)")

    if not regressions and not improvements:
        print("All scripts have the same fix status as the baseline.")

    print(f"\nCurrent results file: {latest.name}")
    print(f"Baseline file: {BASELINE_PATH.name}")
    print("If current run is acceptable, update baseline: "
          f"cp {latest} {BASELINE_PATH}")


if __name__ == "__main__":
    main()