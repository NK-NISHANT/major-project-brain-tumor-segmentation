"""Gate 6A Multi-Chunk Sequential Orchestrator.

Runs Val_eval evaluation in isolated sequential subprocesses to guarantee pristine
WDDM GPU driver context across all 105 cases, followed by unified statistical aggregation.

Workspace: D:\\MajorProject
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON_EXE = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
EVAL_SCRIPT = PROJECT_ROOT / "tools" / "run_gate6a_evaluation.py"

BATCH_SIZE = 15
TOTAL_CASES = 105
CHUNKS = [(i, min(i + BATCH_SIZE, TOTAL_CASES)) for i in range(0, TOTAL_CASES, BATCH_SIZE)]


def run_chunk(start_idx: int, end_idx: int) -> None:
    print(f"\n================================================================================")
    print(f"=== LAUNCHING CHUNK [{start_idx}:{end_idx}] (Cases {start_idx+1} to {end_idx})")
    print(f"================================================================================", flush=True)

    cmd = [
        str(PYTHON_EXE),
        str(EVAL_SCRIPT),
        "--start-idx", str(start_idx),
        "--end-idx", str(end_idx),
    ]

    t0 = time.perf_counter()
    res = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    elapsed = time.perf_counter() - t0

    if res.returncode != 0:
        print(f"\n[ERROR] Chunk [{start_idx}:{end_idx}] failed with return code {res.returncode}!", flush=True)
        sys.exit(res.returncode)

    print(f"\n[SUCCESS] Chunk [{start_idx}:{end_idx}] completed in {elapsed/60.0:.2f} minutes.", flush=True)


def run_aggregation() -> None:
    print(f"\n================================================================================")
    print(f"=== LAUNCHING FINAL AGGREGATION & FIGURE GENERATION")
    print(f"================================================================================", flush=True)

    cmd = [
        str(PYTHON_EXE),
        str(EVAL_SCRIPT),
        "--aggregate-only",
    ]

    t0 = time.perf_counter()
    res = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    elapsed = time.perf_counter() - t0

    if res.returncode != 0:
        print(f"\n[ERROR] Aggregation failed with return code {res.returncode}!", flush=True)
        sys.exit(res.returncode)

    print(f"\n[SUCCESS] Full Gate 6A evaluation aggregated in {elapsed/60.0:.2f} minutes.", flush=True)


def main() -> None:
    t_global = time.perf_counter()
    print("=== Gate 6A: Sequential 3-Chunk Evaluation Orchestrator ===", flush=True)

    for start_idx, end_idx in CHUNKS:
        run_chunk(start_idx, end_idx)

    run_aggregation()

    total_time = (time.perf_counter() - t_global) / 60.0
    print(f"\n================================================================================")
    print(f"=== GATE 6A EVALUATION COMPLETE: Total time: {total_time:.2f} minutes")
    print(f"================================================================================", flush=True)


if __name__ == "__main__":
    main()
