"""Gate 6A Remediation: Multi-Chunk Sequential Evaluation Orchestrator.

Runs Val_eval evaluation in isolated sequential subprocesses to guarantee pristine
WDDM GPU driver context across all 105 cases, followed by unified statistical aggregation.

Logs for each chunk and final aggregation are saved to results/gate6a_remediation_v1/logs/.

Workspace: D:\\MajorProject
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON_EXE = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
EVAL_SCRIPT = PROJECT_ROOT / "tools" / "run_remediation_evaluation.py"
LOGS_DIR = PROJECT_ROOT / "results" / "gate6a_remediation_v1" / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 15
TOTAL_CASES = 105
CHUNKS = [(i, min(i + BATCH_SIZE, TOTAL_CASES)) for i in range(0, TOTAL_CASES, BATCH_SIZE)]


def run_chunk(start_idx: int, end_idx: int) -> None:
    print(f"\n================================================================================")
    print(f"=== LAUNCHING CHUNK [{start_idx}:{end_idx}] (Cases {start_idx+1} to {end_idx})")
    print(f"================================================================================", flush=True)

    log_file = LOGS_DIR / f"chunk_{start_idx:03d}_{end_idx:03d}.log"

    cmd = [
        str(PYTHON_EXE),
        str(EVAL_SCRIPT),
        "--start-idx", str(start_idx),
        "--end-idx", str(end_idx),
    ]

    t0 = time.perf_counter()
    with open(log_file, "w", encoding="utf-8") as lf:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(PROJECT_ROOT),
            text=True,
            bufsize=1,
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            lf.write(line)
            lf.flush()
        proc.wait()

    elapsed = time.perf_counter() - t0

    if proc.returncode != 0:
        print(f"\n[ERROR] Chunk [{start_idx}:{end_idx}] failed with return code {proc.returncode}!", flush=True)
        print(f"Check log at {log_file}")
        sys.exit(proc.returncode)

    print(f"\n[SUCCESS] Chunk [{start_idx}:{end_idx}] completed in {elapsed/60.0:.2f} minutes.", flush=True)


def run_aggregation() -> None:
    print(f"\n================================================================================")
    print(f"=== LAUNCHING FINAL AGGREGATION & PROVENANCE AUDIT")
    print(f"================================================================================", flush=True)

    log_file = LOGS_DIR / "aggregation.log"

    cmd = [
        str(PYTHON_EXE),
        str(EVAL_SCRIPT),
        "--aggregate-only",
    ]

    t0 = time.perf_counter()
    with open(log_file, "w", encoding="utf-8") as lf:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(PROJECT_ROOT),
            text=True,
            bufsize=1,
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            lf.write(line)
            lf.flush()
        proc.wait()

    elapsed = time.perf_counter() - t0

    if proc.returncode != 0:
        print(f"\n[ERROR] Aggregation failed with return code {proc.returncode}!", flush=True)
        print(f"Check log at {log_file}")
        sys.exit(proc.returncode)

    print(f"\n[SUCCESS] Full Gate 6A Remediation evaluation aggregated in {elapsed/60.0:.2f} minutes.", flush=True)


def main() -> None:
    t_global = time.perf_counter()
    print("=== Gate 6A Remediation: Multi-Chunk Evaluation Orchestrator ===", flush=True)
    print(f"Total evaluation cases: {TOTAL_CASES} across {len(CHUNKS)} sequential chunks (batch size {BATCH_SIZE})")

    for start_idx, end_idx in CHUNKS:
        run_chunk(start_idx, end_idx)

    run_aggregation()

    total_time = (time.perf_counter() - t_global) / 60.0
    print(f"\n================================================================================")
    print(f"=== GATE 6A REMEDIATION COMPLETE: Total time: {total_time:.2f} minutes")
    print(f"================================================================================", flush=True)


if __name__ == "__main__":
    main()
