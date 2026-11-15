import shutil
import subprocess
import sys
import json
import csv
from pathlib import Path
import torch

PROJECT_ROOT = Path(r"D:\MajorProject")
ORIG_DIR = PROJECT_ROOT / "results" / "full_training_15ep" / "unet_resnet34"
SMOKE_DIR = PROJECT_ROOT / "results" / "smoke_test_resume_unet_resnet34"

def run_smoke_test():
    print("=================================================================")
    print("GATE 5D RUN 2: RESUME SMOKE TEST (EPOCH 1 -> EPOCH 2 -> EPOCH 3)")
    print("=================================================================")

    # 1. Prepare isolated sandbox
    if SMOKE_DIR.exists():
        shutil.rmtree(SMOKE_DIR)
    SMOKE_DIR.mkdir(parents=True, exist_ok=True)

    # Copy existing verified Epoch 1 checkpoints and history
    shutil.copy2(ORIG_DIR / "final_model.pt", SMOKE_DIR / "final_model.pt")
    shutil.copy2(ORIG_DIR / "history.csv", SMOKE_DIR / "history.csv")
    print(f"Prepared sandbox at: {SMOKE_DIR}")
    print(f"Copied Epoch 1 checkpoint ({SMOKE_DIR / 'final_model.pt'}) and history.csv.")

    # 2. Launch train_scaled.py with resume up to epoch 3, 2 batches per epoch
    python_exe = sys.executable
    cmd = [
        python_exe,
        "-u",
        str(PROJECT_ROOT / "src" / "training" / "train_scaled.py"),
        "--model", "unet_resnet34",
        "--results-dir", str(SMOKE_DIR),
        "--resume",
        "--epochs", "3",
        "--batch-size", "2",
        "--base-lr", "0.001",
        "--warmup-epochs", "2",
        "--max-grad-norm", "1.0",
        "--seed", "42",
        "--max-batches-per-epoch", "2",
    ]

    print("\nExecuting command:")
    print(" ".join(cmd))
    print("\n--- Subprocess Output ---")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    output_lines = []
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        output_lines.append(line)

    proc.wait()
    print(f"\n--- Subprocess exited with code {proc.returncode} ---")
    assert proc.returncode == 0, f"Smoke test failed with return code {proc.returncode}!"

    # 3. Verify Sandbox Artifacts
    print("\n--- Verifying Smoke Test Outputs ---")
    smoke_history_path = SMOKE_DIR / "history.csv"
    smoke_final_path = SMOKE_DIR / "final_model.pt"
    smoke_best_path = SMOKE_DIR / "best_model.pt"
    smoke_summary_path = SMOKE_DIR / "summary.json"

    assert smoke_history_path.exists(), "history.csv missing in smoke test!"
    with open(smoke_history_path, "r", encoding="utf-8") as f:
        history = list(csv.DictReader(f))

    print(f"history.csv rows count: {len(history)}")
    for r in history:
        print(f"  Epoch {r['epoch']}: Mean Dice={float(r['mean_dice']):.4f}, Train Loss={float(r['train_loss']):.4f}, LR_Enc={r['learning_rate_encoder']}, LR_Dec={r['learning_rate_primary']}")

    assert len(history) == 3, f"Expected 3 history rows (epochs 1, 2, 3), got {len(history)}"
    assert [int(r["epoch"]) for r in history] == [1, 2, 3], "Epoch sequence in history is not [1, 2, 3]!"

    # Check that Epoch 2 had warmup LRs [0.000055, 0.000550]
    ep2_row = history[1]
    assert float(ep2_row["learning_rate_encoder"]) == 5.5e-5, f"Ep 2 Enc LR: {ep2_row['learning_rate_encoder']}"
    assert float(ep2_row["learning_rate_primary"]) == 5.5e-4, f"Ep 2 Dec LR: {ep2_row['learning_rate_primary']}"

    # Check that Epoch 3 had peak warmup LRs [0.000100, 0.001000]
    ep3_row = history[2]
    assert float(ep3_row["learning_rate_encoder"]) == 1e-4, f"Ep 3 Enc LR: {ep3_row['learning_rate_encoder']}"
    assert float(ep3_row["learning_rate_primary"]) == 1e-3, f"Ep 3 Dec LR: {ep3_row['learning_rate_primary']}"

    # Verify Checkpoint on CPU
    assert smoke_final_path.exists(), "final_model.pt missing in smoke test!"
    ckpt = torch.load(smoke_final_path, map_location="cpu")
    print(f"\nFinal Checkpoint Audit (CPU):")
    print(f"  Epoch in checkpoint: {ckpt['epoch']}")
    assert ckpt["epoch"] == 3, f"Checkpoint epoch expected 3, got {ckpt['epoch']}"
    print(f"  Scheduler milestones: {ckpt['scheduler_state_dict']['_milestones']}")
    print(f"  Scheduler last_epoch: {ckpt['scheduler_state_dict']['last_epoch']}")
    print(f"  Scheduler _last_lr: {ckpt['scheduler_state_dict']['_last_lr']}")

    # Clean up sandbox
    shutil.rmtree(SMOKE_DIR)
    print(f"\nCleaned up sandbox {SMOKE_DIR}.")
    print("\n=================================================================")
    print("RESUME SMOKE TEST PASSED 100%! READY FOR PRODUCTION RESUME.")
    print("=================================================================")

if __name__ == "__main__":
    run_smoke_test()
