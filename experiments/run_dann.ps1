# ================================================================
# DANN Baseline: Complete Training + Evaluation Pipeline
# ================================================================
# Trains DANN across 5 seeds, then runs few-shot, robustness,
# and zero-shot evaluation for each seed.
#
# Usage (on friend's GPU): 
#   powershell -ExecutionPolicy Bypass -File .\experiments\run_dann.ps1
# ================================================================

$ErrorActionPreference = "Stop"
$seeds = @(42, 123, 456, 789, 2024)

Write-Host "============================================="
Write-Host "  DANN BASELINE TRAINING PIPELINE"
Write-Host "  Start Time: $(Get-Date)"
Write-Host "============================================="

# --- 1. Train DANN across 5 seeds ---
foreach ($seed in $seeds) {
    $ckpt = "experiments/dann_seed${seed}.pth"
    if (-not (Test-Path $ckpt)) {
        Write-Host "`n--- Training DANN seed=$seed ---"
        python -m src.train_dann --epochs 50 --seed $seed --num_workers 0 --batch_size 64
    } else {
        Write-Host "--- DANN seed=$seed already exists. Skipping. ---"
    }
}

# --- 2. Few-Shot Evaluation ---
foreach ($seed in $seeds) {
    $fs = "experiments/fewshot_dann_seed${seed}.csv"
    $ckpt = "experiments/dann_seed${seed}.pth"
    if ((-not (Test-Path $fs)) -and (Test-Path $ckpt)) {
        Write-Host "`n--- Few-shot evaluation DANN seed=$seed ---"
        Copy-Item $ckpt "experiments/dann_endpoint.pth" -Force
        python -m src.test_fewshot --model dann
        if (Test-Path "experiments/fewshot_dann.csv") {
            Copy-Item "experiments/fewshot_dann.csv" $fs
        }
    }
}

# --- 3. Robustness Evaluation ---
foreach ($seed in $seeds) {
    $rb = "experiments/robustness_dann_seed${seed}.csv"
    $ckpt = "experiments/dann_seed${seed}.pth"
    if ((-not (Test-Path $rb)) -and (Test-Path $ckpt)) {
        Write-Host "`n--- Robustness test DANN seed=$seed ---"
        Copy-Item $ckpt "experiments/dann_endpoint.pth" -Force
        python -m src.test_robustness --model dann
        if (Test-Path "experiments/robustness_dann.csv") {
            Copy-Item "experiments/robustness_dann.csv" $rb
        }
    }
}

Write-Host "`n============================================="
Write-Host "  DANN PIPELINE COMPLETE"
Write-Host "  End Time: $(Get-Date)"
Write-Host "============================================="
