# ================================================================
# Ablation Retraining with Conv1DStem Architecture
# ================================================================
# This script retrains all WavKAN ablation variants:
#   - Wavelet type: mexican_hat, morlet
#   - Depth: 2, 3, 4
#   - Hidden dim: 32, 64, 128
# All using the new Conv1DStem architecture (default in WavKANClassifier).
# Each config is trained with 5 random seeds.
# After training, few-shot evaluation is also performed.
# ================================================================

$ErrorActionPreference = "Continue"
$seeds = @(42, 123, 456, 789, 2024)

Write-Host "============================================="
Write-Host "  ABLATION RETRAINING PIPELINE"
Write-Host "  Start Time: $(Get-Date)"
Write-Host "============================================="

# ---------------------------------------------------------------
# 1. Delete old ablation checkpoints (OLD architecture, no Conv1DStem)
# ---------------------------------------------------------------
Write-Host "`n--- Checking for old ablation checkpoints ---"
# Remove-Item experiments/ablation_*_seed*.pth -Force -ErrorAction SilentlyContinue
# Remove-Item experiments/ablation_*_history_seed*.csv -Force -ErrorAction SilentlyContinue
# Remove-Item experiments/fewshot_depth*_seed*.csv -Force -ErrorAction SilentlyContinue
# Remove-Item experiments/fewshot_hdim*_seed*.csv -Force -ErrorAction SilentlyContinue
# Remove-Item experiments/fewshot_wavelet_*_seed*.csv -Force -ErrorAction SilentlyContinue
Write-Host "Old checkpoints preserved (to prevent destructive restart)."

# ---------------------------------------------------------------
# 2. Wavelet Type Ablation (depth=3, hidden_dim=64)
# ---------------------------------------------------------------
$wavelet_types = @("mexican_hat", "morlet")
foreach ($wt in $wavelet_types) {
    foreach ($seed in $seeds) {
        $tag = "ablation_wavelet_${wt}"
        $ckpt = "experiments/${tag}_seed${seed}.pth"
        if (-not (Test-Path $ckpt)) {
            Write-Host "--- Training $tag seed=$seed ---"
            python -m src.train --model wavkan --epochs 50 --seed $seed --num_workers 4 --batch_size 64 --wavelet_type $wt --hidden_dim 64 --depth 3
            if (Test-Path "experiments/wavkan_endpoint.pth") {
                Copy-Item "experiments/wavkan_endpoint.pth" $ckpt
            }
            if (Test-Path "experiments/wavkan_history.csv") {
                Copy-Item "experiments/wavkan_history.csv" "experiments/${tag}_history_seed${seed}.csv"
            }
        } else {
            Write-Host "--- $tag seed=$seed already exists. Skipping. ---"
        }
        
        # Few-shot evaluation
        $fs = "experiments/fewshot_wavelet_${wt}_seed${seed}.csv"
        if ((-not (Test-Path $fs)) -and (Test-Path $ckpt)) {
            Copy-Item $ckpt "experiments/wavkan_endpoint.pth" -Force
            python -m src.test_fewshot --model wavkan
            if (Test-Path "experiments/fewshot_wavkan.csv") {
                Copy-Item "experiments/fewshot_wavkan.csv" $fs
            }
        }
    }
}

# ---------------------------------------------------------------
# 3. Depth Ablation (wavelet=mexican_hat, hidden_dim=64)
# ---------------------------------------------------------------
$depths = @(2, 3, 4)
foreach ($d in $depths) {
    foreach ($seed in $seeds) {
        $tag = "ablation_depth${d}"
        $ckpt = "experiments/${tag}_seed${seed}.pth"
        if (-not (Test-Path $ckpt)) {
            Write-Host "--- Training $tag seed=$seed ---"
            python -m src.train --model wavkan --epochs 50 --seed $seed --num_workers 4 --batch_size 64 --wavelet_type mexican_hat --hidden_dim 64 --depth $d
            if (Test-Path "experiments/wavkan_endpoint.pth") {
                Copy-Item "experiments/wavkan_endpoint.pth" $ckpt
            }
            if (Test-Path "experiments/wavkan_history.csv") {
                Copy-Item "experiments/wavkan_history.csv" "experiments/${tag}_history_seed${seed}.csv"
            }
        } else {
            Write-Host "--- $tag seed=$seed already exists. Skipping. ---"
        }
        
        # Few-shot evaluation
        $fs = "experiments/fewshot_depth${d}_seed${seed}.csv"
        if ((-not (Test-Path $fs)) -and (Test-Path $ckpt)) {
            Copy-Item $ckpt "experiments/wavkan_endpoint.pth" -Force
            python -m src.test_fewshot --model wavkan
            if (Test-Path "experiments/fewshot_wavkan.csv") {
                Copy-Item "experiments/fewshot_wavkan.csv" $fs
            }
        }
    }
}

# ---------------------------------------------------------------
# 4. Hidden Dim Ablation (wavelet=mexican_hat, depth=3)
# ---------------------------------------------------------------
$hdims = @(32, 64, 128)
foreach ($h in $hdims) {
    foreach ($seed in $seeds) {
        $tag = "ablation_hdim${h}"
        $ckpt = "experiments/${tag}_seed${seed}.pth"
        if (-not (Test-Path $ckpt)) {
            Write-Host "--- Training $tag seed=$seed ---"
            python -m src.train --model wavkan --epochs 50 --seed $seed --num_workers 4 --batch_size 64 --wavelet_type mexican_hat --hidden_dim $h --depth 3
            if (Test-Path "experiments/wavkan_endpoint.pth") {
                Copy-Item "experiments/wavkan_endpoint.pth" $ckpt
            }
            if (Test-Path "experiments/wavkan_history.csv") {
                Copy-Item "experiments/wavkan_history.csv" "experiments/${tag}_history_seed${seed}.csv"
            }
        } else {
            Write-Host "--- $tag seed=$seed already exists. Skipping. ---"
        }
        
        # Few-shot evaluation
        $fs = "experiments/fewshot_hdim${h}_seed${seed}.csv"
        if ((-not (Test-Path $fs)) -and (Test-Path $ckpt)) {
            Copy-Item $ckpt "experiments/wavkan_endpoint.pth" -Force
            python -m src.test_fewshot --model wavkan
            if (Test-Path "experiments/fewshot_wavkan.csv") {
                Copy-Item "experiments/fewshot_wavkan.csv" $fs
            }
        }
    }
}

Write-Host "`n============================================="
Write-Host "  ABLATION RETRAINING COMPLETE"
Write-Host "  End Time: $(Get-Date)"
Write-Host "============================================="
