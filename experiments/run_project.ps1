# ================================================================
# WavKAN-CL: Full Experiment Pipeline (Publication-Ready)
# ================================================================
# This script runs the complete experiment pipeline:
#   1. Data Processing (PTB-XL parallel)
#   2. Multi-seed Training (5 seeds × 5 models)
#   3. Robustness Testing
#   4. Few-Shot Adaptation
#   5. Statistical Significance Tests
#   6. Visualization (t-SNE, Confusion Matrices, Plots)
# ================================================================

$ErrorActionPreference = "Continue"
$seeds = @(42, 123, 456, 789, 2024)
$models = @("wavkan", "resnet", "vit", "spline_kan", "mlp")

Write-Host "============================================="
Write-Host "  WavKAN-CL EXPERIMENT PIPELINE"
Write-Host "  Start Time: $(Get-Date)"
Write-Host "============================================="

# ---------------------------------------------------------------
# 1. Data Processing
# ---------------------------------------------------------------
if (-not (Test-Path "data/ptbxl_processed.csv")) {
    Write-Host "`n--- Step 1: Processing Data (Parallel) ---"
    python src/process_ptbxl_parallel.py
} else {
    Write-Host "`n--- Step 1: Data already processed. Skipping. ---"
}

# ---------------------------------------------------------------
# 2. Multi-Seed Training (50 Epochs)
# ---------------------------------------------------------------
foreach ($m in $models) {
    foreach ($seed in $seeds) {
        if (-not (Test-Path "experiments/${m}_seed${seed}.pth")) {
            Write-Host "--- Step 2: Training $m seed=$seed ---"
            python -m src.train --model $m --epochs 50 --seed $seed --num_workers 4 --batch_size 64
            if (Test-Path "experiments/${m}_endpoint.pth") {
                Copy-Item "experiments/${m}_endpoint.pth" "experiments/${m}_seed${seed}.pth"
            }
            if (Test-Path "experiments/${m}_history.csv") {
                Copy-Item "experiments/${m}_history.csv" "experiments/${m}_history_seed${seed}.csv"
            }
        } else {
            Write-Host "--- Step 2: $m seed=$seed already trained. Skipping. ---"
        }
    }
}

# ---------------------------------------------------------------
# 3. Robustness Testing (Noise Stress Test per seed)
# ---------------------------------------------------------------
$robust_models = @("wavkan", "resnet", "vit", "spline_kan")
foreach ($m in $robust_models) {
    foreach ($seed in $seeds) {
        if (-not (Test-Path "experiments/robustness_${m}_seed${seed}.csv")) {
            $w = "experiments/${m}_seed${seed}.pth"
            if (Test-Path $w) {
                # Need to copy seed weights to endpoint for test script
                Copy-Item $w "experiments/${m}_endpoint.pth" -Force
                Write-Host "--- Step 3: Noise Test $m seed=$seed ---"
                python -m src.test_robustness --model $m
                if (Test-Path "experiments/robustness_${m}.csv") {
                    Copy-Item "experiments/robustness_${m}.csv" "experiments/robustness_${m}_seed${seed}.csv"
                }
            }
        } else {
            Write-Host "--- Step 3: Noise Test $m seed=$seed already done. Skipping. ---"
        }
    }
}

# ---------------------------------------------------------------
# 4. Few-Shot Adaptation (per seed)
# ---------------------------------------------------------------
foreach ($m in $models) {
    foreach ($seed in $seeds) {
        if (-not (Test-Path "experiments/fewshot_${m}_seed${seed}.csv")) {
            $w = "experiments/${m}_seed${seed}.pth"
            if (Test-Path $w) {
                Copy-Item $w "experiments/${m}_endpoint.pth" -Force
                Write-Host "--- Step 4: Few-Shot $m seed=$seed ---"
                python -m src.test_fewshot --model $m
                if (Test-Path "experiments/fewshot_${m}.csv") {
                    Copy-Item "experiments/fewshot_${m}.csv" "experiments/fewshot_${m}_seed${seed}.csv"
                }
            }
        } else {
            Write-Host "--- Step 4: Few-Shot $m seed=$seed already done. Skipping. ---"
        }
    }
}

# ---------------------------------------------------------------
# 5. SSL Pre-training + Evaluation
# ---------------------------------------------------------------
$ssl_models = @("wavkan", "spline_kan")
foreach ($m in $ssl_models) {
    $ssl_path = "experiments/ssl/${m}_pretrained.pth"
    if (-not (Test-Path $ssl_path)) {
        Write-Host "--- Step 5a: SSL Pre-training $m (300 epochs) ---"
        python -m src.train_ssl --model $m --epochs 300 --batch_size 256 --lr 1e-3
    } else {
        Write-Host "--- Step 5a: SSL Pre-training $m already done. Skipping. ---"
    }
    
    if (Test-Path $ssl_path) {
        if (-not (Test-Path "experiments/fewshot_${m}_ssl.csv")) {
            Write-Host "--- Step 5b: SSL Few-Shot $m ---"
            python -m src.test_fewshot --model $m --pretrained_path $ssl_path
            if (Test-Path "experiments/fewshot_${m}.csv") {
                Copy-Item "experiments/fewshot_${m}.csv" "experiments/fewshot_${m}_ssl.csv"
            }
        }
        
        if (-not (Test-Path "experiments/fewshot_${m}_ssl_linear_probe.csv")) {
            Write-Host "--- Step 5c: SSL Linear Probe $m ---"
            python -m src.test_fewshot --model $m --pretrained_path $ssl_path --linear_probe
            if (Test-Path "experiments/fewshot_${m}.csv") {
                Copy-Item "experiments/fewshot_${m}.csv" "experiments/fewshot_${m}_ssl_linear_probe.csv"
            }
        }
    }
}

# ---------------------------------------------------------------
# 6. Aggregate Results
# ---------------------------------------------------------------
Write-Host "--- Step 6: Aggregating Results ---"
python -m src.aggregate_results

# ---------------------------------------------------------------
# 7. Statistical Significance Testing
# ---------------------------------------------------------------
Write-Host "--- Step 7: Statistical Significance Tests ---"
python -m src.statistical_tests

# ---------------------------------------------------------------
# 8. Generate Visualizations
# ---------------------------------------------------------------
Write-Host "--- Step 8a: Analysis Plots ---"
python -m src.plot_results

Write-Host "--- Step 8b: t-SNE Visualizations ---"
python -m src.visualize_tsne

Write-Host "--- Step 8c: Confusion Matrices ---"
python -m src.visualize_analysis

Write-Host "`n============================================="
Write-Host "  PIPELINE COMPLETE"
Write-Host "  End Time: $(Get-Date)"
Write-Host "============================================="
