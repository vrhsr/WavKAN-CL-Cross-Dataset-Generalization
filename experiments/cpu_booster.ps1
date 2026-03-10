# ================================================================
# CPU Maximum Performance Enforcer Daemon
# ================================================================
# This script continuously monitors for any running Python processes
# and forces their CPU priority to 'RealTime' (the absolute maximum
# priority in Windows, above all background OS tasks).
# It ensures all 35 sequential ablation runs get maximum power.

Write-Host "Starting CPU Maximum Performance Enforcer..."
Write-Host "Monitoring every 5 seconds..."

while ($true) {
    $processes = Get-Process python -ErrorAction SilentlyContinue
    if ($processes) {
        foreach ($p in $processes) {
            # Force to all available CPU cores (Processor Affinity)
            try {
                $cores = (Get-WmiObject Win32_Processor -ErrorAction SilentlyContinue | Measure-Object -Property NumberOfLogicalProcessors -Sum).Sum
                if ($cores) {
                    $affinityMask = [intptr]([Math]::Pow(2, $cores) - 1)
                    if ($p.ProcessorAffinity -ne $affinityMask) {
                        $p.ProcessorAffinity = $affinityMask
                    }
                }
            } catch {}

            # Elevate to absolute max priority
            if ($p.PriorityClass -ne "RealTime") {
                try {
                    $p.PriorityClass = "RealTime"
                    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Elevated Python PID $($p.Id) to RealTime and ALL CORES"
                } catch {
                }
            }
        }
    }
    Start-Sleep -Seconds 5
}
