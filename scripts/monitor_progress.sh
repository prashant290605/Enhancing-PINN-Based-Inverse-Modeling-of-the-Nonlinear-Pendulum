#!/bin/bash

# Monitor progress of the full pipeline run

echo "Monitoring pipeline progress..."
echo "================================"
echo ""

# Check if run is active
if pgrep -f "run_all.sh" > /dev/null; then
    echo "✓ Pipeline is RUNNING"
else
    echo "✗ Pipeline is NOT running"
fi

echo ""
echo "Current status:"
echo ""

# Check baseline
if [ -d "outputs/baseline" ]; then
    echo "✓ [1/5] Baseline complete"
else
    echo "⏳ [1/5] Baseline in progress..."
fi

# Check inverse no pass
if [ -d "outputs/inverse_no_pass" ]; then
    echo "✓ [2/5] Inverse (no passivity) complete"
else
    echo "⏳ [2/5] Inverse (no passivity) in progress..."
fi

# Check inverse with pass
if [ -d "outputs/inverse_with_pass" ]; then
    echo "✓ [3/5] Inverse (with passivity) complete"
else
    echo "⏳ [3/5] Inverse (with passivity) in progress..."
fi

# Check ensemble
if [ -d "outputs/ensemble" ]; then
    ENSEMBLE_DIRS=$(ls -d outputs/ensemble/*/ 2>/dev/null | wc -l)
    if [ "$ENSEMBLE_DIRS" -gt 0 ]; then
        echo "✓ [4/5] Ensemble complete"
    else
        echo "⏳ [4/5] Ensemble in progress..."
    fi
else
    echo "⏳ [4/5] Ensemble in progress..."
fi

# Check mini grid
if [ -f "outputs/mini_grid/summary.csv" ]; then
    echo "✓ [5/5] Mini grid complete"
else
    echo "⏳ [5/5] Mini grid in progress..."
fi

echo ""
echo "Log files:"
for log in outputs/*.log; do
    if [ -f "$log" ]; then
        SIZE=$(wc -l < "$log")
        echo "  - $(basename $log): $SIZE lines"
    fi
done

echo ""
echo "Latest log activity:"
if [ -f "outputs/run_full.log" ]; then
    echo "---"
    tail -5 outputs/run_full.log
    echo "---"
fi

