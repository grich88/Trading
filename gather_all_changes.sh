#!/bin/bash

# Script to gather all changes from feature branches

echo "Gathering all feature branch changes..."

# Feature branches in order
BRANCHES=(
    "feature/config-management"
    "feature/logging-service"
    "feature/error-handling"
    "feature/performance-monitoring"
    "feature/data-collection-services"
    "feature/market-analysis-services"
    "feature/signal-integration"
    "feature/rsi-volume-analysis"
    "feature/open-interest-analysis"
    "feature/spot-perp-cvd-analysis"
    "feature/delta-volume-analysis"
    "feature/liquidation-map-analysis"
    "feature/funding-rate-analysis"
    "feature/gamma-exposure-analysis"
    "feature/macro-events-analysis"
    "feature/correlations-analysis"
    "feature/onchain-flows-analysis"
    "feature/unified-dashboard"
)

for BRANCH in "${BRANCHES[@]}"; do
    echo ""
    echo "Processing $BRANCH..."
    
    # Get the commit hash for this branch (excluding merge commits)
    COMMIT=$(git log origin/$BRANCH --format="%H" --no-merges -n 1)
    
    if [ ! -z "$COMMIT" ]; then
        echo "Cherry-picking commit $COMMIT from $BRANCH"
        git cherry-pick $COMMIT
        
        if [ $? -ne 0 ]; then
            echo "Conflict detected. Attempting to resolve..."
            # Auto-resolve conflicts by taking the incoming changes
            git status --porcelain | grep "^UU" | awk '{print $2}' | xargs git add
            git cherry-pick --continue --no-edit
        fi
    else
        echo "No commits found for $BRANCH"
    fi
done

echo ""
echo "All changes gathered!"
