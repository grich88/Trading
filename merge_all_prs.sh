#!/bin/bash

# Script to merge all open PRs in order
# Following Master Guide principles: always pull latest develop before each merge

echo "Starting PR merge process..."

# List of PRs to merge in order (from oldest to newest)
PRS=(59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76)

for PR in "${PRS[@]}"; do
    echo ""
    echo "========================================="
    echo "Processing PR #$PR"
    echo "========================================="
    
    # Pull latest develop branch
    echo "Pulling latest develop branch..."
    git pull origin develop
    
    # Merge the PR
    echo "Merging PR #$PR..."
    gh pr merge $PR --squash --delete-branch
    
    if [ $? -eq 0 ]; then
        echo "✓ Successfully merged PR #$PR"
    else
        echo "✗ Failed to merge PR #$PR"
        echo "Stopping script. Please resolve the issue and continue manually."
        exit 1
    fi
    
    # Brief pause to ensure GitHub processes the merge
    sleep 2
done

# Final pull to get all changes
echo ""
echo "Final pull of develop branch..."
git pull origin develop

echo ""
echo "========================================="
echo "All PRs have been successfully merged!"
echo "========================================="
