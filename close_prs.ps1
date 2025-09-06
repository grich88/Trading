# Close all open PRs from 60 to 76
$prs = 60..76

foreach ($pr in $prs) {
    Write-Host "Closing PR #$pr..."
    gh pr close $pr
}

Write-Host "All PRs closed."
