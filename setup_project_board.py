#!/usr/bin/env python3
"""
Script to create a GitHub project board and add issues to it.
"""

import subprocess
import json
import time
import os

def run_command(cmd):
    """Run a command and return the output."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e.stderr}")
        return None

def create_project_board(title="Trading Algorithm Feature Board", owner="grich88"):
    """Create a GitHub project board."""
    cmd = [
        "gh", "project", "create",
        "--title", title,
        "--owner", owner
    ]
    result = run_command(cmd)
    if result:
        print(f"Created project board: {result}")
        # Extract project number from URL
        project_number = result.split("/")[-1]
        return project_number
    return None

def create_project_columns(project_number):
    """Create columns in the project board."""
    columns = [
        "Loose Ideas",
        "Backlog",
        "ToDo",
        "In Progress",
        "In Review",
        "Done"
    ]
    
    column_ids = {}
    
    for column in columns:
        cmd = [
            "gh", "api",
            f"projects/{project_number}/columns",
            "--method", "POST",
            "-f", f"name={column}"
        ]
        result = run_command(cmd)
        if result:
            data = json.loads(result)
            column_ids[column] = data["id"]
            print(f"Created column: {column} with ID {data['id']}")
    
    return column_ids

def add_issues_to_columns(issue_map, column_ids):
    """Add issues to the appropriate columns."""
    for issue_number, category in issue_map.items():
        if category in column_ids:
            column_id = column_ids[category]
            cmd = [
                "gh", "api",
                f"projects/columns/{column_id}/cards",
                "--method", "POST",
                "-f", f"content_id={issue_number}",
                "-f", "content_type=Issue"
            ]
            result = run_command(cmd)
            if result:
                print(f"Added issue #{issue_number} to column {category}")
            
            # Wait a bit to avoid rate limiting
            time.sleep(1)

def main():
    """Create project board and add issues to it."""
    # Check if issue_map.json exists
    if not os.path.exists("issue_map.json"):
        print("issue_map.json not found. Please run create_issues.py first.")
        return
    
    # Load issue map
    with open("issue_map.json", "r") as f:
        issue_map = json.load(f)
    
    # Create project board
    project_number = create_project_board(owner="grich88")
    if not project_number:
        print("Failed to create project board.")
        return
    
    # Create columns
    column_ids = create_project_columns(project_number)
    if not column_ids:
        print("Failed to create columns.")
        return
    
    # Add issues to columns
    add_issues_to_columns(issue_map, column_ids)
    
    print("Project board setup complete!")

if __name__ == "__main__":
    main()