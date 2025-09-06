#!/usr/bin/env python3
"""
Script to create a GitHub project board v2 and add issues to it.
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

def create_project_v2(title="Trading Algorithm Feature Board", owner="grich88"):
    """Create a GitHub project v2."""
    cmd = [
        "gh", "api", "graphql",
        "-f", f'query=mutation {{ createProjectV2(input: {{ title: "{title}", ownerId: "MDQ6VXNlcjczNTAxNzA0" }}) {{ projectV2 {{ id, number }} }} }}'
    ]
    result = run_command(cmd)
    if result:
        data = json.loads(result)
        project_id = data.get("data", {}).get("createProjectV2", {}).get("projectV2", {}).get("id")
        project_number = data.get("data", {}).get("createProjectV2", {}).get("projectV2", {}).get("number")
        if project_id:
            print(f"Created project board v2: ID={project_id}, Number={project_number}")
            return project_id, project_number
    return None, None

def create_project_fields(project_id):
    """Create a custom field for categorization."""
    # Create a single select field for status
    cmd = [
        "gh", "api", "graphql",
        "-f", f'query=mutation {{ createProjectV2Field(input: {{ projectId: "{project_id}", dataType: SINGLE_SELECT, name: "Status" }}) {{ projectV2Field {{ id, name }} }} }}'
    ]
    result = run_command(cmd)
    if result:
        data = json.loads(result)
        field_id = data.get("data", {}).get("createProjectV2Field", {}).get("projectV2Field", {}).get("id")
        if field_id:
            print(f"Created Status field: ID={field_id}")
            return field_id
    return None

def create_status_options(project_id, field_id):
    """Create status options for the field."""
    statuses = [
        "Loose Ideas",
        "Backlog",
        "ToDo",
        "In Progress",
        "In Review",
        "Done"
    ]
    
    option_ids = {}
    
    for status in statuses:
        cmd = [
            "gh", "api", "graphql",
            "-f", f'query=mutation {{ createProjectV2FieldOption(input: {{ projectId: "{project_id}", fieldId: "{field_id}", name: "{status}" }}) {{ projectV2FieldOption {{ id, name }} }} }}'
        ]
        result = run_command(cmd)
        if result:
            data = json.loads(result)
            option_id = data.get("data", {}).get("createProjectV2FieldOption", {}).get("projectV2FieldOption", {}).get("id")
            if option_id:
                option_ids[status] = option_id
                print(f"Created status option: {status} with ID={option_id}")
        
        # Wait a bit to avoid rate limiting
        time.sleep(1)
    
    return option_ids

def add_issues_to_project(project_id, repo="grich88/Trading"):
    """Add all issues from the repository to the project."""
    # Get all issues
    cmd = [
        "gh", "issue", "list",
        "--repo", repo,
        "--json", "number,title",
        "--limit", "100"
    ]
    result = run_command(cmd)
    if result:
        issues = json.loads(result)
        issue_ids = {}
        
        for issue in issues:
            issue_number = issue["number"]
            title = issue["title"]
            
            # Add issue to project
            cmd = [
                "gh", "api", "graphql",
                "-f", f'query=mutation {{ addProjectV2ItemById(input: {{ projectId: "{project_id}", contentId: "I_kwDOLJLLOs5qwrZl" }}) {{ item {{ id }} }} }}'
            ]
            result = run_command(cmd)
            if result:
                data = json.loads(result)
                item_id = data.get("data", {}).get("addProjectV2ItemById", {}).get("item", {}).get("id")
                if item_id:
                    issue_ids[issue_number] = item_id
                    print(f"Added issue #{issue_number} ({title}) to project")
            
            # Wait a bit to avoid rate limiting
            time.sleep(1)
        
        return issue_ids
    return {}

def set_issue_status(project_id, item_id, field_id, option_id):
    """Set the status of an issue in the project."""
    cmd = [
        "gh", "api", "graphql",
        "-f", f'query=mutation {{ updateProjectV2ItemFieldValue(input: {{ projectId: "{project_id}", itemId: "{item_id}", fieldId: "{field_id}", value: {{ singleSelectOptionId: "{option_id}" }} }}) {{ projectV2Item {{ id }} }} }}'
    ]
    result = run_command(cmd)
    if result:
        data = json.loads(result)
        updated_item_id = data.get("data", {}).get("updateProjectV2ItemFieldValue", {}).get("projectV2Item", {}).get("id")
        if updated_item_id:
            return True
    return False

def main():
    """Create project board and add issues to it."""
    # Check if issue_map.json exists
    if not os.path.exists("issue_map.json"):
        print("issue_map.json not found. Please run create_issues.py first.")
        return
    
    # Load issue map
    with open("issue_map.json", "r") as f:
        issue_map = json.load(f)
    
    # Create project board v2
    project_id, project_number = create_project_v2()
    if not project_id:
        print("Failed to create project board.")
        return
    
    # Create status field
    field_id = create_project_fields(project_id)
    if not field_id:
        print("Failed to create status field.")
        return
    
    # Create status options
    option_ids = create_status_options(project_id, field_id)
    if not option_ids:
        print("Failed to create status options.")
        return
    
    # Add issues to project
    issue_ids = add_issues_to_project(project_id)
    if not issue_ids:
        print("Failed to add issues to project.")
        return
    
    # Set issue statuses
    for issue_number, category in issue_map.items():
        if issue_number in issue_ids and category in option_ids:
            item_id = issue_ids[issue_number]
            option_id = option_ids[category]
            success = set_issue_status(project_id, item_id, field_id, option_id)
            if success:
                print(f"Set issue #{issue_number} status to {category}")
            else:
                print(f"Failed to set issue #{issue_number} status to {category}")
            
            # Wait a bit to avoid rate limiting
            time.sleep(1)
    
    print(f"Project board setup complete! View at https://github.com/users/grich88/projects/{project_number}")

if __name__ == "__main__":
    main()
