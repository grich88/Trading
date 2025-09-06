#!/usr/bin/env python3
"""
Script to create GitHub issues from ticket files.
"""

import os
import re
import json
import subprocess
import time
from typing import List, Dict, Any

# Configuration
REPO = "grich88/Trading"
TICKET_FILES = [
    "project_setup_tickets.md",
    "data_collection_tickets.md",
    "analysis_models_tickets.md"
]

def parse_ticket_file(file_path: str) -> List[Dict[str, Any]]:
    """Parse a ticket file and extract ticket information."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Extract the category from the file name
    category = os.path.basename(file_path).replace("_tickets.md", "").replace("_", " ").title()
    
    # Split the content into tickets
    ticket_sections = re.split(r"## Ticket \d+:", content)[1:]
    
    tickets = []
    for section in ticket_sections:
        # Extract ticket title
        title_match = re.search(r"^([^\n]+)", section.strip())
        if not title_match:
            continue
        title = title_match.group(1).strip()
        
        # Extract problem statement
        problem_match = re.search(r"Problem Statement:(.*?)Definition of Done:", section, re.DOTALL)
        if not problem_match:
            continue
        problem = problem_match.group(1).strip()
        
        # Extract definition of done
        dod_match = re.search(r"Definition of Done:(.*?)(?:$|##)", section, re.DOTALL)
        if not dod_match:
            continue
        dod = dod_match.group(1).strip()
        
        # Format the body
        body = f"""## Problem Statement
{problem}

## Definition of Done
{dod}

## Category
{category}
"""
        
        tickets.append({
            "title": title,
            "body": body,
            "category": category
        })
    
    return tickets

def create_github_issue(ticket: Dict[str, Any]) -> int:
    """Create a GitHub issue using gh CLI."""
    title = ticket["title"]
    body = ticket["body"]
    
    # Create a temporary file for the body
    temp_file = f"temp_issue_body_{int(time.time())}.md"
    with open(temp_file, "w", encoding="utf-8") as f:
        f.write(body)
    
    try:
        # Create the issue
        cmd = ["gh", "issue", "create", 
               "--repo", REPO, 
               "--title", title, 
               "--body-file", temp_file]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Extract the issue number from the URL
        issue_url = result.stdout.strip()
        issue_number = int(issue_url.split("/")[-1])
        
        print(f"Created issue #{issue_number}: {title}")
        return issue_number
    except subprocess.CalledProcessError as e:
        print(f"Error creating issue: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return 0
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file):
            os.remove(temp_file)

def main():
    """Main function."""
    all_tickets = []
    
    # Parse all ticket files
    for file_path in TICKET_FILES:
        if os.path.exists(file_path):
            tickets = parse_ticket_file(file_path)
            all_tickets.extend(tickets)
            print(f"Parsed {len(tickets)} tickets from {file_path}")
        else:
            print(f"File not found: {file_path}")
    
    # Create GitHub issues
    issue_map = []
    for ticket in all_tickets:
        issue_number = create_github_issue(ticket)
        if issue_number > 0:
            issue_map.append({
                "title": ticket["title"],
                "category": ticket["category"],
                "issue_number": issue_number
            })
        time.sleep(1)  # Avoid rate limiting
    
    # Save the issue map
    with open("issue_map.json", "w", encoding="utf-8") as f:
        json.dump(issue_map, f, indent=2)
    
    print(f"Created {len(issue_map)} issues")

if __name__ == "__main__":
    main()
