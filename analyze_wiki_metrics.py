import os
import re
from pathlib import Path

def analyze_wiki():
    wiki_dir = Path("data/04_wiki")
    if not wiki_dir.exists():
        print("Wiki directory not found.")
        return

    files = list(wiki_dir.glob("*.md"))
    total_links = 0
    total_conflicts = 0
    pages_with_links = 0
    pages_with_conflicts = 0

    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            content = f.read()
            
            # Count internal links like [[Page Name]]
            links = re.findall(r"\[\[(.*?)\]\]", content)
            if links:
                total_links += len(links)
                pages_with_links += 1
                
            # Count contradiction/conflict markers
            conflict_keywords = ["conflict", "contradict", "discrepancy", "differs"]
            has_conflict = any(kw in content.lower() for kw in conflict_keywords)
            if has_conflict:
                total_conflicts += 1
                pages_with_conflicts += 1

    print("=== Wiki Compounding Metrics ===")
    print(f"Total Pages Analyzed: {len(files)}")
    print(f"Total Internal Links: {total_links} (across {pages_with_links} pages)")
    print(f"Total Flagged Conflicts: {total_conflicts} (across {pages_with_conflicts} pages)")
    print(f"Average Links per Page: {total_links / max(1, len(files)):.2f}")

if __name__ == "__main__":
    analyze_wiki()
