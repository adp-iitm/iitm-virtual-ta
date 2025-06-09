import os
import yaml
from bs4 import BeautifulSoup
from markdown import markdown
import json

def process_markdown_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split YAML frontmatter and markdown body
    if content.startswith('---'):
        parts = content.split('---', 2)
        meta = yaml.safe_load(parts[1])
        md_body = parts[2]
    else:
        meta = {}
        md_body = content

    # Convert markdown body to plain text
    html = markdown(md_body)
    text = BeautifulSoup(html, "html.parser").get_text(separator="\n")
    clean_text = ' '.join(text.strip().split())

    return {
        "title": meta.get("title", "Untitled"),
        "url": meta.get("original_url", ""),
        "text": clean_text,
        "date": meta.get("downloaded_at", "")[:10]  # just the date
    }

# Directory containing .md files
folder = "markdown_files"
cleaned_docs = []

for filename in os.listdir(folder):
    if filename.endswith(".md"):
        full_path = os.path.join(folder, filename)
        cleaned_docs.append(process_markdown_file(full_path))

# Save to JSON
with open("cleaned_website_docs.json", "w", encoding="utf-8") as f:
    json.dump(cleaned_docs, f, indent=2)
