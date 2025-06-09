import json
from bs4 import BeautifulSoup
from collections import defaultdict
from datetime import datetime

def clean_text(text):
    # Remove HTML tags and unnecessary newlines
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text()
    return ' '.join(text.strip().split())

def group_posts_by_topic(posts):
    topics = defaultdict(list)
    metadata = {}

    for post in posts:
        topic_id = post["topic_id"]
        topics[topic_id].append(post)
        metadata[topic_id] = {
            "title": post["topic_title"],
            "url": f"https://discourse.onlinedegree.iitm.ac.in/t/{post['topic_title'].replace(' ', '-').lower()}/{topic_id}",
            "date": post["created_at"][:10]  # Just the date
        }

    cleaned_data = []

    for topic_id, thread_posts in topics.items():
        # Sort by post_number
        thread_posts = sorted(thread_posts, key=lambda x: x["post_number"])

        # Build combined conversation text
        text = ""
        for i, post in enumerate(thread_posts):
            author = post["author"]
            body = clean_text(post["content"])
            prefix = "Student" if i == 0 else f"Reply {i}"
            text += f"{prefix}: {body}\n"

        # Only keep posts within allowed time range
        if "2025-01-01" <= metadata[topic_id]["date"] <= "2025-04-14":
            cleaned_data.append({
                "title": metadata[topic_id]["title"],
                "url": metadata[topic_id]["url"],
                "text": text.strip(),
                "date": metadata[topic_id]["date"]
            })

    return cleaned_data

# === USAGE ===
with open("discourse_posts.json", "r", encoding="utf-8") as f:
    raw_posts = json.load(f)

cleaned_threads = group_posts_by_topic(raw_posts)

# Save the cleaned data
with open("cleaned_discourse_threads.json", "w", encoding="utf-8") as f:
    json.dump(cleaned_threads, f, indent=2)
