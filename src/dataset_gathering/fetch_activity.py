import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import praw

CUTOFFS_FILE = Path("data/cutoffs.json")
OUTPUT = Path("data/reddit_activity.jsonl")

reddit = praw.Reddit(
    client_id=os.environ["REDDIT_CLIENT_ID"],
    client_secret=os.environ["REDDIT_CLIENT_SECRET"],
    user_agent="linux:reddit.user.collector:v1.0.0",
)

cutoffs = json.loads(CUTOFFS_FILE.read_text())

already_done = set()
if OUTPUT.exists():
    for line in OUTPUT.read_text().splitlines():
        if line.strip():
            already_done.add(json.loads(line)["username"])


def ts(dt_str: str) -> float:
    return datetime.fromisoformat(dt_str).replace(tzinfo=timezone.utc).timestamp()


def serialize_comment(comment) -> dict:
    return {
        "id": comment.id,
        "author": str(comment.author) if comment.author else None,
        "body": comment.body,
        "created_utc": comment.created_utc,
        "score": comment.score,
        "replies": [serialize_comment(r) for r in comment.replies if hasattr(r, "body")],
    }


def fetch_thread(submission) -> dict:
    submission.comments.replace_more(limit=0)
    return {
        "id": submission.id,
        "title": submission.title,
        "author": str(submission.author) if submission.author else None,
        "selftext": submission.selftext,
        "url": submission.url,
        "subreddit": str(submission.subreddit),
        "created_utc": submission.created_utc,
        "score": submission.score,
        "comments": [serialize_comment(c) for c in submission.comments],
    }


with OUTPUT.open("a") as out:
    for username, cutoff_str in cutoffs.items():
        if username in already_done:
            print(f"Skip {username} (already done)")
            continue

        cutoff_ts = ts(cutoff_str)
        print(f"Fetching {username} (cutoff: {cutoff_str})")

        try:
            redditor = reddit.redditor(username)
            user_comments = [c for c in redditor.comments.new(limit=None) if c.created_utc <= cutoff_ts]
            user_posts = [p for p in redditor.submissions.new(limit=None) if p.created_utc <= cutoff_ts]
        except Exception as e:
            print(f"  Error fetching user items: {e}")
            continue

        records = []

        for post in user_posts:
            try:
                thread = fetch_thread(post)
                records.append({"type": "post", "user_item_id": post.id, "thread": thread})
                time.sleep(0.5)
            except Exception as e:
                print(f"  Error fetching thread for post {post.id}: {e}")

        for comment in user_comments:
            try:
                submission = comment.submission
                thread = fetch_thread(submission)
                records.append({"type": "comment", "user_item_id": comment.id, "thread": thread})
                time.sleep(0.5)
            except Exception as e:
                print(f"  Error fetching thread for comment {comment.id}: {e}")

        record = {
            "username": username,
            "cutoff": cutoff_str,
            "activity": records,
        }
        out.write(json.dumps(record) + "\n")
        print(f"  {len(user_posts)} posts, {len(user_comments)} comments → {len(records)} threads")
        time.sleep(1)
