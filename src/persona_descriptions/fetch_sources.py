import json
import os
import time
import praw
from pathlib import Path

INPUT = Path("data/redditmetis.jsonl")
CACHE = Path("data/comment_cache.json")
DELAY = 0.65  # ~92 req/min, safely under the 100/min limit

reddit = praw.Reddit(
    client_id=os.environ["REDDIT_CLIENT_ID"],
    client_secret=os.environ["REDDIT_CLIENT_SECRET"],
    user_agent="linux:reddit.user.collector:v1.0.0",
)

# (section, key, min_count)
FETCH_FIELDS = [
    ("attributes", "data", 1),
    ("family_members", "data", 1),
    ("favorites", "data", 2),
    ("possessions", "data_extra", 3),
]


def comment_id_from_url(url):
    return url.rstrip("/").split("/")[-1]


def collect_ids(records):
    ids = {}  # comment_id -> url (deduplicated, one source per item)
    for record in records:
        synopsis = record["result"].get("synopsis", {})
        for section, key, min_count in FETCH_FIELDS:
            for item in synopsis.get(section, {}).get(key, []):
                if item["count"] >= min_count and item.get("sources"):
                    url = item["sources"][0]
                    ids[comment_id_from_url(url)] = url
    return ids


def main():
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    print(f"Cache: {len(cache)} entries")

    records = []
    for line in open(INPUT):
        line = line.strip()
        if line:
            r = json.loads(line)
            if r.get("exists"):
                records.append(r)

    all_ids = collect_ids(records)
    to_fetch = {cid: url for cid, url in all_ids.items() if cid not in cache}
    print(f"To fetch: {len(to_fetch)} (skipping {len(all_ids) - len(to_fetch)} cached)")

    for i, comment_id in enumerate(to_fetch):
        try:
            body = reddit.comment(id=comment_id).body
            cache[comment_id] = body if body not in ("[deleted]", "[removed]") else None
        except Exception as e:
            print(f"  [{i+1}] error {comment_id}: {e}")
            cache[comment_id] = None

        if (i + 1) % 50 == 0:
            CACHE.write_text(json.dumps(cache))
            print(f"  [{i+1}] checkpoint saved")

        time.sleep(DELAY)

    CACHE.write_text(json.dumps(cache))
    fetched = sum(1 for v in cache.values() if v)
    print(f"Done. {fetched}/{len(cache)} non-null entries in cache.")


if __name__ == "__main__":
    main()
