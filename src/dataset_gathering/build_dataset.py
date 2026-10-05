import json
from pathlib import Path

ACTIVITY_FILE = Path("data/reddit_activity.jsonl")
PERSONAS_FILE = Path("data/personas.jsonl")
OUTPUT = Path("data/dataset.jsonl")

personas = {
    json.loads(l)["username"]: json.loads(l)
    for l in PERSONAS_FILE.read_text().splitlines()
    if l.strip()
}

REMOVED = {"[removed]", "[deleted]"}


def invalid_reason(text: str) -> str | None:
    t = (text or "").strip()
    if t in REMOVED:
        return "removed"
    if len(t.split()) < 5:
        return "too short"
    return None


def is_valid(text: str) -> bool:
    return invalid_reason(text) is None


def first_valid_reply(replies: list) -> dict | None:
    valid = [r for r in replies if is_valid(r.get("body", ""))]
    if not valid:
        return None
    return min(valid, key=lambda r: r["created_utc"])


def extract_exchanges(comments: list, user_item_id: str, parent: dict | None = None, parent_is_op: bool = False) -> list[dict]:
    exchanges = []
    for comment in comments:
        body = (comment.get("body") or "").strip()
        is_user = comment["id"] == user_item_id

        if is_user:
            reason = invalid_reason(body)
            if reason:
                exchanges.append({"valid": False, "type": None, "reason": f"user comment {reason}"})
            elif not parent or not is_valid(parent.get("body", "")):
                reply = first_valid_reply(comment.get("replies", []))
                if not reply:
                    exchanges.append({"valid": False, "type": None, "reason": "no parent context and no reply"})
                else:
                    exchanges.append({
                        "valid": True,
                        "user_author": comment["author"] or "[deleted]",
                        "user": body,
                        "reply_author": reply["author"] or "[deleted]",
                        "reply": reply["body"].strip(),
                    })
            else:
                exchange = {
                    "valid": True,
                    "reply_to_op": parent_is_op,
                    "user_author": comment["author"] or "[deleted]",
                    "user": body,
                    "parent_author": parent["author"] or "[deleted]",
                    "parent": parent["body"].strip(),
                }
                reply = first_valid_reply(comment.get("replies", []))
                if reply:
                    exchange["reply_author"] = reply["author"] or "[deleted]"
                    exchange["reply"] = reply["body"].strip()
                exchanges.append(exchange)

        exchanges.extend(extract_exchanges(comment.get("replies", []), user_item_id, parent=comment, parent_is_op=False))
    return exchanges


def build_records(activity: dict, base: dict) -> list[dict]:
    thread = activity["thread"]
    user_item_id = activity["user_item_id"]
    url = thread.get("url") or f"https://www.reddit.com/r/{thread['subreddit']}/comments/{thread['id']}/"
    results = []

    if activity["type"] == "post":
        selftext = (thread.get("selftext") or "").strip()
        reason = invalid_reason(selftext)
        if reason:
            results.append({**base, "url": url, "valid": False, "type": None, "reason": f"post body {reason}", "conversation": None})
            return results
        reply = first_valid_reply(thread["comments"])
        if not reply:
            results.append({**base, "url": url, "valid": False, "type": None, "reason": "no valid reply to post", "conversation": None})
            return results
        conversation = [
            {"author": thread["author"] or "[deleted]", "text": selftext, "is_user": True},
            {"author": reply["author"] or "[deleted]", "text": reply["body"].strip(), "is_user": False},
        ]
        results.append({**base, "url": url, "valid": True, "type": "user_post", "conversation": conversation})

    else:
        post_as_parent = {
            "id": thread["id"],
            "author": thread["author"],
            "body": (thread.get("title", "") + "\n" + thread.get("selftext", "")).strip(),
            "created_utc": thread["created_utc"],
            "replies": [],
        }
        exchanges = extract_exchanges(thread["comments"], user_item_id, parent=post_as_parent, parent_is_op=True)
        if not exchanges:
            results.append({**base, "url": url, "valid": False, "type": None, "reason": "user comment not found in thread", "conversation": None})
            return results
        for exchange in exchanges:
            if not exchange["valid"]:
                results.append({**base, "url": url, "valid": False, "type": None, "reason": exchange["reason"], "conversation": None})
            else:
                turns = []
                if "parent" in exchange:
                    turns.append({"author": exchange["parent_author"], "text": exchange["parent"], "is_user": False})
                turns.append({"author": exchange["user_author"], "text": exchange["user"], "is_user": True})
                if "reply" in exchange:
                    turns.append({"author": exchange["reply_author"], "text": exchange["reply"], "is_user": False})
                    kind = "user_reply_to_op_with_response" if exchange.get("reply_to_op") else "user_reply_with_response"
                else:
                    kind = "user_reply_to_op" if exchange.get("reply_to_op") else "user_reply"
                results.append({**base, "url": url, "valid": True, "type": kind, "conversation": turns})

    return results


records = []
idx = 0

for line in ACTIVITY_FILE.read_text().splitlines():
    if not line.strip():
        continue
    rec = json.loads(line)
    username = rec["username"]
    system_prompt = personas.get(username, {}).get("system_prompt", "")

    for activity in rec["activity"]:
        base = {
            "id": f"thread_{idx:06d}",
            "username": username,
            "system_prompt": system_prompt,
        }
        for record in build_records(activity, base):
            records.append(record)
            idx += 1

OUTPUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n")
valid = sum(1 for r in records if r["valid"])
print(f"Written {len(records)} records ({valid} valid, {len(records) - valid} invalid) to {OUTPUT}")
