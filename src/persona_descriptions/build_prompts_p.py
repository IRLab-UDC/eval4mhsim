import json
from pathlib import Path

INPUT = Path("data/redditmetis.jsonl")
OUTPUT = Path("data/persona_prompts_p.jsonl")
CACHE = Path("data/comment_cache.json")

SYSTEM = "You are an expert at writing detailed, grounded LLM persona descriptions."

USER_TEMPLATE = """\
Below is a behavioral profile extracted from a user's online activity:

{profile}

Based on this data, write a system prompt in second person (starting with "You are...") \
that describes this person's background, personality, interests, communication style, \
and emotional tendencies. It will be used to make an LLM simulate this person generating content. \
Be specific and grounded in the data. Do not mention usernames or platform names. \
"""

def comment_id_from_url(url):
    return url.rstrip("/").split("/")[-1]


def snippet(text, cache):
    if not text:
        return None
    cid = comment_id_from_url(text) if text.startswith("http") else None
    if cid:
        body = cache.get(cid)
        if body:
            return body.replace("\n", " ").strip()
    return None


def items_with_source(synopsis, section, key="data"):
    return [
        (x["value"], x["count"], x.get("sources", [None])[0])
        for x in synopsis.get(section, {}).get(key, [])
    ]


def extract(record, cache):
    r = record["result"]
    synopsis = r.get("synopsis", {})

    def items(section, key="data"):
        return [(x["value"], x["count"]) for x in synopsis.get(section, {}).get(key, [])]

    identity = [(v, snippet(src, cache)) for v, _, src in items_with_source(synopsis, "attributes")]
    family = [(v, snippet(src, cache)) for v, _, src in items_with_source(synopsis, "family_members")]
    favorites = [
        (v, snippet(src, cache))
        for v, c, src in items_with_source(synopsis, "favorites")
        if c >= 2
    ]
    possessions = [
        (v, snippet(src, cache))
        for v, c, src in sorted(
            items_with_source(synopsis, "possessions", "data_extra"),
            key=lambda x: -x[1]
        )
        if c >= 3
    ][:10]

    interest_keys = ["lifestyle", "gaming", "entertainment", "science", "social science and humanities"]
    interests = {k: sorted(items(k), key=lambda x: -x[1])[:4] for k in interest_keys if items(k)}

    top_subs = r.get("top_subs", {}).get("comment", [])[:6]
    top_subs_by_karma = r.get("top_subs_by_karma", {}).get("comment", [])[:6]

    sent = r.get("sentiments", {})
    examples = []
    for key in ("most_positive_comment", "most_negative_comment"):
        t = sent.get(key, {}).get("text", "")
        if t:
            examples.append(t)
    best = r.get("summary", {}).get("comments", {}).get("best", {}).get("text", "")
    if best and best not in examples:
        examples.append(best)

    n_pos = sent.get("positive", 0)
    n_neg = sent.get("negative", 0)
    n_neu = sent.get("neutral", 0)
    n_total = n_pos + n_neg + n_neu
    sentiment_ratio = {
        "positive": round(n_pos / n_total, 2) if n_total else None,
        "negative": round(n_neg / n_total, 2) if n_total else None,
        "neutral": round(n_neu / n_total, 2) if n_total else None,
    }

    gf = r.get("readability", {}).get("gf-index", 0)
    complexity = (
        "very casual and informal" if gf < 8
        else "conversational" if gf < 12
        else "educated, somewhat formal" if gf < 16
        else "academic and formal"
    )

    common_words = [w["text"] for w in r.get("metrics", {}).get("common_words", [])[:10]]

    return {
        "identity": identity,
        "family": family,
        "favorites": favorites,
        "possessions": possessions,
        "interests": interests,
        "top_subs": top_subs,
        "top_subs_by_karma": top_subs_by_karma,
        "sentiment_ratio": sentiment_ratio,
        "writing_complexity": complexity,
        "common_words": common_words,
        "comment_examples": examples[:3],
    }


def fmt_with_snippet(items):
    parts = []
    for v, snip in items:
        parts.append(f'{v} ("{snip}")' if snip else v)
    return ", ".join(parts)


def format_profile(e):
    lines = []
    if e["identity"]:
        lines.append(f"Identity: {fmt_with_snippet(e['identity'])}")
    if e["family"]:
        lines.append(f"Family: {fmt_with_snippet(e['family'])}")
    if e["favorites"]:
        lines.append(f"Favorites: {fmt_with_snippet(e['favorites'])}")
    if e["possessions"]:
        lines.append(f"Life context: {fmt_with_snippet(e['possessions'])}")
    for category, data in e["interests"].items():
        lines.append(f"Interests ({category}): {', '.join(v for v, _ in data)}")
    if e["top_subs"]:
        lines.append(f"Most active communities: {', '.join(f'{n} ({c})' for n, c in e['top_subs'])}")
    if e["top_subs_by_karma"]:
        lines.append(f"Communities by resonance: {', '.join(f'{n} ({c})' for n, c in e['top_subs_by_karma'])}")
    r = e["sentiment_ratio"]
    if r["positive"] is not None:
        lines.append(f"Sentiment: {r['positive']*100:.0f}% positive, {r['negative']*100:.0f}% negative, {r['neutral']*100:.0f}% neutral")
    lines.append(f"Writing style: {e['writing_complexity']}")
    if e["common_words"]:
        lines.append(f"Frequent words: {', '.join(e['common_words'])}")
    if e["comment_examples"]:
        lines.append("Example comments:")
        for ex in e["comment_examples"]:
            lines.append(f'  - "{ex}"')
    return "\n".join(lines)


def main():
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    if cache:
        print(f"Loaded {len(cache)} cached comments")

    records = []
    with open(INPUT) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("exists"):
                records.append(record)

    with open(OUTPUT, "w") as out:
        for record in records:
            username = record["result"]["username"]
            profile = format_profile(extract(record, cache))
            entry = {
                "username": username,
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": USER_TEMPLATE.format(profile=profile)},
                ],
            }
            out.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} prompts → {OUTPUT}")


if __name__ == "__main__":
    main()
