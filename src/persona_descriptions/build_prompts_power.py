import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

INPUT = Path("data/redditmetis.jsonl")
TRAIN = Path("data/dataset_train.jsonl")
OUTPUT = Path("data/persona_prompts_power.jsonl")
CACHE = Path("data/comment_cache.json")

SYSTEM = "You are an expert at writing detailed, grounded LLM persona descriptions."

USER_TEMPLATE = """\
=== WRITING SAMPLES ===
Below are examples of how this person actually writes on Reddit. \
Each example shows the post they were responding to and their reply:

{samples}

=== BEHAVIORAL PROFILE ===
Background and behavioral data extracted from this person's online activity \
(use this for identity and interests; the writing samples above are the ground truth for style and emotion):

{profile}

=== TASK ===
Write a system prompt in second person (starting with "You are...") \
that an LLM will use to simulate this person writing Reddit replies. \
It must contain three labeled sections:

## 1. Background and Persona
Their strictly evidence-based identity, interests, and values derived from the profile. \
Keep it concise and grounded in the provided data.

## 2. Writing Style Mechanics
Concrete, imitable instructions distilled from the writing samples. \
Detail actionable mechanics: typical reply length, structural habits (how they open, close, and hedge), \
specific punctuation and capitalization quirks, and syntactic patterns. \
Avoid vague labels like 'informal' or 'casual'. Give exact rules so another LLM \
could produce a structurally identical reply. Do not quote the samples directly; generalize the rules.

## 3. Emotional Lexicon Baseline
Distill their emotional vocabulary from the samples using Ekman's 7 basic emotions \
(happiness, sadness, anger, fear, disgust, surprise, contempt) as buckets. \
Instead of abstract psychological descriptions, establish a quantitative lexical baseline. \
Your output must instruct the final LLM to mirror the user's proportional use of emotional words. \
You must include:
- The dominant Ekman categories and an estimate of their proportional frequency in the user's text.
- Which Ekman categories are noticeably absent or suppressed.
- Specific emotional trigger words or go-to phrasing the user relies on for their dominant emotions.
Provide actionable instructions to match this exact vocabulary ratio.

Do not mention usernames or platform names.\
"""

USER_PROFILE_ONLY = """\
=== BEHAVIORAL PROFILE ===
Background and behavioral data extracted from a user's online activity:

{profile}

=== TASK ===
Write a system prompt in second person (starting with "You are...") \
that an LLM will use to simulate this person writing Reddit replies. \
It must contain three labeled sections:

## 1. Background and Persona
Their strictly evidence-based identity, interests, and values derived from the profile. \
Keep it concise and grounded in the provided data.

## 2. Writing Style Mechanics
Concrete instructions for replicating their likely writing style, inferred from \
the writing complexity level, frequent words, and sentiment signals in the profile. \
Detail actionable mechanics: typical reply length, structural habits, punctuation and \
capitalization quirks. Avoid vague labels; give exact rules.

## 3. Emotional Lexicon Baseline
Using Ekman's 7 basic emotions (happiness, sadness, anger, fear, disgust, surprise, contempt) \
as buckets, establish a likely quantitative lexical baseline inferred from the sentiment \
distribution, polarity scores, subjectivity level, example comments, and community participation. \
Estimate proportional frequency for each dominant category. Identify absent or suppressed categories. \
Provide actionable instructions to mirror this vocabulary ratio.

Do not mention usernames or platform names.\
"""

TARGET_TYPES = {"user_reply_to_op", "user_reply_to_op_with_response"}
FEW_SHOT_SEED = 42
MAX_SAMPLES = 20


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
    n_pos = sent.get("positive", 0)
    n_neg = sent.get("negative", 0)
    n_neu = sent.get("neutral", 0)
    n_total = n_pos + n_neg + n_neu
    sentiment_ratio = {
        "positive": round(n_pos / n_total, 2) if n_total else None,
        "negative": round(n_neg / n_total, 2) if n_total else None,
        "neutral": round(n_neu / n_total, 2) if n_total else None,
    }
    pos_polarity = sent.get("pos_polarity")
    neg_polarity = sent.get("neg_polarity")
    subjectivity = sent.get("subjectivity")

    gf = r.get("readability", {}).get("gf-index", 0)
    complexity = (
        "very casual and informal" if gf < 8
        else "conversational" if gf < 12
        else "educated, somewhat formal" if gf < 16
        else "academic and formal"
    )

    common_words = [w["text"] for w in r.get("metrics", {}).get("common_words", [])[:10]]

    examples = []
    for key in ("most_positive_comment", "most_negative_comment"):
        t = sent.get(key, {}).get("text", "")
        if t:
            examples.append((key, t))
    best = r.get("summary", {}).get("comments", {}).get("best", {}).get("text", "")
    if best and not any(best == t for _, t in examples):
        examples.append(("best", best))

    return {
        "identity": identity,
        "family": family,
        "favorites": favorites,
        "possessions": possessions,
        "interests": interests,
        "top_subs": top_subs,
        "top_subs_by_karma": top_subs_by_karma,
        "sentiment_ratio": sentiment_ratio,
        "pos_polarity": pos_polarity,
        "neg_polarity": neg_polarity,
        "subjectivity": subjectivity,
        "writing_complexity": complexity,
        "common_words": common_words,
        "comment_examples": examples[:3],
    }


def fmt_with_snippet(items):
    parts = []
    for v, snip in items:
        parts.append(f'{v} ("{snip}")' if snip else v)
    return ", ".join(parts)


def format_profile(e, include_style_signals=True):
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
        lines.append(f"Sentiment distribution: {r['positive']*100:.0f}% positive, {r['negative']*100:.0f}% negative, {r['neutral']*100:.0f}% neutral")
    if e["pos_polarity"] is not None and e["neg_polarity"] is not None:
        lines.append(f"Emotional intensity: positive polarity {e['pos_polarity']:.2f}, negative polarity {e['neg_polarity']:.2f}")
    if e["subjectivity"] is not None:
        subj_label = (
            "highly opinionated and personal" if e["subjectivity"] > 0.6
            else "moderately subjective" if e["subjectivity"] > 0.35
            else "predominantly factual and impersonal"
        )
        lines.append(f"Subjectivity: {subj_label} ({e['subjectivity']:.2f})")
    if include_style_signals:
        lines.append(f"Writing style: {e['writing_complexity']}")
        if e["common_words"]:
            lines.append(f"Frequent words: {', '.join(e['common_words'])}")
    if e["comment_examples"]:
        lines.append("Example comments:")
        for label, text in e["comment_examples"]:
            lines.append(f'  [{label}] "{text}"')
    return "\n".join(lines)


def diverse_shuffle(train_records, rng):
    seen_prefixes = set()
    diverse = []
    fallback = []
    shuffled = train_records[:]
    rng.shuffle(shuffled)
    for r in shuffled:
        op = r["conversation"][0]["text"]
        prefix = " ".join(op.split()[:6]).lower()
        if prefix not in seen_prefixes:
            seen_prefixes.add(prefix)
            diverse.append(r)
        else:
            fallback.append(r)
    return diverse + fallback


def format_samples(records):
    lines = []
    for r in records:
        op = r["conversation"][0]["text"]
        reply = r["conversation"][1]["text"]
        lines.append(f'Post: "{op}"\nReply: "{reply}"')
    return "\n\n".join(lines)


def serialize_sample(r):
    return {
        "op": r["conversation"][0]["text"],
        "reply": r["conversation"][1]["text"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--few-shot-k", type=int, default=MAX_SAMPLES)
    args = parser.parse_args()

    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}

    train_by_user = defaultdict(list)
    for line in open(TRAIN):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("type") in TARGET_TYPES:
            train_by_user[r["username"]].append(r)

    rng = random.Random(FEW_SHOT_SEED)

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
            e = extract(record, cache)
            train_records = train_by_user.get(username, [])

            if train_records:
                ordered = diverse_shuffle(train_records, rng)
                first_batch = ordered[:args.few_shot_k]
                remaining = ordered[args.few_shot_k:]
                profile = format_profile(e, include_style_signals=False)
                user_content = USER_TEMPLATE.format(
                    samples=format_samples(first_batch),
                    profile=profile,
                )
                remaining_samples = [serialize_sample(r) for r in remaining]
            else:
                profile = format_profile(e, include_style_signals=True)
                user_content = USER_PROFILE_ONLY.format(profile=profile)
                remaining_samples = []

            messages = [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user_content},
            ]

            out.write(json.dumps({
                "username": username,
                "messages": messages,
                "remaining_samples": remaining_samples,
                "few_shot_k": args.few_shot_k,
            }, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} prompts → {OUTPUT}")


if __name__ == "__main__":
    main()
