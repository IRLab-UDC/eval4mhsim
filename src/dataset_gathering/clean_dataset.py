import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

INPUT = Path("data/dataset.jsonl")
OUTPUT = Path("data/dataset_clean.jsonl")
TRAIN = Path("data/dataset_train.jsonl")
TEST = Path("data/dataset_test.jsonl")
STATS = Path("data/statistics.json")

SPLIT_SEED = 42
TEST_RATIO = 0.2
MIN_TRAIN = 3

records = [json.loads(l) for l in INPUT.read_text().splitlines() if l.strip()]


def is_clean(record: dict) -> tuple[bool, str | None]:
    if not record["valid"]:
        return False, record.get("reason")
    if "imgur.com" in (record.get("url") or ""):
        return False, "imgur url"
    for turn in record.get("conversation") or []:
        text = turn.get("text", "")
        if "[deleted]" in text or "[removed]" in text:
            return False, "conversation contains deleted/removed text"
    return True, None


valid, invalid = [], []
for r in records:
    clean, reason = is_clean(r)
    if clean:
        valid.append(r)
    else:
        invalid.append({**r, "reason": reason})

OUTPUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in valid) + "\n")

# --- stratified train/test split ---
rng = random.Random(SPLIT_SEED)
by_user = defaultdict(list)
for r in valid:
    by_user[r["username"]].append(r)

train, test = [], []
dropped_users = []
for username, records in by_user.items():
    n_test = max(1, math.ceil(len(records) * TEST_RATIO))
    n_train = len(records) - n_test
    if n_train < MIN_TRAIN:
        dropped_users.append(username)
        continue
    shuffled = records[:]
    rng.shuffle(shuffled)
    train.extend(shuffled[:n_train])
    test.extend(shuffled[n_train:])

TRAIN.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in train) + "\n")
TEST.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in test) + "\n")


def word_count(text: str) -> int:
    return len(text.split())


def describe(values: list[int | float]) -> dict:
    if not values:
        return {}
    return {
        "min": round(min(values), 1),
        "max": round(max(values), 1),
        "avg": round(sum(values) / len(values), 1),
    }


conv_lengths = []
total_word_counts = []
user_word_counts = []
subreddits = Counter()
type_user_words = defaultdict(list)

for r in valid:
    conv = r["conversation"]
    conv_lengths.append(len(conv))
    all_words = sum(word_count(t["text"]) for t in conv)
    total_word_counts.append(all_words)
    user_turns = [t for t in conv if t["is_user"]]
    user_words = sum(word_count(t["text"]) for t in user_turns)
    user_word_counts.append(user_words)
    subreddit = r["url"].split("/r/")[1].split("/")[0] if "/r/" in r["url"] else "unknown"
    subreddits[subreddit] += 1
    type_user_words[r["type"]].append(user_words)

stats = {
    "total": len(records),
    "valid": len(valid),
    "invalid": len(invalid),
    "invalid_reasons": dict(Counter(r["reason"] for r in invalid)),
    "users": len({r["username"] for r in valid}),
    "users_with_no_valid_records": sorted(
        {r["username"] for r in invalid} - {r["username"] for r in valid}
    ),
    "type_breakdown": dict(Counter(r["type"] for r in valid)),
    "conversation_turns": describe(conv_lengths),
    "total_words_per_conversation": describe(total_word_counts),
    "user_words_per_conversation": describe(user_word_counts),
    "user_words_by_type": {t: describe(v) for t, v in type_user_words.items()},
    "top_subreddits": dict(subreddits.most_common(20)),
    "records_per_user": dict(Counter(r["username"] for r in valid)),
    "split": {
        "seed": SPLIT_SEED,
        "test_ratio": TEST_RATIO,
        "min_train_per_user": MIN_TRAIN,
        "dropped_users": sorted(dropped_users),
        "train_users": len({r["username"] for r in train}),
        "test_users": len({r["username"] for r in test}),
        "train_records": len(train),
        "test_records": len(test),
        "train_type_breakdown": dict(Counter(r["type"] for r in train)),
        "test_type_breakdown": dict(Counter(r["type"] for r in test)),
    },
}

STATS.write_text(json.dumps(stats, indent=2, ensure_ascii=False))
print(f"{len(valid)} valid records written to {OUTPUT}")
print(f"{len(train)} train / {len(test)} test records ({len(dropped_users)} users dropped)")
print(f"Statistics written to {STATS}")
