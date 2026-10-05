import json
import glob
import random
import re
from pathlib import Path
from collections import defaultdict

TARGET_TYPES = {"user_reply_to_op", "user_reply_to_op_with_response"}
SEED = 42
TRAIN_FILE = "data/dataset_train.jsonl"
TEST_FILE = "data/dataset_test.jsonl"
SIMULATIONS_GLOB = "data/simulations/*.jsonl"
OUTPUT_BASE = Path("data/consistency")
CHUNK_SIZE = 5


def load_replies_by_user(file_path: str) -> dict:
    by_user = defaultdict(list)
    for line in open(file_path):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("type") in TARGET_TYPES:
            by_user[r["username"]].append(r["conversation"][1]["text"])
    return dict(by_user)


def load_sim_by_user(file_path: str) -> dict:
    by_user = defaultdict(list)
    for line in open(file_path):
        if not line.strip():
            continue
        r = json.loads(line)
        by_user[r["username"]].append(r["simulation"])
    return dict(by_user)


def write_pairs(pairs: list, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "pairs.jsonl", "w") as pf, open(output_dir / "truth.jsonl", "w") as tf:
        for i, (text1, text2, same) in enumerate(pairs, 1):
            pf.write(json.dumps({"id": str(i), "pair": [text1, text2]}) + "\n")
            tf.write(json.dumps({"id": str(i), "same": same}) + "\n")


def make_train_pairs(by_user: dict, rng: random.Random) -> list:
    """
    Train pairs: real/real same-author and diff-author pairs.
    Same-author: chunk i vs chunk i+mid of the same user's records.
    Diff-author: chunk of user A vs chunk of user B.
    Uses chunking for sufficient text length.
    """
    users = [u for u, texts in by_user.items() if len(texts) >= 2]

    user_chunks = {}
    for user in users:
        texts = by_user[user][:]
        rng.shuffle(texts)
        n_chunks = max(2, len(texts) // CHUNK_SIZE)
        chunks = [" ".join(texts[i::n_chunks]).strip() for i in range(n_chunks)]
        chunks = [c for c in chunks if c]
        if len(chunks) >= 2:
            user_chunks[user] = chunks

    users = list(user_chunks.keys())

    same_pairs = []
    for user in users:
        chunks = user_chunks[user]
        mid = len(chunks) // 2
        for i in range(mid):
            text1 = chunks[i]
            text2 = chunks[i + mid]
            if text1 and text2:
                same_pairs.append((text1, text2, True))

    pool = [(user, chunk) for user in users for chunk in user_chunks[user]]
    rng.shuffle(pool)

    diff_pairs = []
    for i in range(len(pool)):
        for j in range(i + 1, len(pool)):
            if pool[i][0] != pool[j][0]:
                if pool[i][1] and pool[j][1]:
                    diff_pairs.append((pool[i][1], pool[j][1], False))
                break
        if len(diff_pairs) >= len(same_pairs):
            break

    n = min(len(same_pairs), len(diff_pairs))
    pairs = same_pairs[:n] + diff_pairs[:n]
    rng.shuffle(pairs)
    return pairs


def make_test_pairs(train_by_user: dict, text2_by_user: dict, rng: random.Random) -> list:
    """
    Test pairs: text1 is always all train records of user A (real Reddit).
    text2 is either all test records of user A (Reddit baseline) or all simulations of user A.
    Same-author: train_A vs text2_A.
    Diff-author: train_A vs text2_B.
    Only users present in both train and text2 are included.
    This keeps text1 domain-consistent with training and makes Reddit/simulation tests directly comparable.
    """
    common_users = sorted(set(train_by_user.keys()) & set(text2_by_user.keys()))

    same_pairs = []
    for user in common_users:
        text1 = " ".join(train_by_user[user]).strip()
        text2 = " ".join(text2_by_user[user]).strip()
        if text1 and text2:
            same_pairs.append((text1, text2, True))

    shuffled = common_users[:]
    rng.shuffle(shuffled)
    diff_pairs = []
    for i in range(len(shuffled)):
        user_a = shuffled[i]
        user_b = shuffled[(i + 1) % len(shuffled)]
        text1 = " ".join(train_by_user[user_a]).strip()
        text2 = " ".join(text2_by_user[user_b]).strip()
        if text1 and text2:
            diff_pairs.append((text1, text2, False))

    n = min(len(same_pairs), len(diff_pairs))
    pairs = same_pairs[:n] + diff_pairs[:n]
    rng.shuffle(pairs)
    return pairs


def sim_file_to_name(path: str) -> str:
    stem = Path(path).stem
    return re.sub(r"_t[\d.]+_maxtok\d+$", "", stem)


if __name__ == "__main__":
    rng = random.Random(SEED)

    train_by_user = load_replies_by_user(TRAIN_FILE)
    test_gt_by_user = load_replies_by_user(TEST_FILE)

    # Train: chunked real/real pairs from dataset_train
    train_dir = OUTPUT_BASE / "train"
    if (train_dir / "pairs.jsonl").exists():
        print("Train: skipping (already exists)")
    else:
        train_pairs = make_train_pairs(train_by_user, rng)
        write_pairs(train_pairs, train_dir)
        print(f"Train: {len(train_pairs)} pairs")

    # Reddit baseline test: train records (text1) vs test records (text2), same user
    gt_dir = OUTPUT_BASE / "ground_truth" / "test"
    if (gt_dir / "pairs.jsonl").exists():
        print("Ground truth test: skipping (already exists)")
    else:
        gt_test_pairs = make_test_pairs(train_by_user, test_gt_by_user, rng)
        write_pairs(gt_test_pairs, gt_dir)
        print(f"Ground truth test: {len(gt_test_pairs)} pairs")

    # Simulation tests: train records (text1) vs simulation (text2), same user
    for sim_file in sorted(glob.glob(SIMULATIONS_GLOB)):
        name = sim_file_to_name(sim_file)
        out_dir = OUTPUT_BASE / name / "test"
        if (out_dir / "pairs.jsonl").exists():
            print(f"{name} test: skipping (already exists)")
            continue
        sim_by_user = load_sim_by_user(sim_file)
        pairs = make_test_pairs(train_by_user, sim_by_user, rng)
        write_pairs(pairs, out_dir)
        print(f"{name} test: {len(pairs)} pairs")
