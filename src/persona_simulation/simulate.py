import json
import argparse
import random
import torch
from collections import defaultdict
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

TARGET_TYPES = {"user_reply_to_op", "user_reply_to_op_with_response"}
FEW_SHOT_SEED = 42

TASK_SUFFIX = "\n\nYou are now on Reddit. Embody this person fully and write their reply to the post below. Be authentic to their voice — casual, direct, and natural, as real Reddit comments are. Do not explain, analyze, or be verbose. Just reply as they would."
TASK_ONLY = "You are on Reddit. Write a reply to the post below. Be casual, direct, and natural, as real Reddit comments are. Do not explain, analyze, or be verbose. Just reply."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-3-12b-it")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--input", default="data/dataset_test.jsonl")
    parser.add_argument("--train", default="data/dataset_train.jsonl")
    parser.add_argument("--persona-source", choices=["none", "default", "optimized"], default="default")
    parser.add_argument("--personas-optimized", default="data/personas_power.jsonl")
    parser.add_argument("--persona-tag", default=None, help="Override the persona label used in the output filename")
    parser.add_argument("--few-shot-k", type=int, default=10)
    parser.add_argument("--few-shot-source", choices=["same_user", "random_user"], default="same_user")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    model_slug = args.model.replace("/", "_")
    persona_tag = f"persona_{args.persona_tag}" if args.persona_tag else f"persona_{args.persona_source}"
    fs_tag = f"fs{args.few_shot_k}{args.few_shot_source}" if args.few_shot_k > 0 else "fs0"
    output_path = f"data/simulations/{model_slug}_t{args.temperature}_maxtok{args.max_tokens}_{persona_tag}_{fs_tag}.jsonl"

    records = [
        json.loads(l)
        for l in open(args.input)
        if l.strip() and json.loads(l).get("type") in TARGET_TYPES
    ]
    if args.debug:
        records = records[:50]

    optimized_personas = {}
    if args.persona_source == "optimized":
        for l in open(args.personas_optimized):
            if not l.strip():
                continue
            r = json.loads(l)
            optimized_personas[r["username"]] = r["system_prompt"]

    rng = random.Random(FEW_SHOT_SEED)
    train_by_user = defaultdict(list)
    all_train = []
    for l in open(args.train):
        if not l.strip():
            continue
        r = json.loads(l)
        if r.get("type") in TARGET_TYPES:
            train_by_user[r["username"]].append(r)
            all_train.append(r)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def build_messages(record):
        if args.persona_source == "none":
            system = TASK_ONLY
        elif args.persona_source == "optimized":
            system = optimized_personas.get(record["username"], record["system_prompt"]) + TASK_SUFFIX
        else:
            system = record["system_prompt"] + TASK_SUFFIX

        messages = [{"role": "system", "content": system}]

        if args.few_shot_k > 0:
            if args.few_shot_source == "same_user":
                pool = train_by_user.get(record["username"], [])
            else:
                pool = [r for r in all_train if r["username"] != record["username"]]
            sampled = rng.sample(pool, min(args.few_shot_k, len(pool)))
            for ex in sampled:
                messages.append({"role": "user", "content": ex["conversation"][0]["text"]})
                messages.append({"role": "assistant", "content": ex["conversation"][1]["text"]})

        messages.append({"role": "user", "content": record["conversation"][0]["text"]})
        return messages

    prompts = [
        tokenizer.apply_chat_template(
            build_messages(r),
            tokenize=False,
            add_generation_prompt=True,
        )
        for r in records
    ]

    n_gpus = torch.cuda.device_count()
    llm = LLM(model=args.model, tensor_parallel_size=n_gpus)
    outputs = llm.generate(prompts, SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens))

    with open(output_path, "w") as out:
        for record, prompt, output in zip(records, prompts, outputs):
            out.write(json.dumps({
                "id": record["id"],
                "username": record["username"],
                "url": record["url"],
                "type": record["type"],
                "op_text": record["conversation"][0]["text"],
                "ground_truth": record["conversation"][1]["text"],
                "simulation": output.outputs[0].text.strip(),
                "llm_input": prompt,
                "llm_output": output.outputs[0].text,
            }, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
