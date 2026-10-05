import json
import argparse
from pathlib import Path
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

INPUT = Path("data/persona_prompts_p.jsonl")
OUTPUT = Path("data/personas_p.jsonl")


PREFIX = "You are"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-tokens", type=int, default=2048)
    args = parser.parse_args()

    records = [json.loads(l) for l in open(INPUT) if l.strip()]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompts = [
        tokenizer.apply_chat_template(
            r["messages"] + [{"role": "assistant", "content": PREFIX}],
            tokenize=False,
            add_generation_prompt=False,
            continue_final_message=True,
        )
        for r in records
    ]

    llm = LLM(model=args.model)
    outputs = llm.generate(prompts, SamplingParams(max_tokens=args.max_tokens, temperature=0.2))

    with open(OUTPUT, "w") as out:
        for i, (record, output) in enumerate(zip(records, outputs)):
            out.write(json.dumps({
                "id": f"user_{i:04d}",
                "username": record["username"],
                "system_prompt": (PREFIX + output.outputs[0].text).strip(),
            }, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} personas → {OUTPUT}")


if __name__ == "__main__":
    main()
