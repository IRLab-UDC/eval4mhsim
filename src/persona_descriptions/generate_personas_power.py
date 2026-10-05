import gc
import json
import argparse
import copy
from pathlib import Path
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import torch

INPUT = Path("data/persona_prompts_power.jsonl")
OUTPUT = Path("data/personas_power.jsonl")

PREFIX = "You are"

GENERATOR_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
CRITIC_MODEL = "microsoft/phi-4"

CRITIC_SYSTEM = (
    "You are evaluating a persona description used to make an LLM simulate a specific person's Reddit replies. "
    "Given the original writing samples and behavioral profile, critique the draft on these three dimensions:\n\n"
    "1. BEHAVIORAL ACCURACY: Is the background and personality strictly evidence-based? Flag any traits that are "
    "inaccurate, generic, or inconsistent with the samples.\n\n"
    "2. WRITING STYLE REPLICABILITY: Are the style instructions highly concrete? Flag vague labels (e.g., 'casual'). "
    "The draft must detail actionable mechanics: specific punctuation/capitalization quirks, formatting, and "
    "structural habits (how they open, close, and hedge).\n\n"
    "3. EMOTIONAL LEXICON MATCHING: Does the draft explicitly instruct the LLM to mirror the user's proportional "
    "use of emotional vocabulary? Using Ekman's 7 basic emotions (happiness, sadness, anger, fear, disgust, "
    "surprise, contempt) as buckets, the draft must quantify the user's baseline. Check for: instructions to "
    "match the exact percentage/frequency of emotional words per category, identification of heavily over-indexed "
    "or completely absent emotions, and specific go-to emotional trigger words used in the samples.\n\n"
    "Be specific and concise."
)

REVISION_PREFIX = "Here is feedback on your draft:\n\n{critique}\n\nPlease revise the persona description accordingly."

REVISION_PREFIX_WITH_SAMPLES = """\
=== ADDITIONAL WRITING SAMPLES ===
Here are more examples of how this person writes, not seen in previous iterations:

{samples}

Here is feedback on your draft:

{critique}

Please revise the persona description incorporating both the new samples and the feedback."""


def apply_chat(tokenizer, messages, prefill=None):
    if prefill:
        return tokenizer.apply_chat_template(
            messages + [{"role": "assistant", "content": prefill}],
            tokenize=False,
            add_generation_prompt=False,
            continue_final_message=True,
        )
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def release():
    gc.collect()
    torch.cuda.empty_cache()


def truncate_gen_history(history, tokenizer, budget):
    """Trim the first user message until the full rendered history fits in budget.

    Trimming is always applied to user_0 (the original samples + profile block)
    regardless of which iteration we are on, so later revision turns are never cut.
    We iterate because removing tokens from user_0 reduces the rendered length by
    slightly fewer tokens than removed (due to subword boundaries), so one pass may
    not be enough.
    """
    history = copy.deepcopy(history)
    first_user_idx = next((i for i, m in enumerate(history) if m["role"] == "user"), None)
    if first_user_idx is None:
        return history

    for _ in range(3):  # up to 3 passes to absorb subword boundary slack
        rendered = tokenizer.apply_chat_template(
            history + [{"role": "assistant", "content": PREFIX}],
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
        )
        overflow = len(rendered) - budget
        if overflow <= 0:
            break
        ids = tokenizer.encode(history[first_user_idx]["content"])
        trimmed = max(0, len(ids) - overflow)
        if trimmed == 0:
            history[first_user_idx]["content"] = ""
            break
        history[first_user_idx]["content"] = tokenizer.decode(ids[:trimmed], skip_special_tokens=True)

    return history


def truncate_context(ctx, draft, tokenizer, max_model_len, max_output_tokens, overhead=256):
    budget = max_model_len - max_output_tokens - overhead
    draft_ids = tokenizer.encode(draft)
    ctx_ids = tokenizer.encode(ctx)
    allowed_ctx = budget - len(draft_ids)
    if allowed_ctx <= 0:
        return ""
    if len(ctx_ids) > allowed_ctx:
        ctx_ids = ctx_ids[:allowed_ctx]
        return tokenizer.decode(ctx_ids, skip_special_tokens=True)
    return ctx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator", default=GENERATOR_MODEL)
    parser.add_argument("--critic", default=CRITIC_MODEL)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--max-tokens-gen", type=int, default=2048)
    parser.add_argument("--max-tokens-critic", type=int, default=1024)
    parser.add_argument("--max-model-len-gen", type=int, default=26384)
    parser.add_argument("--max-model-len-critic", type=int, default=16384)
    args = parser.parse_args()

    records = [json.loads(l) for l in open(INPUT) if l.strip()]

    gen_tokenizer = AutoTokenizer.from_pretrained(args.generator)
    critic_tokenizer = AutoTokenizer.from_pretrained(args.critic)

    histories = [copy.deepcopy(r["messages"]) for r in records]
    # critic_contexts accumulates all evidence seen so far (initial prompt + any
    # additional samples injected in revision turns), so the critic is never blind
    # to samples the generator already had access to.
    critic_contexts = [r["messages"][-1]["content"] for r in records]
    sample_pools = [r.get("remaining_samples", []) for r in records]
    few_shot_ks = [r.get("few_shot_k", 20) for r in records]
    sample_cursors = [0] * len(records)

    drafts = []

    gen_budget = args.max_model_len_gen - args.max_tokens_gen - 256

    for iteration in range(args.iterations):
        raw_prompts = [apply_chat(gen_tokenizer, truncate_gen_history(h, gen_tokenizer, gen_budget), prefill=PREFIX) for h in histories]
        gen_prompts = []
        for p in raw_prompts:
            ids = gen_tokenizer.encode(p)
            if len(ids) >= args.max_model_len_gen:
                ids = ids[:args.max_model_len_gen - 1]
                p = gen_tokenizer.decode(ids, skip_special_tokens=False)
            gen_prompts.append(p)

        gen_llm = LLM(model=args.generator, tensor_parallel_size=torch.cuda.device_count(), max_model_len=args.max_model_len_gen)
        gen_outputs = gen_llm.generate(gen_prompts, SamplingParams(max_tokens=args.max_tokens_gen, temperature=0.2))
        del gen_llm
        release()

        drafts = [(PREFIX + o.outputs[0].text).strip() for o in gen_outputs]

        for h, draft in zip(histories, drafts):
            h.append({"role": "assistant", "content": draft})

        if iteration == args.iterations - 1:
            break

        critic_messages = []
        for ctx, draft in zip(critic_contexts, drafts):
            ctx_truncated = truncate_context(ctx, draft, critic_tokenizer, args.max_model_len_critic, args.max_tokens_critic)
            critic_messages.append([
                {"role": "system", "content": CRITIC_SYSTEM},
                {"role": "user", "content": f"{ctx_truncated}\n\n=== DRAFT PERSONA ===\n{draft}"},
            ])
        critic_prompts = [apply_chat(critic_tokenizer, m) for m in critic_messages]

        critic_llm = LLM(model=args.critic, tensor_parallel_size=torch.cuda.device_count(), max_model_len=args.max_model_len_critic)
        critic_outputs = critic_llm.generate(critic_prompts, SamplingParams(max_tokens=args.max_tokens_critic, temperature=0.2))
        del critic_llm
        release()

        critiques = [o.outputs[0].text.strip() for o in critic_outputs]

        for i, (h, critique) in enumerate(zip(histories, critiques)):
            pool = sample_pools[i]
            cursor = sample_cursors[i]
            k = few_shot_ks[i]
            batch = pool[cursor:cursor + k]
            sample_cursors[i] = cursor + len(batch)

            if batch:
                sample_lines = "\n\n".join(
                    f'Post: "{s["op"]}"\nReply: "{s["reply"]}"' for s in batch
                )
                revision = REVISION_PREFIX_WITH_SAMPLES.format(samples=sample_lines, critique=critique)
                # extend critic context with the new samples for the next iteration
                critic_contexts[i] += f"\n\n=== ADDITIONAL WRITING SAMPLES ===\n{sample_lines}"
            else:
                revision = REVISION_PREFIX.format(critique=critique)

            h.append({"role": "user", "content": revision})

    with open(OUTPUT, "w") as out:
        for i, (record, draft) in enumerate(zip(records, drafts)):
            out.write(json.dumps({
                "id": f"user_{i:04d}",
                "username": record["username"],
                "system_prompt": draft,
            }, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
