"""Label agreement statements with a chat model's reasoned answers, as distributions over the 1-7 scale.

Each record's statement is posed the way SpeciEval poses its items to chat models: the record's own scale instructions
as the system message, the statement plus SpeciEval's step-by-step template as the user turn, thinking off. The model
answers `--samples` times; the parsed answers become the record's soft target (`target`, level index -> probability).

`--engine vllm` (run from an environment with vLLM, e.g. a separate venv) refills its batch as answers finish; `hf`
runs each batch to its longest answer. Results are appended per batch to `<out>` (one line per record), and a rerun skips records already written, so the job
survives preemption.

    uv run --no-sync python scripts/gev_teacher_likert.py --data runs/gev/hc/data/ethics-likert.jsonl \
        --per_subset 0 --out runs/gev/teacher/ethics-likert-27b.jsonl
"""

import argparse
import collections
import json
import os
import random
import re
import time


# SpeciEval's English cot_template (src/specieval/translations/data/translations.csv), levels = 7.
COT = ('{prompt}\n\nBefore answering, reason in a step-by-step manner. Provide your answer at the end on its own line in '
       'the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is a number between 1 and 7.')
ANSWER = re.compile(r"ANSWER\s*:\s*(\d+)")


def pick(rows, per_subset, seed):
    by = collections.defaultdict(list)
    for r in rows:
        by[r["_meta"].get("subset", "all")].append(r)
    rng = random.Random(seed)
    out = []
    for subset in sorted(by):
        rs = by[subset]
        rng.shuffle(rs)
        out += rs[:per_subset] if per_subset else rs
    return out


def sampling(llm):
    """The model's own generation_config sampling settings (what HF generate uses by default)."""
    g = llm.llm_engine.model_config.get_diff_sampling_param()
    return {k: g[k] for k in ("temperature", "top_p", "top_k", "repetition_penalty") if k in g}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--per_subset", type=int, default=100, help="records per _meta.subset; 0 = all")
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--engine", choices=["hf", "vllm"], default="hf")
    ap.add_argument("--gpu_memory", type=float, default=0.9, help="vllm: share of GPU memory for weights + KV cache")
    ap.add_argument("--max_num_seqs", type=int, default=256, help="vllm: sequences decoded at once")
    ap.add_argument("--batch", type=int, default=4, help="records per generate call (x samples sequences)")
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rows = pick([json.loads(l) for l in open(a.data, encoding="utf-8")], a.per_subset, a.seed)
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["_meta"]["id"] for l in open(a.out, encoding="utf-8")}
    todo = [r for r in rows if r["_meta"]["id"] not in done]
    print(f"{len(rows)} records, {len(done)} done, {len(todo)} to go", flush=True)
    if not todo:
        return

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    if a.engine == "vllm":
        from vllm import LLM, SamplingParams
        llm = LLM(a.model, revision=a.revision, dtype="bfloat16", max_model_len=4096, seed=a.seed,
                  gpu_memory_utilization=a.gpu_memory, max_num_seqs=a.max_num_seqs,
                  limit_mm_per_prompt={"image": 0, "video": 0})   # text only: skips the vision encoder
        tok = llm.get_tokenizer()
        params = SamplingParams(n=a.samples, max_tokens=a.max_new_tokens, seed=a.seed, **sampling(llm))

        def generate(prompts):
            return [o.text for out in llm.generate(prompts, params, use_tqdm=False) for o in out.outputs]
    else:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(a.model, revision=a.revision, dtype=torch.bfloat16,
                                                     device_map="cuda").eval()
        torch.manual_seed(a.seed + len(done))

        def generate(prompts):
            enc = tok(prompts, return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                gen = model.generate(**enc, do_sample=True, num_return_sequences=a.samples,
                                     max_new_tokens=a.max_new_tokens)
            return tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    t0 = time.time()
    for i in range(0, len(todo), a.batch):
        chunk = todo[i:i + a.batch]
        prompts = []
        for r in chunk:
            q = next(iter(r["questions"].values()))
            msgs = [{"role": "system", "content": q["instructions"]},
                    {"role": "user", "content": COT.format(prompt=r["state"])}]
            prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                   enable_thinking=False))
        texts = generate(prompts)
        with open(a.out, "a", encoding="utf-8") as f:
            for j, r in enumerate(chunk):
                outs = texts[j * a.samples:(j + 1) * a.samples]
                answers = []
                for t in outs:
                    m = ANSWER.findall(t)
                    answers.append(int(m[-1]) if m and 1 <= int(m[-1]) <= 7 else None)
                valid = [x for x in answers if x is not None]
                target = ({str(k - 1): c / len(valid) for k, c in sorted(collections.Counter(valid).items())}
                          if valid else None)
                rec = dict(r, questions={k: dict(v, target=target) for k, v in r["questions"].items()})
                rec["_meta"] = dict(r["_meta"], teacher={"model": a.model, "revision": a.revision, "answers": answers})
                rec["_meta"]["teacher_samples"] = outs
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n = i + len(chunk)
        print(f"{n}/{len(todo)} records, {time.time() - t0:.0f}s", flush=True)
    print("TEACHERDONE", flush=True)


if __name__ == "__main__":
    main()
