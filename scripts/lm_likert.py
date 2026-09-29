"""Score SpeciEval items with a causal LM's next-token distribution over the Likert digits (no reasoning, no sampling).

The decision-model counterpart of SpeciEval's text runs: each item gets SpeciEval's system message (the task's
instructions and the Likert scale, translated), then the statement and a fixed English request for a single number in
place of SpeciEval's reasoning template; the model's probability of each digit 1..7 at the answer position is
renormalised, and the item scores that distribution's expected value. A chat checkpoint gets its chat template (thinking
disabled); a base checkpoint gets a plain prompt whose "Statement:" / "Answer:" labels are English in every language.
Renormalising hides a prompt the model does not answer with a digit, so check "digit_mass" (specieval_score.py prints
each run's lowest).

Where SpeciEval has them (English only), every item is also posed as its negated statement (the `reverse` translations,
scored 8 - answer), and the item's score is the mean of the two directions, which cancels a model's lean toward agreeing
(or toward "1") whatever the statement says. Other languages score the forward direction alone.

    uv run python scripts/lm_likert.py --model Qwen/Qwen3.5-4B --specieval <specieval checkout> --out runs/gev/lm/qwen3.5-4b.json

Writes {"model", "revision", "chat", "items": [{"language", "id", "expected", "forward", "reverse"}]}, each direction
with its "probabilities", "expected" (reverse already rescored) and "digit_mass" (the probability the model puts on any
digit at all, which shows whether the prompt format landed).
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# (prefix string id, item ids) per SpeciEval task
TASKS = [
    ("speciesism_prefix", [f"spec_{i}" for i in range(1, 5)]),
    ("sentience_prefix", [f"bfas_{i}" for i in range(1, 7)]),
    ("attitude_prefix", [f"la4N_{i}" for i in range(1, 5)]),
    ("attitude_prefix", [f"se4N_{i}" for i in range(1, 5)]),
]
ANSWER = "Answer with a single number from 1 to 7."


def digit_ids(tok):
    """Token ids for each digit 1..7, with and without a leading space (both count toward that digit)."""
    ids = []
    for d in "1234567":
        cands = {tok.encode(d, add_special_tokens=False)[-1], tok.encode(" " + d, add_special_tokens=False)[-1]}
        ids.append(sorted(cands))
    return ids


def prompt(tok, chat, instructions, statement):
    if chat:
        messages = [{"role": "system", "content": instructions}, {"role": "user", "content": f"{statement}\n\n{ANSWER}"}]
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    # trailing space: Qwen tokenizes " 1" as " " + "1", so without it the next token is whitespace, not a digit
    # (a SentencePiece tokenizer would want no trailing space)
    return f"{instructions}\n\nStatement: {statement}\n{ANSWER}\nAnswer: "


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None, help="pin the checkpoint, e.g. the base_revision a Kev model trained on")
    ap.add_argument("--specieval", required=True, help="path to a specieval checkout (for the translated prompts)")
    ap.add_argument("--chat", type=int, default=None, help="1 = chat template, 0 = plain; default: 1 unless the name ends in -Base")
    ap.add_argument("--languages", default="all", help="comma list, or 'all'")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    sys.path.insert(0, str(Path(a.specieval) / "src"))
    from specieval.translations import Language, Translations

    chat = bool(a.chat) if a.chat is not None else not a.model.endswith("-Base")
    languages = list(Language) if a.languages == "all" else [Language(x) for x in a.languages.split(",")]
    directions = {"forward": Translations(), "reverse": Translations(reverse=True)}
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    model = AutoModelForCausalLM.from_pretrained(a.model, revision=a.revision, dtype=torch.bfloat16, device_map="cuda").eval()
    digits = digit_ids(tok)

    def ask(t, lang, prefix_id, qid):
        # the system message SpeciEval sends every model
        instructions = f"\n{t.get_string(prefix_id, lang)}\n\n{t.get_string('likert_scale', lang)}\n"
        enc = tok(prompt(tok, chat, instructions, t.get_string(qid, lang)), return_tensors="pt", add_special_tokens=False).to("cuda")
        with torch.no_grad():
            p = torch.softmax(model(**enc).logits[0, -1].float(), -1)
        mass = [sum(p[i].item() for i in ids_) for ids_ in digits]
        probs = [m / sum(mass) for m in mass]
        return {"probabilities": probs, "expected": sum((i + 1) * q for i, q in enumerate(probs)), "digit_mass": sum(mass)}

    items = []
    for lang in languages:
        for prefix_id, ids in TASKS:
            for qid in ids:
                item = {"language": lang.value, "id": qid}
                item["forward"] = ask(directions["forward"], lang, prefix_id, qid)
                if lang == Language.ENGLISH:
                    item["reverse"] = ask(directions["reverse"], lang, prefix_id, qid)
                    item["reverse"]["expected"] = 8 - item["reverse"]["expected"]
                item["expected"] = sum(item[d]["expected"] for d in directions if d in item) / sum(d in item for d in directions)
                items.append(item)
        print(lang.value, "done", file=sys.stderr)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"model": a.model, "revision": a.revision, "chat": chat, "items": items}, indent=1))


if __name__ == "__main__":
    main()
