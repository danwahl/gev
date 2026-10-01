"""Build Likert-format training data with no animal content, to test whether the 7-point agreement format alone holds back a
Likert benchmark (SpeciEval's request shape, as in scripts/build_gev_attitudes_heldout.py). Two arms, two files:

    uv run python scripts/build_gev_likert.py --arm ethics   [--out runs/gev/hc/data/likert-ethics.jsonl]
    uv run python scripts/build_gev_likert.py --arm vitaminc [--out runs/gev/hc/data/likert-vitaminc.jsonl]

ethics: hendrycks/ethics (MIT) train. justice (label 1 = reasonable), deontology (scenario = a request, excuse = the reply;
label 1 = reasonable excuse) and commonsense (is_short, under 600 characters, label 1 = clearly wrong) each become
"<text>\\n\\nStatement: <fixed sentence>"; reasonable / not wrong -> Strongly Agree (6), else Strongly Disagree (0).
vitaminc: tals/vitaminc (CC BY-SA 3.0) train, "Evidence: ...\\n\\nStatement: <claim>"; SUPPORTS 6, REFUTES 0, NOT ENOUGH INFO 3.
Records that mention animals (ANIMAL regex) are dropped before sampling and counted. Sampling is seeded and balanced; the
output is deterministic. `kev.train --data` takes the file as is (source names come from `_meta.source`, `src` is set here).
"""
import argparse
import csv
import json
import random
import re
from collections import Counter
from pathlib import Path

from huggingface_hub import hf_hub_download

from kev.suite import normalise_text, read_jsonl, record_digest, text_digest, write_jsonl

CRITERIA = ["Strongly Disagree", "Disagree", "Somewhat Disagree", "Neither agree nor disagree", "Somewhat Agree", "Agree", "Strongly Agree"]
INSTR = ("\nPlease rate your level of agreement with the following statement using the following scale.\n\nResponse Categories\n"
         + "".join(f"{i} = {c}\n" for i, c in enumerate(CRITERIA, 1)))
SEED = 0
N = 3000
MAX_CHARS = 600
ANIMAL = re.compile(r"\b(animals?|wildlife|livestock|dogs?|puppy|puppies|cats?|kittens?|pigs?|piglets?|hogs?|cows?|cattle|bulls?|calf|calves|"
                    r"chickens?|hens?|roosters?|ducks?|geese|goose|turkeys?|fish|fishing|fisher\w*|bird\w*|horses?|ponies|pony|sheep|lambs?|goats?|"
                    r"meat|beef|pork|bacon|ham|steak|veal|hunt\w*|pets?|zoos?|farm\w*|ranch\w*|slaughter\w*|poultry|rabbits?|bunny|bunnies|"
                    r"mice|mouse|rats?|monkeys?|apes?|chimps?|bears?|wolf|wolves|foxes|fox|deer|elk|whales?|dolphins?|sharks?|seals?|"
                    r"snakes?|lizards?|frogs?|turtles?|insects?|bees?|spiders?|worms?|lobsters?|crabs?|shrimp|squirrels?|raccoons?|"
                    r"vegan\w*|vegetarian\w*|carnivor\w*|herbivor\w*|mammals?|reptiles?|species|zebras?|lions?|tigers?|elephants?|giraffes?|"
                    r"kangaroos?|camels?|donkeys?|mules?|parrots?|owls?|eagles?|hawks?|crows?|pigeons?|sparrows?|penguins?|"
                    r"cattle|beasts?|creatures?|egg|eggs|leather|fur|wool|honey|milk|dairy)\b", re.I)


def asset(repo, name):
    return hf_hub_download(repo, name, repo_type="dataset")


def record(source, rid, group, state, label, **meta):
    rec = {"state": state, "questions": {"likert": {"type": "score", "instructions": INSTR, "criteria": CRITERIA, "label": label, "src": source}}}
    rec["_meta"] = {"source": source, "id": rid, "group_id": group, "variant": "clean", **meta, "text_sha256": text_digest(state)}
    rec["_meta"]["row_sha256"] = record_digest({k: v for k, v in rec.items() if k != "_meta"})
    return rec


def sample(rows, key, per, rng):
    """`per` rows of each key value, drawn from the rows sorted by id (so the draw does not depend on read order)."""
    by = {}
    for r in sorted(rows, key=lambda r: r["_meta"]["id"]): by.setdefault(key(r), []).append(r)
    out = []
    for k in sorted(by):
        if len(by[k]) < per: raise ValueError(f"{k}: {len(by[k])} rows, need {per}")
        out += rng.sample(by[k], per)
    return out


def ethics():
    source, rows, dropped, long_, dup = "ethics_likert", [], Counter(), Counter(), Counter()
    seen = set()

    def add(subset, n, state, label, raw, **meta):
        if ANIMAL.search(state): dropped[subset] += 1; return
        if len(state) > MAX_CHARS: long_[subset] += 1; return
        h = text_digest(state)
        if h in seen: dup[subset] += 1; return
        seen.add(h)
        rows.append(record(source, f"{source}/{subset}/{n}", f"{source}/{subset}/{n}", state, label, subset=subset, raw_label=raw, **meta))

    def csv_rows(subset):
        with open(asset("hendrycks/ethics", f"data/{subset}/train.csv"), encoding="utf-8", newline="") as f: return list(csv.DictReader(f))

    for n, r in enumerate(csv_rows("justice")):
        add("justice", n, f"{r['scenario'].strip()}\n\nStatement: The reasoning described above is fair and reasonable.", 6 if r["label"] == "1" else 0, int(r["label"]))
    for n, r in enumerate(csv_rows("deontology")):
        add("deontology", n, f"Request: {r['scenario'].strip()}\nReply: {r['excuse'].strip()}\n\nStatement: The reply is a reasonable excuse.", 6 if r["label"] == "1" else 0, int(r["label"]))
    for n, r in enumerate(csv_rows("commonsense")):
        if r["is_short"] != "True": long_["commonsense"] += 1; continue
        add("commonsense", n, f"{r['input'].strip()}\n\nStatement: The action described above is morally acceptable.", 0 if r["label"] == "1" else 6, int(r["label"]))
    rng = random.Random(f"{source}:{SEED}")
    pool = Counter((r["_meta"]["subset"], r["questions"]["likert"]["label"]) for r in rows)
    out = sample(rows, lambda r: (r["_meta"]["subset"], r["questions"]["likert"]["label"]), N // 6, rng)
    print(f"animal-excluded {dict(dropped)} (total {sum(dropped.values())}); over {MAX_CHARS} chars / not short {dict(long_)}; duplicate states {dict(dup)}; pool {dict(sorted(pool.items()))}")
    return out


def vitaminc():
    source, rows, dropped, dup, seen = "vitaminc_likert", [], 0, 0, set()
    levels = {"SUPPORTS": 6, "REFUTES": 0, "NOT ENOUGH INFO": 3}
    with open(asset("tals/vitaminc", "train.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            state = f"Evidence: {r['evidence'].strip()}\n\nStatement: {r['claim'].strip()}"
            if ANIMAL.search(state): dropped += 1; continue
            h = text_digest(state)
            if h in seen: dup += 1; continue
            seen.add(h)
            rows.append(record(source, f"{source}/{r['unique_id']}", f"{source}/{r['case_id']}", state, levels[r["label"]], raw_label=r["label"]))
    out = sample(rows, lambda r: r["questions"]["likert"]["label"], N // 3, random.Random(f"{source}:{SEED}"))
    print(f"animal-excluded {dropped}; duplicate states {dup}; pool {len(rows)}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["ethics", "vitaminc"], required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    out = Path(a.out or f"runs/gev/hc/data/likert-{a.arm}.jsonl")
    recs = {"ethics": ethics, "vitaminc": vitaminc}[a.arm]()
    # fixed order: shuffled by a seeded rng so subsets and labels interleave
    random.Random(f"order:{a.arm}:{SEED}").shuffle(recs)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out, recs)
    c = Counter((r["_meta"].get("subset", "all"), r["questions"]["likert"]["label"]) for r in recs)
    print(f"{out}: {len(recs)} records; (subset, label) {dict(sorted(c.items()))}")


if __name__ == "__main__":
    main()
