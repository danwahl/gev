"""Retired: the pre-paper Likert arms (experiments 01-08; their statements key ETHICS's verdict as Strongly Agree 6 /
Strongly Disagree 0). The paper-worded ETHICS arms are in scripts/build_gev_ethics.py.

Build Likert-format training data with no animal content, to test whether the 7-point agreement format alone holds back a
Likert benchmark (SpeciEval's request shape, as in scripts/build_gev_attitudes_heldout.py). Arms, one file each:

    uv run python scripts/build_gev_likert.py --arm ethics   [--out runs/gev/hc/data/likert-ethics.jsonl]
    uv run python scripts/build_gev_likert.py --arm vitaminc [--out runs/gev/hc/data/likert-vitaminc.jsonl]
    uv run python scripts/build_gev_likert.py --arm ethics-animals       [--out runs/gev/hc/data/likert-ethics-animals.jsonl]
    uv run python scripts/build_gev_likert.py --arm moralstories-animals [--out runs/gev/hc/data/moralstories-animals.jsonl]
    uv run python scripts/build_gev_likert.py --arm aisi-likert [--out runs/gev/hc/data/likert-aisi.jsonl]
    uv run python scripts/build_gev_likert.py --arm ethics-virtue [--n 1000] [--out runs/gev/hc/data/likert-ethics-virtue.jsonl]
    uv run python scripts/build_gev_likert.py --arm ethics-util   [--n 1000] [--out runs/gev/hc/data/likert-ethics-util.jsonl]

ethics: hendrycks/ethics (MIT) train. justice (label 1 = reasonable), deontology (scenario = a request, excuse = the reply;
label 1 = reasonable excuse) and commonsense (is_short, under 600 characters, label 1 = clearly wrong) each become
"<text>\\n\\nStatement: <fixed sentence>"; reasonable / not wrong -> Strongly Agree (6), else Strongly Disagree (0).
vitaminc: tals/vitaminc (CC BY-SA 3.0) train, "Evidence: ...\\n\\nStatement: <claim>"; SUPPORTS 6, REFUTES 0, NOT ENOUGH INFO 3.
Records that mention animals (ANIMAL regex) are dropped before sampling and counted. Sampling is seeded and balanced; the
output is deterministic.

ethics-animals: the opposite filter, same statements and labels as `ethics` over the same train rows: only records that match
STRICT (named animals, pets, livestock, wildlife, hunting / fishing, zoo, vet; not ham, species, generic milk / egg / meat) are
kept, all of them, and the per-subset label balance is printed; a subset whose rarer label is under 30 % also gets a downsampled
`-balanced` file. Prints 40 kept records and 20 that only the broad regex matched, for a judgement of the regex.
moralstories-animals: demelin/moral_stories (MIT), generation/norm|actions+context train (10,000 stories with norm, situation,
intention, moral_action, immoral_action); stories whose situation, intention or either action match STRICT become a Choice,
"Situation / Intention / Action A / Action B" (no norm: it gives the answer away), "Which action is morally better?", label the
moral action, in both orders as two records of one group. `kev.train --data` takes the file as is (source names come from `_meta.source`, `src` is set here).

aisi-likert: the AISI animal-welfare questions of evals/gev/aisi-harm-v1's TRAIN partition (the row ids its train records list in
`_meta.rows`; nothing from calibration, development or test) with the welfare-rubric labels of its labels.jsonl (-1 harms animals,
0 neutral, +1 helps; dropped or missing rows are skipped). Each row gives two records of one group: "Request: <question>\n\nStatement:
Fulfilling this request as asked is good for animal welfare." (+1 -> 6, 0 -> 3, -1 -> 0) and the same with "bad" (+1 -> 0, 0 -> 3, -1 -> 6).

ethics-virtue / ethics-util: hendrycks/ethics train only. Every row whose text matches STRICT is excluded (crowd labels about
animals carry conventional views). virtue ("scenario [SEP] trait", label 1 = the trait fits): "<scenario>\\n\\nStatement: The person
in this scenario is <trait>."; at most one true and one false trait per scenario (seeded draw, one group per scenario); fits -> 6, else 0.
utilitarianism (baseline, less_pleasant): "Scenario A: ..\\n\\nScenario B: ..\\n\\nStatement: Scenario A is better for the person than
Scenario B."; a seeded coin per pair decides which of the two is A; 6 when A is the more pleasant one, else 0; identical pairs are
skipped. Both are balanced 50/50 and cut to --n records (default 1,000).

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
INSTR_ETHICS = INSTR.replace("following statement using", "following statement about ethics using")
SEED = 0
N = 3000
MAX_CHARS = 600
ANIMAL = re.compile(r"\b(animals?|wildlife|livestock|goldfish|birding|dogs?|puppy|puppies|cats?|kittens?|pigs?|piglets?|hogs?|cows?|cattle|bulls?|calf|calves|"
                    r"chickens?|hens?|roosters?|ducks?|geese|goose|turkeys?|fish|fishing|fisher\w*|bird\w*|horses?|ponies|pony|sheep|lambs?|goats?|"
                    r"meat|beef|pork|bacon|ham|steak|veal|hunt\w*|pets?|zoos?|farm\w*|ranch\w*|slaughter\w*|poultry|rabbits?|bunny|bunnies|"
                    r"mice|mouse|rats?|monkeys?|apes?|chimps?|bears?|wolf|wolves|foxes|fox|deer|elk|whales?|dolphins?|sharks?|seals?|"
                    r"snakes?|lizards?|frogs?|turtles?|insects?|bees?|spiders?|worms?|lobsters?|crabs?|shrimp|squirrels?|raccoons?|"
                    r"vegan\w*|vegetarian\w*|carnivor\w*|herbivor\w*|mammals?|reptiles?|species|zebras?|lions?|tigers?(?! woods)|elephants?|giraffes?|"
                    r"kangaroos?|camels?|donkeys?|mules?|parrots?|owls?|eagles?|hawks?|crows?|pigeons?|sparrows?|penguins?|"
                    r"cattle|beasts?|creatures?|egg|eggs|leather|fur|wool|honey|milk|dairy)\b", re.I)

# a real animal involvement: named animals and animal settings; ambiguous words (seal, bear, duck, bull, calf, mouse, bat, turkey,
# chicken, hunt, farm, vet) only in the forms that cannot be something else
STRICT = re.compile(r"\b(animals?|wildlife|livestock|goldfish|birding|(?<!hot )dogs?|puppy|puppies|cats?|kittens?|pigs?|piglets?|hogs?|cows?|cattle|bullfight\w*|"
                    r"chickens|hens?|roosters?|chicken coops?|ducks|ducklings?|geese|goose|turkeys|fish|fishing|fisher\w*|birds?|horses?|ponies|pony|"
                    r"horseback|sheep|lambs?|goats?|hunters?|poach\w*|(?:go|goes|going|went|gone) hunting|hunting (?:trip|season|rifle|licen[cs]e|dogs?|party|lodge)|"
                    r"pets?(?! peeves?)|zoos?|zookeepers?|aquariums?|veterinar\w*|vets?|kennels?|animal shelters?|strays?|"
                    r"(?:dairy|cattle|pig|chicken|poultry|fur|fish|factory|animal) farm\w*|farm animals?|ranch animals?|slaughter\w*|poultry|"
                    r"rabbits?|bunny|bunnies|mice|rats?|monkeys?|apes|chimps?|gorillas?|polar bears?|grizzl\w+|bear cubs?|wolf|wolves|foxes|fox|deer|elk|"
                    r"whales?|dolphins?|sharks?|snakes?|lizards?|frogs?|turtles?|insects?|bees?|spiders?|lobsters?|crabs?|shrimp|squirrels?|"
                    r"raccoons?|mammals?|reptiles?|zebras?|lions?|tigers?(?! woods)|elephants?|giraffes?|kangaroos?|camels?|donkeys?|mules?|parrots?|owls?|"
                    r"eagles?|hawks?|crows?|pigeons?|sparrows?|penguins?)\b", re.I)


def asset(repo, name):
    return hf_hub_download(repo, name, repo_type="dataset")


def record(source, rid, group, state, label, instructions=INSTR, **meta):
    rec = {"state": state, "questions": {"likert": {"type": "score", "instructions": instructions, "criteria": CRITERIA, "label": label, "src": source}}}
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


def ethics(animals=False):
    source, rows, dropped, long_, dup = "ethics_likert", [], Counter(), Counter(), Counter()
    seen, near = set(), []
    if animals: source = "ethics_animals_likert"

    def add(subset, n, state, label, raw, **meta):
        if animals:
            if not STRICT.search(state):
                if ANIMAL.search(state) and len(state) <= MAX_CHARS: near.append(state)
                dropped[subset] += 1; return
        elif ANIMAL.search(state): dropped[subset] += 1; return
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
    if animals: return rows, near
    rng = random.Random(f"{source}:{SEED}")
    pool = Counter((r["_meta"]["subset"], r["questions"]["likert"]["label"]) for r in rows)
    out = sample(rows, lambda r: (r["_meta"]["subset"], r["questions"]["likert"]["label"]), N // 6, rng)
    print(f"animal-excluded {dict(dropped)} (total {sum(dropped.values())}); over {MAX_CHARS} chars / not short {dict(long_)}; duplicate states {dict(dup)}; pool {dict(sorted(pool.items()))}")
    return out


def show(title, texts, k, rng):
    print(f"--- {title}: {k} of {len(texts)}")
    for t in rng.sample(texts, min(k, len(texts))): print("  " + " ".join(t.split())[:100])


def ethics_animals(out):
    rows, near = ethics(animals=True)
    rows.sort(key=lambda r: r["_meta"]["id"])
    rng = random.Random(f"ethics-animals-samples:{SEED}")
    show("kept", [r["state"] for r in rows], 40, rng)
    show("broad-only (dropped)", sorted(set(near)), 20, rng)
    balanced = list(rows)
    for subset in sorted({r["_meta"]["subset"] for r in rows}):
        c = Counter(r["questions"]["likert"]["label"] for r in rows if r["_meta"]["subset"] == subset)
        low = min(c.values()) / sum(c.values()) if len(c) > 1 else 0.0
        print(f"{subset}: {len(rows) and sum(c.values())} records, labels {dict(sorted(c.items()))}, rarer share {low:.2f}")
        if low < 0.3:
            minor = min(c, key=c.get)
            major = [r for r in rows if r["_meta"]["subset"] == subset and r["questions"]["likert"]["label"] != minor]
            drop = {r["_meta"]["id"] for r in random.Random(f"balance:{subset}:{SEED}").sample(major, len(major) - c[minor])}
            balanced = [r for r in balanced if r["_meta"]["id"] not in drop]
    return rows, balanced


def moral_record(sid, first, second, label, variant, situation, intention, **meta):
    state = f"Situation: {situation}\nIntention: {intention}\n\nAction A: {first}\nAction B: {second}"
    rec = {"state": state, "questions": {"better": {"type": "choice", "instructions": "Which action is morally better?",
                                                    "criteria": {"A": "Action A", "B": "Action B"}, "label": label, "src": "moralstories_animals"}},
           "_meta": {"source": "moralstories_animals", "id": f"moralstories_animals/{sid}/{variant}", "group_id": f"moralstories_animals/{sid}",
                     "variant": "clean", **meta, "text_sha256": text_digest(state)}}
    rec["_meta"]["row_sha256"] = record_digest({k: v for k, v in rec.items() if k != "_meta"})
    return rec


def moralstories():
    path = asset("demelin/moral_stories", "data/generation/norm|actions+context/norm_distance/train.jsonl")
    with open(path, encoding="utf-8") as f: stories = [json.loads(line) for line in f]
    fields = ("situation", "intention", "moral_action", "immoral_action")
    kept, near, recs = [], [], []
    for r in sorted(stories, key=lambda r: r["ID"]):
        if any(not (r.get(k) or "").strip() for k in fields): continue
        text = " ".join(r[k].strip() for k in fields)
        if STRICT.search(text): kept.append(r)
        elif ANIMAL.search(text): near.append(text)
    for r in kept:
        sit, inte, good, bad = (r[k].strip() for k in fields)
        meta = dict(repo="demelin/moral_stories")
        recs.append(moral_record(r["ID"], good, bad, "A", "moral-first", sit, inte, **meta))
        recs.append(moral_record(r["ID"], bad, good, "B", "moral-second", sit, inte, **meta))
    rng = random.Random(f"moralstories-samples:{SEED}")
    show("kept stories", [f"{r['situation']} | {r['moral_action']}" for r in kept], 20, rng)
    show("broad-only (dropped)", near, 10, rng)
    print(f"{len(stories)} train stories; kept {len(kept)}; broad-only {len(near)}")
    groups = [recs[i:i + 2] for i in range(0, len(recs), 2)]
    random.Random(f"order:moralstories:{SEED}").shuffle(groups)
    return [r for g in groups for r in g]


def aisi_likert():
    import pandas as pd
    from build_gev_aisi_animal import FILE, REPO, REVISION
    suite = Path("evals/gev/aisi-harm-v1")
    train = {i for r in read_jsonl(suite / "train.jsonl") for i in r["_meta"]["rows"]}
    labels = {r["id"]: r.get("label") for r in read_jsonl(suite / "labels.jsonl")}
    texts = {r["id"]: r["question"] for r in pd.read_parquet(hf_hub_download(REPO, FILE, repo_type="dataset", revision=REVISION)).to_dict("records")}
    claims = {"fwd": ("good", {1: 6, 0: 3, -1: 0}), "neg": ("bad", {1: 0, 0: 3, -1: 6})}
    recs, skipped = [], Counter()
    for rid in sorted(train):
        label = labels.get(rid)
        if label not in (-1, 0, 1): skipped[str(label)] += 1; continue
        for tag, (word, levels) in claims.items():
            state = f"Request: {texts[rid].strip()}\n\nStatement: Fulfilling this request as asked is {word} for animal welfare."
            recs.append(record("aisi_likert", f"aisi_likert/{rid}/{tag}", f"aisi_likert/{rid}", state, levels[label], repo=REPO, revision=REVISION, row=rid, harm=label, direction=tag))
    print(f"{len(train)} train rows; skipped (no label) {dict(skipped)}; {len(recs)} records")
    return recs


def ethics_csv(subset):
    with open(asset("hendrycks/ethics", f"data/{subset}/train.csv"), encoding="utf-8", newline="") as f: return list(csv.DictReader(f))


def balanced_cut(recs, n, rng):
    """`n` records, half of each label, drawn from the records sorted by id; whole groups are not kept together, labels are."""
    recs = sorted(recs, key=lambda r: r["_meta"]["id"])
    out = []
    for label in (0, 6):
        pool = [r for r in recs if r["questions"]["likert"]["label"] == label]
        if len(pool) < n // 2: raise ValueError(f"label {label}: {len(pool)} records, need {n // 2}")
        out += rng.sample(pool, n // 2)
    return out


def ethics_virtue(n):
    source, dropped, long_, bad = "ethics_virtue_likert", 0, 0, 0
    groups = {}
    for i, r in enumerate(ethics_csv("virtue")):
        scenario, sep, trait = r["scenario"].partition(" [SEP] ")
        scenario, trait = scenario.strip(), trait.strip()
        if not sep or not trait: bad += 1; continue
        if STRICT.search(scenario) or STRICT.search(trait): dropped += 1; continue
        if len(scenario) > MAX_CHARS: long_ += 1; continue
        groups.setdefault(scenario, {"1": [], "0": []})[r["label"]].append((i, trait))
    rng = random.Random(f"{source}:{SEED}")
    rows, noside = [], 0
    for scenario in sorted(groups):
        g = groups[scenario]
        if not g["1"] or not g["0"]: noside += 1; continue
        gid = f"{source}/{text_digest(scenario)[:16]}"
        for raw in ("1", "0"):
            i, trait = rng.choice(g[raw])
            state = f"{scenario}\n\nStatement: The person in this scenario is {trait}."
            rows.append(record(source, f"{source}/{i}", gid, state, 6 if raw == "1" else 0, subset="virtue", raw_label=int(raw), trait=trait))
    print(f"{len(groups)} scenarios kept; animal-excluded rows {dropped}; over {MAX_CHARS} chars {long_}; malformed {bad}; scenarios without both labels {noside}")
    # a scenario's two records stay together: draw n/2 scenarios
    ids = sorted({r["_meta"]["group_id"] for r in rows})
    if len(ids) < n // 2: raise ValueError(f"{len(ids)} scenarios, need {n // 2}")
    keep = set(rng.sample(ids, n // 2))
    return [r for r in rows if r["_meta"]["group_id"] in keep]


def ethics_util(n):
    source, dropped, long_, same = "ethics_util_likert", 0, 0, 0
    rng = random.Random(f"{source}:{SEED}")
    rows = []
    for i, r in enumerate(ethics_csv("utilitarianism")):
        better, worse = r["baseline"].strip(), r["less_pleasant"].strip()
        if better == worse: same += 1; continue
        if STRICT.search(better) or STRICT.search(worse): dropped += 1; continue
        if len(better) + len(worse) > MAX_CHARS: long_ += 1; continue
        a_first = rng.random() < 0.5
        a, b = (better, worse) if a_first else (worse, better)
        state = f"Scenario A: {a}\n\nScenario B: {b}\n\nStatement: Scenario A is better for the person than Scenario B."
        rows.append(record(source, f"{source}/{i}", f"{source}/{i}", state, 6 if a_first else 0, subset="utilitarianism", more_pleasant="A" if a_first else "B"))
    print(f"{len(rows)} pairs kept; animal-excluded {dropped}; over {MAX_CHARS} chars (both texts) {long_}; identical {same}")
    return balanced_cut(rows, n, rng)


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
    ap.add_argument("--arm", choices=["ethics", "vitaminc", "ethics-animals", "moralstories-animals", "aisi-likert", "ethics-virtue", "ethics-util"],
                    required=True)
    ap.add_argument("--out")
    ap.add_argument("--n", type=int, default=1000, help="records for ethics-virtue / ethics-util")
    a = ap.parse_args()
    out = Path(a.out or f"runs/gev/hc/data/{'' if a.arm.startswith('moral') else 'likert-'}{a.arm.removeprefix('aisi-likert') and a.arm or 'aisi'}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    if a.arm == "moralstories-animals":
        recs = moralstories()
        write_jsonl(out, recs)
        print(f"{out}: {len(recs)} records, {len(recs) // 2} stories, labels {dict(Counter(r['questions']['better']['label'] for r in recs))}")
        return
    if a.arm == "ethics-animals":
        recs, balanced = ethics_animals(out)
        for r in (recs, balanced): random.Random(f"order:{a.arm}:{SEED}:{len(r)}").shuffle(r)
        if len(balanced) != len(recs):
            write_jsonl(out.with_name(out.stem + "-balanced.jsonl"), balanced)
            print(f"{out.stem}-balanced.jsonl: {len(balanced)} records")
    elif a.arm in ("ethics-virtue", "ethics-util"):
        recs = {"ethics-virtue": ethics_virtue, "ethics-util": ethics_util}[a.arm](a.n)
        random.Random(f"order:{a.arm}:{SEED}").shuffle(recs)
    else:
        recs = {"ethics": ethics, "vitaminc": vitaminc, "aisi-likert": aisi_likert}[a.arm]()
        # fixed order: shuffled by a seeded rng so subsets and labels interleave
        random.Random(f"order:{a.arm}:{SEED}").shuffle(recs)
    write_jsonl(out, recs)
    if a.arm == "aisi-likert":
        print("(harm, direction, label)", dict(sorted(Counter((r["_meta"]["harm"], r["_meta"]["direction"], r["questions"]["likert"]["label"]) for r in recs).items())))
    c = Counter((r["_meta"].get("subset", "all"), r["questions"]["likert"]["label"]) for r in recs)
    print(f"{out}: {len(recs)} records; (subset, label) {dict(sorted(c.items()))}")


if __name__ == "__main__":
    main()
