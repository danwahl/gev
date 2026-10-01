"""Screen a training JSONL against the evaluation texts Gev must never train on.

    uv run python scripts/screen_gev_overlap.py DATA.jsonl [--drop OUT.jsonl] [--specieval DIR]

Unlike scripts/screen_overlap.py, which reports per-partition 8-gram Jaccard counts of a generated suite against JevBench's
public items and never drops anything, this screens a training file for verbatim reuse of short eval strings and writes the
clean rest with --drop.

Eval texts: SpeciEval items (every language, forward and reverse; string_id spec_ / bfas_ / la4N_ / se4N_; --specieval,
default /home/dan/specieval), the states of evals/gev/{speciesismbench-v1 (development, controls), tac-v1, attitudes-heldout-v1}
and of aisi-animal-v1 / aisi-harm-v1 (development, calibration, test only: their train partitions are what AISI-derived training
may use; harm states are also split into their "Request A: ..." / "Request B: ..." lines), plus every question instruction and
option text of those suites (4 words or more; shorter ones are generic). Everything is NFKC-normalised, casefolded and
whitespace-collapsed first. A record's state is flagged when it shares an 8-word sequence with an eval text, contains one
(3 words or 12 characters or more) verbatim, shares a 12-character run with a CJK / Thai / kana / hangul eval text (\\w+ does not
split those scripts), or its character 5-gram Jaccard with one exceeds 0.5. Prints counts and the first flags; --drop writes the rest.
"""
import argparse
import csv
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from kev.suite import read_jsonl, write_jsonl

PREFIXES = ("spec_", "bfas_", "la4N_", "se4N_")
SUITES = [("speciesismbench-v1", ("development", "controls")), ("tac-v1", ("development",)), ("attitudes-heldout-v1", ("development",)),
          ("aisi-animal-v1", ("development", "calibration", "test")), ("aisi-harm-v1", ("development", "calibration", "test"))]
WORDS, SHINGLE, RUN, JACCARD, MIN_QUESTION_WORDS = 8, 5, 12, 0.5, 4
NO_SPACES = re.compile(r"[⺀-鿿가-힯豈-﫿฀-๿぀-ヿ]")
REQUEST = re.compile(r"(?:^|\n)\s*Request [AB]: ")


def norm(text): return " ".join(unicodedata.normalize("NFKC", text).casefold().split())
def words(text): return re.findall(r"\w+", text)
def grams(text, n=WORDS): w = words(text); return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}
def chars(text): t = " ".join(words(text)); return {t[i:i + SHINGLE] for i in range(len(t) - SHINGLE + 1)}
def runs(text): t = "".join(text.split()); return {t[i:i + RUN] for i in range(len(t) - RUN + 1)}
def long_enough(text): return len(words(text)) >= 3 or len(text) >= RUN


def question_texts(q):
    out = [q.get("instructions")]
    c = q.get("criteria")
    out += list(c.values()) if isinstance(c, dict) else c if isinstance(c, list) else []
    return [t for t in out if isinstance(t, str) and len(words(t)) >= MIN_QUESTION_WORDS]


def eval_texts(root, specieval):
    out = []
    data = Path(specieval) / "src/specieval/translations/data"
    for name in ("translations.csv", "translations-reverse.csv"):
        if not (data / name).exists(): print(f"missing {data / name}"); continue
        with (data / name).open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row["string_id"].startswith(PREFIXES):
                    out += [(f"specieval/{name}/{row['string_id']}/{k}", v) for k, v in row.items() if k != "string_id" and v and v.strip()]
    for suite, parts in SUITES:
        for part in parts:
            p = root / "evals/gev" / suite / f"{part}.jsonl"
            if not p.exists(): print(f"missing {p}"); continue
            for n, r in enumerate(read_jsonl(p)):
                src = f"{suite}/{part}/{r.get('id') or r.get('_meta', {}).get('id') or n}"
                s = r.get("state") or r.get("statement")
                if isinstance(s, str) and s.strip():
                    out.append((src, s))
                    if suite == "aisi-harm-v1": out += [(src + "/line", t) for t in REQUEST.split(s) if t.strip()]
                for k, q in (r.get("questions") or {}).items(): out += [(f"{src}/{k}", t) for t in question_texts(q)]
    seen, unique = set(), []
    for src, t in out:
        t = norm(t)
        if t and t not in seen: seen.add(t); unique.append((src, t))
    return unique


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--drop")
    ap.add_argument("--specieval", default="/home/dan/specieval")
    a = ap.parse_args()
    texts = eval_texts(Path(__file__).resolve().parent.parent, a.specieval)
    seq, sets, runs_index, contain = defaultdict(set), [], defaultdict(set), []
    for i, (_, t) in enumerate(texts):
        for g in grams(t): seq[g].add(i)
        sets.append(chars(t))
        if NO_SPACES.search(t):
            for g in runs(t): runs_index[g].add(i)
        if long_enough(t): contain.append(i)
    ngram_index = defaultdict(set)
    for i, s in enumerate(sets):
        for g in s: ngram_index[g].add(i)
    recs = list(read_jsonl(a.data))
    flags, kept = [], []
    for n, r in enumerate(recs):
        raw = r["state"] if isinstance(r["state"], str) else json.dumps(r["state"], ensure_ascii=False)
        state = norm(raw)
        hit, why = None, None
        for g in grams(state):
            if g in seq: hit, why = min(seq[g]), "8-word"; break
        if hit is None:
            hit = next((i for i in contain if texts[i][1] in state), None)
            if hit is not None: why = "contains"
        if hit is None and runs_index and NO_SPACES.search(state):
            for g in runs(state):
                if g in runs_index: hit, why = min(runs_index[g]), "12-char run"; break
        if hit is None:
            c, counts = chars(state), Counter()
            for g in c:
                for i in ngram_index.get(g, ()): counts[i] += 1
            for i, k in counts.items():
                if k / (len(c) + len(sets[i]) - k) > JACCARD: hit, why = i, "5-gram jaccard"; break
        if hit is None: kept.append(r)
        else: flags.append((n, why, texts[hit][0], raw[:120].replace("\n", " "), texts[hit][1][:120]))
    print(f"{a.data}: {len(recs)} records against {len(texts)} eval texts: {len(flags)} flagged {dict(Counter(f[1] for f in flags))}, {len(kept)} kept")
    for n, why, src, s, e in flags[:5]: print(f"  row {n} [{why}] vs {src}\n    record: {s}\n    eval:   {e}")
    if a.drop:
        write_jsonl(a.drop, kept)
        print(f"wrote {len(kept)} records to {a.drop}")


if __name__ == "__main__":
    main()
