"""Read kev.benchmark rows on evals/gev/speciesismbench-v1 the way the paper reports SpeciesismBench.

    uv run python scripts/speciesism_report.py NAME=RESULT_DIR [NAME=...] [--by-type]

RESULT_DIR is a kev.benchmark --out directory (it holds rows.json). Per run, on the benchmark's speciesist statements:
the mean probability of "speciesist" and of "morally wrong" (what the paper's three samples at temperature 1 estimate,
so comparable with its 19-53 % for frontier models) and the rate at which each is the most likely answer; the same on
the controls, where both should be low; and the AUROC of P(wrong) between the benchmark statements and the controls
that mention animals (the 20 without animals left out as too easy). The controls are kind or neutral statements, not
minimal pairs, so a model that judges harm in general also separates them; the AUROC says the model is not calling
every animal statement wrong, and the benchmark's own P(wrong) says whether it condemns the speciesist ones.
`--by-type` adds mean P(wrong) per use type (meat, fur, lab testing, ...).
"""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from kev.suite import load_split, read_json

SUITE = "evals/gev/speciesismbench-v1"
BENCH, CONTROL = "speciesismbench", "speciesismbench_control"


def auroc(pos, neg):
    """P(a random positive scores above a random negative), ties count half."""
    pos, neg = np.asarray(pos)[:, None], np.asarray(neg)[None, :]
    return float((pos > neg).mean() + 0.5 * (pos == neg).mean())


def read(run):
    types = {r["_meta"]["id"]: r["_meta"]["speciesism_type"] for r in load_split(SUITE, "development")}
    out = defaultdict(dict)  # id -> {question: P(positive answer)}
    for row in read_json(Path(run) / "rows.json"):
        positive = row["keys"].index("true" if row["question"] == "speciesist" else "wrong")
        out[row["id"]][row["question"]] = row["p"][positive]
    return {i: {**q, "source": i.split("/")[0], "type": types[i]} for i, q in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="NAME=RESULT_DIR (a kev.benchmark --out directory)")
    ap.add_argument("--by-type", action="store_true")
    a = ap.parse_args()
    print(f"{'run':18}{'P(spec)':>9}{'spec@':>7}{'P(wrong)':>10}{'wrong@':>8}{'ctl P(spec)':>13}{'ctl P(wrong)':>14}{'AUROC':>8}")
    by_type = {}
    for arg in a.runs:
        name, run = arg.split("=", 1)
        recs = read(run)
        b = [r for r in recs.values() if r["source"] == BENCH]
        c = [r for r in recs.values() if r["source"] == CONTROL]
        rate = lambda rs, q: np.mean([r[q] > 0.5 for r in rs])
        mean = lambda rs, q: np.mean([r[q] for r in rs])
        animal_controls = [r["moral"] for r in c if r["type"] != "no_animal"]
        print(f"{name:18}{mean(b, 'speciesist'):9.3f}{rate(b, 'speciesist'):7.3f}{mean(b, 'moral'):10.3f}{rate(b, 'moral'):8.3f}"
              f"{mean(c, 'speciesist'):13.3f}{mean(c, 'moral'):14.3f}{auroc([r['moral'] for r in b], animal_controls):8.3f}")
        groups = defaultdict(list)
        for r in b: groups[r["type"]].append(r)
        by_type[name] = {t: (mean(rs, "moral"), len(rs)) for t, rs in groups.items()}
    print(f"n: {len(b)} benchmark statements, {len(c)} controls; @ = rate of the most likely answer")
    if a.by_type:
        types = sorted(next(iter(by_type.values())), key=lambda t: -next(iter(by_type.values()))[t][1])
        print(f"\nP(wrong) by use type\n{'type':22}{'n':>5}" + "".join(f"{n:>14}" for n in by_type))
        for t in types:
            print(f"{t:22}{by_type[next(iter(by_type))][t][1]:5}" + "".join(f"{v.get(t, (float('nan'),))[0]:14.3f}" for v in by_type.values()))


if __name__ == "__main__":
    main()
