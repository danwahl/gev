"""Build evals/gev/aisi-animal-v1: one yes/no question per user question, "does it (or a helpful answer to it) involve
living animals or animal products?", over AISI's ab_animal_welfare_questions.

    uv run python scripts/build_gev_aisi_animal.py [--out evals/gev/aisi-animal-v1]

Source: ai-safety-institute/ab_animal_welfare_questions at a pinned revision (2,000 short questions in 20 sub-categories;
no dataset card or licence, so the partitions stay out of git and this script rebuilds them byte for byte). AISI wrote
the questions as prompts for a model organism that covertly steers answers away from animal harm; only the questions
are used here.

Labels come from `labels.jsonl` next to the manifest (ids only, no text), with how each was decided in `method`:
`rubric` (read one by one under the project owner's rule: any real animal of any species counts, including as food,
product, pest, research subject or abstract welfare topic), `owner` (the owner's call on a group the rule left open:
abstract animal welfare, plant-based substitutes, abstract conservation, circus history), `poll` (diet questions that
name no animal product: yes when most of five Qwen3.5-4B answers name one; `poll_share` is the fraction) or `excluded`.
`group` joins paraphrases and template siblings (reviewer-flagged pairs plus content-word Jaccard >= 0.6), and a group
never straddles partitions.

Split: three whole sub-categories are held out (development and test only, by group), so the read covers phrasings no
training record shares; the rest are split by group into train 70 %, calibration 5 %, development 10 %, test 15 %.
"""
import argparse
import hashlib
from collections import Counter
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

from kev.suite import CONTEXT, digest, read_jsonl, record_digest, text_digest, write_json, write_jsonl

REPO, REVISION = "ai-safety-institute/ab_animal_welfare_questions", "062b384d8bca231af39e934888739cd87c7cc0f6"
FILE = "data/train-00000-of-00001.parquet"
SOURCE = "aisi_animal"
INSTRUCTIONS = "Does this question, or a helpful answer to it, involve living animals or animal products?"
HELD_OUT = ("cruel delicacies", "animal entertainment", "meta-probe: hypothetical scenario")
SHARES = (("train", 0.70), ("calibration", 0.05), ("development", 0.10), ("test", 0.15))
SEED = "gev-aisi-animal-v1"


def unit(key):
    """A stable number in [0, 1) for a group, from its id and the build's seed."""
    return int(hashlib.sha256(f"{SEED}:{key}".encode()).hexdigest()[:12], 16) / 16 ** 12


def partition(group, held_out):
    u = unit(group)
    if held_out:
        return "development" if u < 0.5 else "test"
    for split, share in SHARES:
        if u < share: return split
        u -= share
    return SHARES[-1][0]


def record(row, label):
    rec = {"state": row["question"],
           "questions": {"animals": {"type": "noul", "instructions": INSTRUCTIONS, "label": label["label"], "src": SOURCE}},
           "_meta": {"row": row["id"], "source": SOURCE, "repo": REPO, "revision": REVISION, "id": f"{SOURCE}/{row['id']}",
                     "group_id": f"{SOURCE}/{label['group']}", "variant": "clean", "sub_category": row["sub_category"],
                     "method": label["method"], "text_sha256": text_digest(row["question"])}}
    if "poll_share" in label: rec["_meta"]["poll_share"] = label["poll_share"]
    rec["_meta"]["row_sha256"] = record_digest({k: v for k, v in rec.items() if k != "_meta"})
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="evals/gev/aisi-animal-v1")
    a = ap.parse_args()
    out = Path(a.out)
    labels = {r["id"]: r for r in read_jsonl(out / "labels.jsonl")}
    path = hf_hub_download(REPO, FILE, repo_type="dataset", revision=REVISION)
    rows = pd.read_parquet(path).to_dict("records")
    if {r["id"] for r in rows} != set(labels):
        raise ValueError("labels.jsonl does not cover the pinned revision's ids exactly")
    # a group touching a held-out sub-category goes with it
    held_groups = {labels[r["id"]]["group"] for r in rows if r["sub_category"] in HELD_OUT}
    parts = {s: [] for s, _ in SHARES}
    for r in sorted(rows, key=lambda r: r["id"]):
        label = labels[r["id"]]
        if label["label"] is None: continue
        parts[partition(label["group"], label["group"] in held_groups)].append(record(r, label))
    files = {}
    for split, recs in parts.items():
        write_jsonl(out / f"{split}.jsonl", recs)
        files[f"{split}.jsonl"] = {"sha256": digest(out / f"{split}.jsonl"), "records": len(recs)}
    write_json(out / "manifest.json", {
        "version": 1,
        "external": {"repo": REPO, "revision": REVISION, "file": FILE, "sha256": digest(path), "license": None},
        "labels": {"file": "labels.jsonl", "sha256": digest(out / "labels.jsonl")},
        "base_revisions": {}, "dataset_revisions": {REPO: REVISION}, "holdout_sources": [], "trainable_sources": [SOURCE],
        "eval_only_sources": [], "held_out_sub_categories": list(HELD_OUT), "context": CONTEXT, "files": files,
        "protocol": {"note": "held-out sub-categories appear only in development and test; groups never straddle partitions; "
                             "partitions are rebuilt by scripts/build_gev_aisi_animal.py (no licence on the source, so not in git)"}})
    for split, recs in parts.items():
        c = Counter(r["questions"]["animals"]["label"] for r in recs)
        held = sum(r["_meta"]["sub_category"] in HELD_OUT for r in recs)
        print(f"{split:12}{len(recs):5}  yes {c[True]:4}  no {c[False]:4}  held-out sub-categories {held}")


if __name__ == "__main__":
    main()
