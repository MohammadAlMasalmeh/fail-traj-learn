#!/usr/bin/env python3
"""Annotation metrics across annotator runs. Standard library only.

Usage: python claude_metrics.py --runs RUN [RUN ...] [--reference ref.json] [--out report.json]
"""
import argparse
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

METRICS = ["outcome_correct", "match_status_correct", "category_recall",
           "category_precision", "failure_frame_iou"]


def load_annotations(run_dirs):
    """Return {('<runlabel>/<TAG>', episode_id): annotation} for answers/<EP>_<TAG>.json."""
    anns = {}
    for run in run_dirs:
        run_path = Path(run).resolve()
        for f in sorted((run_path / "answers").glob("*.json")):
            if f.name.endswith("_validation.json") or "_" not in f.stem:
                continue
            tag = f.stem.rsplit("_", 1)[1]
            with open(f, encoding="utf-8") as fh:
                ann = json.load(fh)
            anns[(f"{run_path.name}/{tag}", ann["episode_id"])] = ann
    return anns


def n_frames(ann):
    return max((s["end_frame"] for s in ann["segments"]), default=0)


def fail_mask(ann, n=None):
    n = n_frames(ann) if n is None else n
    mask = [False] * n
    for s in ann["segments"]:
        if s["failure"]:
            for i in range(s["start_frame"], min(s["end_frame"], n)):
                mask[i] = True
    return mask


def failure_cats(ann):
    return {c for s in ann["segments"] if s["failure"] for c in s["categories"]}


def mean(xs):
    xs = [float(x) for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def iou(m1, m2):
    inter = sum(x and y for x, y in zip(m1, m2))
    union = sum(x or y for x, y in zip(m1, m2))
    return inter / union if union else 1.0


def kappa(m1, m2):
    n = len(m1)
    if n == 0:
        return 1.0
    po = sum(x == y for x, y in zip(m1, m2)) / n
    pa, pb = sum(m1) / n, sum(m2) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    if pe == 1.0:  # zero variance on both sides
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1 - pe)


def levenshtein(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def annotation_metrics(key, ann):
    segs = ann["segments"]
    n = n_frames(ann)
    fail_frames = sum(s["end_frame"] - s["start_frame"] for s in segs if s["failure"])
    review = sum(bool(s["review_required"]) for s in segs)
    return {
        "key": key,
        "episode_id": ann["episode_id"],
        "n_segments": len(segs),
        "n_action_ids": len({s["action_id"] for s in segs}),
        "action_sequence": ">".join(s["action"] for s in segs),
        "failure_frac": fail_frames / n if n else 0.0,
        "failure_categories": sorted(failure_cats(ann)),
        "match_status": ann["identity_check"]["match_status"],
        "review_required_frac": review / len(segs) if segs else 0.0,
        "episode_failure": any(s["failure"] for s in segs),
    }


def pairwise(anns):
    by_ep = defaultdict(list)
    for key, ep in anns:
        by_ep[ep].append(key)
    rows = []
    for ep in sorted(by_ep):
        for k1, k2 in combinations(sorted(by_ep[ep]), 2):
            a1, a2 = anns[(k1, ep)], anns[(k2, ep)]
            n = min(n_frames(a1), n_frames(a2))  # compare over shared frame prefix
            m1, m2 = fail_mask(a1, n), fail_mask(a2, n)
            c1, c2 = failure_cats(a1), failure_cats(a2)
            t1 = [s["action"] for s in a1["segments"]]
            t2 = [s["action"] for s in a2["segments"]]
            longest = max(len(t1), len(t2))
            rows.append({
                "episode_id": ep,
                "a": k1,
                "b": k2,
                "frame_agreement": sum(x == y for x, y in zip(m1, m2)) / n if n else 1.0,
                "failure_iou": iou(m1, m2),
                "kappa": kappa(m1, m2),
                "category_jaccard": len(c1 & c2) / len(c1 | c2) if (c1 | c2) else 1.0,
                "match_status_equal": (a1["identity_check"]["match_status"]
                                       == a2["identity_check"]["match_status"]),
                "action_edit_distance": levenshtein(t1, t2) / longest if longest else 0.0,
            })
    return rows


def reference_metrics(anns, ref):
    rows = []
    for (key, ep), ann in sorted(anns.items()):
        r = ref.get(ep)
        if r is None:
            continue
        cats = r.get("failure_categories", [])
        pred, gold = failure_cats(ann), set(cats or [])
        inter = len(pred & gold)
        ep_fail = any(s["failure"] for s in ann["segments"])
        row = {
            "key": key,
            "episode_id": ann["episode_id"],
            "outcome_correct": (r["outcome"] == "failure") == ep_fail,
            # A null reference field means "not verified"; it is left unscored.
            "match_status_correct": None if r.get("match_status") is None
            else ann["identity_check"]["match_status"] == r["match_status"],
            # Empty-set convention: nothing to find and nothing predicted scores 1.0.
            "category_recall": None if cats is None else (inter / len(gold) if gold else float(not pred)),
            "category_precision": None if cats is None else (inter / len(pred) if pred else float(not gold)),
            "failure_frame_iou": None,
        }
        if "failure_frames" in r:
            n = n_frames(ann)
            gold_mask = [False] * n
            for s, e in r["failure_frames"]:
                for i in range(max(s, 0), min(e, n)):
                    gold_mask[i] = True
            row["failure_frame_iou"] = iou(fail_mask(ann), gold_mask)
        rows.append(row)
    return rows


def aggregate(rows):
    out = []
    for key in sorted({r["key"] for r in rows}):
        rs = [r for r in rows if r["key"] == key]
        out.append({"key": key, "n": len(rs),
                    **{m: mean([r[m] for r in rs]) for m in METRICS}})
    return out


def fmt(x):
    if x is None:
        return "-"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, float):
        return f"{x:.3f}"
    if isinstance(x, list):
        return ",".join(x) or "-"
    return str(x)


def md_table(rows, cols):
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="run dirs containing answers/")
    ap.add_argument("--reference", help="reference JSON keyed by episode_id")
    ap.add_argument("--out", help="output JSON (default <first run>/metrics.json)")
    args = ap.parse_args()

    anns = load_annotations(args.runs)
    report = {
        "annotations": [annotation_metrics(k, a) for (k, _), a in sorted(anns.items())],
        "pairwise": pairwise(anns),
    }
    md = [
        "## Annotations",
        md_table(report["annotations"], ["key", "episode_id", "n_segments", "n_action_ids",
                                         "failure_frac", "failure_categories", "match_status",
                                         "review_required_frac", "episode_failure"]),
        "",
        "## Pairwise",
        md_table(report["pairwise"], ["episode_id", "a", "b", "frame_agreement", "failure_iou",
                                      "kappa", "category_jaccard", "match_status_equal",
                                      "action_edit_distance"]),
    ]
    if args.reference:
        with open(args.reference, encoding="utf-8") as fh:
            ref = json.load(fh)
        rows = reference_metrics(anns, ref)
        agg = aggregate(rows)
        report["reference"] = {"per_annotation": rows, "aggregate": agg}
        md += ["", "## Reference (per annotator)", md_table(agg, ["key", "n"] + METRICS)]

    out = Path(args.out) if args.out else Path(args.runs[0]).resolve() / "metrics.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n".join(md))
    print(f"\nFull JSON written to {out}")


if __name__ == "__main__":
    main()
