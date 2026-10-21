"""Failure-focused scoring against teacher, simulator outcome, and frame-verified labels.
  score_failures.py --tags base sft sft_unmasked [--common]

Verified labels live in outputs/distill/eval_verified.json (frame-checked adjudication of the
held-out set; see docs). Writes outputs/distill/scores_failures/{summary.json,summary.md}.
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data
from eval import parse_pred, usable_segments, load_preds

OUT = os.path.join(data.OUT_DIR, "scores_failures")
VERIFIED = os.path.join(data.OUT_DIR, "eval_verified.json")
TOL = 10  # frames; teacher sees every 5th frame, student every 15th


def pred_fail_segs(rec):
    o = parse_pred(rec.get("text", ""))
    if o is None:
        return None
    return [s for s in usable_segments(o) if s.get("failure") is True]


def near(a, b, tol=TOL):
    return a["start_frame"] - tol <= b["end_frame"] and b["start_frame"] - tol <= a["end_frame"]


def cats(s):
    c = s.get("categories")
    return set(c) if isinstance(c, list) else set()


def frame_set(segs):
    return {f for s in segs for f in range(s["start_frame"], s["end_frame"] + 1)}


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def episode_metrics(pred_has, ref_has):
    tp = sum(p and r for p, r in zip(pred_has, ref_has))
    fp = sum(p and not r for p, r in zip(pred_has, ref_has))
    fn = sum(r and not p for p, r in zip(pred_has, ref_has))
    tn = len(ref_has) - tp - fp - fn
    p, r, f = prf(tp, fp, fn)
    return dict(acc=(tp + tn) / len(ref_has), precision=p, recall=r, f1=f, tp=tp, fp=fp, fn=fn, tn=tn)


def boot_f1(pred_has, ref_has, nb=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(ref_has); v = []
    for _ in range(nb):
        ix = rng.integers(0, n, n)
        v.append(episode_metrics([pred_has[i] for i in ix], [ref_has[i] for i in ix])["f1"])
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def score_tag(tag, ids, V):
    P = load_preds(tag, ids)
    rows = []
    for i in ids:
        ps = pred_fail_segs(P[i])
        rows.append(dict(id=i, parsed=ps is not None, pred=ps or []))
    pred_has = [bool(r["pred"]) for r in rows]
    res = {"n": len(ids), "parse_rate": sum(r["parsed"] for r in rows) / len(ids),
           "frac_episodes_predicting_failure": sum(pred_has) / len(ids),
           "pred_fail_segments": sum(len(r["pred"]) for r in rows)}
    for ref in ("verified", "simulator", "teacher"):
        ref_has = [V[i][ref + "_has_failure"] for i in ids]
        m = episode_metrics(pred_has, ref_has)
        m["f1_ci95"] = boot_f1(pred_has, ref_has)
        res["episode_vs_" + ref] = m
    # segment level vs verified reference segments
    tp_ref = n_ref = tp_pred = n_pred = cat_ok = 0
    by_cat = {}
    ious = []
    for r in rows:
        refs = V[r["id"]]["ref_segments"]
        hit_refs = [any(near(q, p) for p in r["pred"]) for q in refs]
        n_ref += len(refs); tp_ref += sum(hit_refs)
        n_pred += len(r["pred"]); tp_pred += sum(any(near(p, q) for q in refs) for p in r["pred"])
        for q, h in zip(refs, hit_refs):
            for c in q["categories"] or ["none"]:
                d = by_cat.setdefault(c, [0, 0]); d[0] += h; d[1] += 1
            if h and any(cats(q) & cats(p) for p in r["pred"] if near(q, p)):
                cat_ok += 1
        if V[r["id"]]["segments_known"] and (refs or r["pred"]):
            a, b = frame_set(refs), frame_set(r["pred"])
            ious.append(len(a & b) / len(a | b) if a | b else 1.0)
    res["segment"] = dict(ref_segments=n_ref, recall=tp_ref / n_ref if n_ref else 0.0,
                          precision=tp_pred / n_pred if n_pred else 0.0,
                          category_match_on_hits=cat_ok / tp_ref if tp_ref else 0.0,
                          recall_by_category={c: dict(hit=h, n=n, recall=h / n) for c, (h, n) in sorted(by_cat.items())},
                          frame_iou_mean=float(np.mean(ious)) if ious else 0.0, iou_episodes=len(ious),
                          tolerance_frames=TOL)
    return res, rows


def md(summary):
    tags = list(summary["tags"])
    L = [f"# Failure scoring (n = {summary['n']} held-out episodes{', common set' if summary['common'] else ''})", ""]
    L.append("| metric | " + " | ".join(tags) + " |"); L.append("|---|" + "---|" * len(tags))
    def row(name, f):
        L.append(f"| {name} | " + " | ".join(f(summary["tags"][t]) for t in tags) + " |")
    row("n", lambda r: str(r["n"]))
    row("valid JSON", lambda r: f"{r['parse_rate']:.2f}")
    row("episodes predicted as failure", lambda r: f"{r['frac_episodes_predicting_failure']:.2f}")
    for ref in ("verified", "simulator", "teacher"):
        k = "episode_vs_" + ref
        row(f"**episode F1 vs {ref}** [95% CI]", lambda r, k=k: f"**{r[k]['f1']:.2f}** [{r[k]['f1_ci95'][0]:.2f}, {r[k]['f1_ci95'][1]:.2f}]")
        row(f"episode P / R vs {ref}", lambda r, k=k: f"{r[k]['precision']:.2f} / {r[k]['recall']:.2f}")
        row(f"episode acc vs {ref}", lambda r, k=k: f"{r[k]['acc']:.2f}")
    row("segment recall (±10 fr)", lambda r: f"{r['segment']['recall']:.2f} ({r['segment']['ref_segments']} ref)")
    row("segment precision", lambda r: f"{r['segment']['precision']:.2f}")
    row("category match on hits", lambda r: f"{r['segment']['category_match_on_hits']:.2f}")
    row("failure frame IoU", lambda r: f"{r['segment']['frame_iou_mean']:.2f}")
    cats_all = sorted({c for t in tags for c in summary["tags"][t]["segment"]["recall_by_category"]})
    for c in cats_all:
        row(f"recall: {c}", lambda r, c=c: (lambda d: f"{d['hit']}/{d['n']}" if d else "-")(r["segment"]["recall_by_category"].get(c)))
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--common", action="store_true", help="restrict to episodes every tag has predictions for")
    a = ap.parse_args()
    V = json.load(open(VERIFIED))["episodes"]
    ids = sorted(V)
    if a.common:
        ids = [i for i in ids if all(os.path.exists(os.path.join(data.OUT_DIR, "preds", t, i + ".json")) for t in a.tags)]
    os.makedirs(OUT, exist_ok=True)
    summary = {"n": len(ids), "common": a.common, "tags": {}}
    for t in a.tags:
        tids = ids if a.common else [i for i in ids if os.path.exists(os.path.join(data.OUT_DIR, "preds", t, i + ".json"))]
        res, rows = score_tag(t, tids, V)
        summary["tags"][t] = res
        with open(os.path.join(OUT, f"{t}_per_episode.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps({"id": r["id"], "parsed": r["parsed"], "n_pred_fail": len(r["pred"]),
                                    "pred_fail": [[s["start_frame"], s["end_frame"], sorted(cats(s))] for s in r["pred"]]}) + "\n")
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    open(os.path.join(OUT, "summary.md"), "w").write(md(summary))
    print(md(summary))


if __name__ == "__main__":
    main()
