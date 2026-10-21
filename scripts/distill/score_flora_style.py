"""Flora-style metrics (NVIDIA GEAR, "FLORA: An Open Model Suite for Scaling Robotics Data Engines", 2026)
applied to our held-out set.
  score_flora_style.py --tags base sft sft_unmasked sft_v3 [--teacher]

Segmentation F1@tIoU (DenseCaptionBench/WGO-Bench, App. B.3): an edge is eligible when tIoU >= t;
caption-blind maximum-cardinality one-to-one matching gives TP; F = 2TP / (|pred| + |ref|), averaged per episode.
Dense failure labels (DenseRewardBench, App. C): per-step labels compared at 1 Hz, pooled over all steps;
macro F1, balanced accuracy, failure recall. Our schema has no "suboptimal" class, so this is binary
(normal vs failure). Failure references are the frame-verified eval labels (eval_verified.json).
Writes outputs/distill/scores_flora/summary.{json,md}.
"""
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data
from eval import parse_pred, usable_segments, load_preds

OUT = os.path.join(data.OUT_DIR, "scores_flora")
VERIFIED = os.path.join(data.OUT_DIR, "eval_verified.json")


def tiou(a, b):
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    inter = max(0, hi - lo)
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def max_matching(adj, n_right):
    """Maximum-cardinality bipartite matching (Kuhn's augmenting paths)."""
    match_r = [-1] * n_right

    def aug(u, seen):
        for v in adj[u]:
            if v in seen:
                continue
            seen.add(v)
            if match_r[v] < 0 or aug(match_r[v], seen):
                match_r[v] = u
                return True
        return False

    return sum(aug(u, set()) for u in range(len(adj)))


def seg_f1(pred, ref, thr):
    P = [(s["start_frame"], s["end_frame"]) for s in pred]
    R = [(s["start_frame"], s["end_frame"]) for s in ref]
    if not P and not R:
        return 1.0
    adj = [[j for j, r in enumerate(R) if tiou(p, r) >= thr] for p in P]
    return 2 * max_matching(adj, len(R)) / (len(P) + len(R))


def step_labels(fail_segs, n_frames, fps):
    """1 Hz steps: step k covers frames [k*fps, (k+1)*fps); failure if it overlaps any failure segment."""
    steps = []
    for k in range(int(n_frames // fps) + (n_frames % fps > 0)):
        a, b = k * fps, min((k + 1) * fps, n_frames) - 1
        steps.append(any(s["start_frame"] <= b and a <= s["end_frame"] for s in fail_segs))
    return steps


def dense_metrics(pred, ref):
    tp = sum(p and r for p, r in zip(pred, ref)); fp = sum(p and not r for p, r in zip(pred, ref))
    fn = sum(r and not p for p, r in zip(pred, ref)); tn = len(ref) - tp - fp - fn
    f1 = lambda tp, fp, fn: 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0
    rec_f = tp / (tp + fn) if tp + fn else 0.0
    rec_n = tn / (tn + fp) if tn + fp else 0.0
    return dict(macro_f1=(f1(tp, fp, fn) + f1(tn, fn, fp)) / 2, balanced_acc=(rec_f + rec_n) / 2,
                failure_f1=f1(tp, fp, fn), failure_recall=rec_f, normal_flagged_as_failure=1 - rec_n,
                steps=len(ref), failure_steps=tp + fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--teacher", action="store_true", help="also score the teacher's own labels as a row")
    a = ap.parse_args()
    V = json.load(open(VERIFIED))["episodes"]
    recs = {r["id"]: r for r in data.load_records() if r["id"] in V}
    T = {i: json.loads(r["messages"][1]["content"])["segments"] for i, r in recs.items()}
    tags = (["teacher"] if a.teacher else []) + a.tags
    summary = {}
    for tag in tags:
        if tag == "teacher":
            P = {i: T[i] for i in V}; ids = sorted(V)
        else:
            ids = [i for i in sorted(V) if os.path.exists(os.path.join(data.OUT_DIR, "preds", tag, i + ".json"))]
            raw = load_preds(tag, ids)
            P = {}
            for i in ids:
                o = parse_pred(raw[i].get("text", ""))
                P[i] = usable_segments(o) if o else []
        f05 = [seg_f1(P[i], T[i], 0.5) for i in ids]
        f075 = [seg_f1(P[i], T[i], 0.75) for i in ids]
        pr, rf = [], []
        for i in ids:
            n, fps = recs[i]["metadata"]["n_frames"], recs[i]["metadata"]["fps"]
            pr += step_labels([s for s in P[i] if s.get("failure") is True], n, fps)
            rf += step_labels(V[i]["ref_segments"], n, fps)
        summary[tag] = dict(n=len(ids), seg_f1_05=sum(f05) / len(ids), seg_f1_075=sum(f075) / len(ids),
                            dense=dense_metrics(pr, rf))
    os.makedirs(OUT, exist_ok=True)
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    L = ["| metric (x100) | " + " | ".join(tags) + " |", "|---|" + "---|" * len(tags)]
    rows = [("n episodes", lambda s: str(s["n"])),
            ("Segmentation F1@0.5 vs teacher", lambda s: f"{100 * s['seg_f1_05']:.1f}"),
            ("Segmentation F1@0.75 vs teacher", lambda s: f"{100 * s['seg_f1_075']:.1f}"),
            ("Dense (1 Hz) macro F1", lambda s: f"{100 * s['dense']['macro_f1']:.1f}"),
            ("Dense balanced acc.", lambda s: f"{100 * s['dense']['balanced_acc']:.1f}"),
            ("Dense failure F1", lambda s: f"{100 * s['dense']['failure_f1']:.1f}"),
            ("Dense failure recall", lambda s: f"{100 * s['dense']['failure_recall']:.1f}"),
            ("Normal steps flagged as failure", lambda s: f"{100 * s['dense']['normal_flagged_as_failure']:.1f}")]
    for name, f in rows:
        L.append(f"| {name} | " + " | ".join(f(summary[t]) for t in tags) + " |")
    d = summary[tags[-1]]["dense"]
    L.append(f"\nDense steps: {d['steps']} at 1 Hz, {d['failure_steps']} failure (verified references).")
    open(os.path.join(OUT, "summary.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
