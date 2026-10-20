"""Before/after evaluation harness.
  eval.py generate --adapter {none|path} --tag TAG [--limit N] [--max-new-tokens 5120]
  eval.py score --tags base sft [--skip-missing]
"""
import argparse, json, os, re, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data

PREDS = os.path.join(data.OUT_DIR, "preds")
SCORES = os.path.join(data.OUT_DIR, "scores")
SEG_KEYS = ["start_frame", "end_frame", "action_id", "action", "object_id", "destination_id", "observed_result",
            "boundary_reason", "failure", "what_happened", "categories", "recovery", "severity", "resetability",
            "evidence", "review_required", "uncertainty"]
NULLABLE = {"object_id", "destination_id", "recovery", "severity", "resetability"}
SEG_TYPES = {"start_frame": int, "end_frame": int, "action_id": str, "action": str, "object_id": str,
             "destination_id": str, "observed_result": str, "boundary_reason": str, "failure": bool,
             "what_happened": str, "categories": list, "recovery": str, "severity": str, "resetability": str,
             "evidence": list, "review_required": bool, "uncertainty": str}
TOP_KEYS = ["episode_id", "identity_check", "segments"]


# ----------------------------------------------------------------- generation
def cmd_generate(a):
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList
    from model import load_model, load_processor
    ids = data.load_split()["eval"]
    if a.limit:
        ids = ids[: a.limit]
    out_dir = os.path.join(PREDS, a.tag)
    os.makedirs(out_dir, exist_ok=True)
    todo = [i for i in ids if not os.path.exists(os.path.join(out_dir, i + ".json"))]
    print(f"[gen] tag={a.tag} adapter={a.adapter} total={len(ids)} todo={len(todo)}", flush=True)
    if not todo:
        return
    recs = {r["id"]: r for r in data.load_records()}
    proc = load_processor()
    adapter = None if a.adapter in (None, "none", "") else a.adapter
    model = load_model(adapter_path=adapter).eval()

    class EOT(StoppingCriteria):
        def __call__(self, input_ids, scores, **kw):
            return input_ids[:, -1] == data.EOT_ID

    for n, eid in enumerate(todo):
        t0 = time.time()
        rec = {"id": eid, "tag": a.tag, "adapter": adapter, "max_new_tokens": a.max_new_tokens}
        try:
            msgs = data.build_student_messages(recs[eid], cfg=data.DEFAULT_CFG)
            inp = data.encode_prompt(proc, msgs, cfg=data.DEFAULT_CFG).to("cuda")
            inp["pixel_values"] = inp["pixel_values"].to(torch.bfloat16)
            P = inp["input_ids"].shape[1]
            with torch.no_grad():
                out = model.generate(**inp, max_new_tokens=a.max_new_tokens, do_sample=False,
                                     stopping_criteria=StoppingCriteriaList([EOT()]))
            gen = out[0, P:]
            rec.update(text=proc.tokenizer.decode(gen, skip_special_tokens=True), n_prompt_tokens=int(P),
                       n_tokens=int(gen.shape[0]),
                       hit_max=bool(gen.shape[0] >= a.max_new_tokens and int(gen[-1]) != data.EOT_ID), error=None)
            del out, inp, gen
        except torch.cuda.OutOfMemoryError as e:
            rec.update(text="", n_tokens=0, hit_max=False, error="OOM: " + str(e)[:200])
        except Exception as e:  # keep going on odd failures
            rec.update(text="", n_tokens=0, hit_max=False, error=f"{type(e).__name__}: {str(e)[:200]}")
        import gc
        gc.collect(); torch.cuda.empty_cache()
        rec["seconds"] = round(time.time() - t0, 2)
        tmp = os.path.join(out_dir, eid + ".json.tmp")
        json.dump(rec, open(tmp, "w"))
        os.replace(tmp, os.path.join(out_dir, eid + ".json"))
        print(f"[gen] {n+1}/{len(todo)} {eid} tokens={rec['n_tokens']} {rec['seconds']}s err={rec['error']}", flush=True)


# -------------------------------------------------------------------- parsing
def parse_pred(text):
    """Lenient: strip ``` fences, take outermost {...}. Returns dict or None."""
    if not text:
        return None
    t = re.sub(r"```(?:json)?", "", text)
    i, j = t.find("{"), t.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        o = json.loads(t[i:j + 1])
    except Exception:
        return None
    return o if isinstance(o, dict) else None


def _tok(v, t):
    if t is bool:
        return isinstance(v, bool)
    if t is int:
        return isinstance(v, int) and not isinstance(v, bool)
    return isinstance(v, t)


def schema_ok(o):
    if not isinstance(o, dict) or not all(k in o for k in TOP_KEYS):
        return False
    if not isinstance(o["identity_check"], dict) or not isinstance(o["segments"], list) or not o["segments"]:
        return False
    for s in o["segments"]:
        if not isinstance(s, dict) or set(s) != set(SEG_KEYS):
            return False
        for k, t in SEG_TYPES.items():
            v = s[k]
            if v is None and k in NULLABLE:
                continue
            if not _tok(v, t):
                return False
    return True


def usable_segments(o):
    """Segments with int start/end (partial credit for schema-invalid output)."""
    out = []
    segs = o.get("segments") if isinstance(o, dict) else None
    for s in segs if isinstance(segs, list) else []:
        if isinstance(s, dict) and _tok(s.get("start_frame"), int) and _tok(s.get("end_frame"), int):
            out.append(s)
    return out


# -------------------------------------------------------------------- metrics
def lev(a, b):
    if not a and not b:
        return 0
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def frame_labels(segs, lo, hi, key="action"):
    lab = [None] * (hi - lo)
    for s in segs:
        for f in range(max(s["start_frame"], lo), min(s["end_frame"], hi)):
            if lab[f - lo] is None:
                lab[f - lo] = s.get(key)
    return lab


def fail_mask(segs, lo, hi):
    m = np.zeros(hi - lo, bool)
    for s in segs:
        if s.get("failure") is True:
            a, b = max(s["start_frame"], lo) - lo, min(s["end_frame"], hi) - lo
            if b > a:
                m[a:b] = True
    return m


def boundary_f1(pb, tb, tol=5):
    if not pb and not tb:
        return 1.0
    if not pb or not tb:
        return 0.0
    used, tp = set(), 0
    for p in sorted(pb):
        best = None
        for k, t in enumerate(tb):
            if k not in used and abs(p - t) <= tol and (best is None or abs(p - t) < abs(p - tb[best])):
                best = k
        if best is not None:
            used.add(best)
            tp += 1
    pr, rc = tp / len(pb), tp / len(tb)
    return 0.0 if tp == 0 else 2 * pr * rc / (pr + rc)


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def teacher_info(rec):
    t = json.loads(rec["messages"][1]["content"])
    segs = t["segments"]
    return {"obj": t, "segs": segs,
            "has_fail": any(s["failure"] for s in segs),
            "confident": not any(s["failure"] and s.get("review_required") for s in segs),
            "cats": sorted({c for s in segs if s["failure"] for c in s["categories"]})}


def score_episode(pred_rec, T):
    segs_t = T["segs"]
    lo = min(s["start_frame"] for s in segs_t)
    hi = max(s["end_frame"] for s in segs_t)
    r = {"id": pred_rec["id"], "error": pred_rec.get("error"), "n_tokens": pred_rec.get("n_tokens", 0),
         "seconds": pred_rec.get("seconds", 0), "hit_max": bool(pred_rec.get("hit_max")),
         "teacher_has_failure": T["has_fail"], "teacher_confident": T["confident"], "teacher_cats": T["cats"],
         "teacher_n_segments": len(segs_t)}
    o = parse_pred(pred_rec.get("text", ""))
    r["parsed"] = o is not None
    r["schema_valid"] = bool(o is not None and schema_ok(o))
    ps = usable_segments(o) if o else []
    r["n_segments"] = len(ps)
    r["pred_has_failure"] = any(s.get("failure") is True for s in ps)
    pc = set()
    for s in ps:
        if s.get("failure") is True and isinstance(s.get("categories"), list):
            pc |= {c for c in s["categories"] if isinstance(c, str)}
    r["pred_cats"] = sorted(pc)
    ta = [s["action"] for s in segs_t]
    pa = [s.get("action") for s in ps]
    r["seg_count_abs_err"] = abs(len(ps) - len(segs_t)) if o else len(segs_t)
    r["action_edit_dist_norm"] = (lev(pa, ta) / max(len(pa), len(ta), 1)) if o else 1.0
    lt, lp = frame_labels(segs_t, lo, hi), frame_labels(ps, lo, hi)
    r["frame_action_acc"] = float(np.mean([x == y for x, y in zip(lt, lp)])) if o else 0.0
    tb = sorted({s["start_frame"] for s in segs_t[1:]})
    pb = sorted({s["start_frame"] for s in ps[1:]})
    r["boundary_f1"] = boundary_f1(pb, tb) if o else 0.0
    mt, mp = fail_mask(segs_t, lo, hi), fail_mask(ps, lo, hi)
    u = (mt | mp).sum()
    r["failure_iou"] = float((mt & mp).sum() / u) if u else None  # None -> neither has failure
    return r


def mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def agg(rows, subset=None):
    rows = [r for r in rows if subset is None or subset(r)]
    n = len(rows)
    if n == 0:
        return {"n": 0}
    m = {"n": n, "parse_rate": mean([r["parsed"] for r in rows]),
         "schema_valid_rate": mean([r["schema_valid"] for r in rows])}
    tp = sum(r["pred_has_failure"] and r["teacher_has_failure"] for r in rows)
    fp = sum(r["pred_has_failure"] and not r["teacher_has_failure"] for r in rows)
    fn = sum(not r["pred_has_failure"] and r["teacher_has_failure"] for r in rows)
    tn = n - tp - fp - fn
    p, rc, f = prf(tp, fp, fn)
    m.update(fail_acc=(tp + tn) / n, fail_precision=p, fail_recall=rc, fail_f1=f)
    m.update(seg_count_mae=mean([r["seg_count_abs_err"] for r in rows]),
             action_edit_dist_norm=mean([r["action_edit_dist_norm"] for r in rows]),
             frame_action_acc=mean([r["frame_action_acc"] for r in rows]),
             boundary_f1_tol5=mean([r["boundary_f1"] for r in rows]),
             failure_iou=mean([r["failure_iou"] for r in rows]))
    cats = sorted({c for r in rows for c in r["teacher_cats"] + r["pred_cats"]})
    ctp = cfp = cfn = 0
    per = []
    for c in cats:
        a_ = sum(c in r["pred_cats"] and c in r["teacher_cats"] for r in rows)
        b_ = sum(c in r["pred_cats"] and c not in r["teacher_cats"] for r in rows)
        c_ = sum(c not in r["pred_cats"] and c in r["teacher_cats"] for r in rows)
        ctp += a_; cfp += b_; cfn += c_
        per.append(prf(a_, b_, c_)[2])
    m["category_micro_f1"] = prf(ctp, cfp, cfn)[2] if cats else None
    m["category_macro_f1"] = float(np.mean(per)) if per else None
    m.update(mean_tokens=mean([r["n_tokens"] for r in rows]), mean_seconds=mean([r["seconds"] for r in rows]),
             frac_hit_max=mean([r["hit_max"] for r in rows]), n_errors=sum(bool(r["error"]) for r in rows))
    return m


HEAD = ["parse_rate", "schema_valid_rate", "fail_f1", "frame_action_acc", "failure_iou"]


def boot(rows_by_tag, tags, nb=1000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(rows_by_tag[tags[0]])
    idx = rng.integers(0, n, (nb, n))
    vals = {t: {k: [] for k in HEAD} for t in tags}
    for ix in idx:
        for t in tags:
            m = agg([rows_by_tag[t][i] for i in ix])
            for k in HEAD:
                vals[t][k].append(m[k])
    arr = lambda v: np.array([np.nan if x is None else x for x in v], float)
    ci = lambda v: [float(x) for x in np.nanpercentile(arr(v), [2.5, 97.5])]
    res = {t: {k: ci(v) for k, v in vals[t].items()} for t in tags}
    diff = {}
    if "base" in tags and "sft" in tags:
        for k in HEAD:
            d = arr(vals["sft"][k]) - arr(vals["base"][k])
            diff[k] = {"mean": float(np.nanmean(d)), "ci95": [float(x) for x in np.nanpercentile(d, [2.5, 97.5])]}
    extra = {}
    for a_, b_ in (("sft_unmasked", "base"), ("sft_unmasked", "sft")):
        if a_ in tags and b_ in tags:
            extra[f"{a_}_minus_{b_}"] = {k: {"mean": float(np.nanmean(arr(vals[a_][k]) - arr(vals[b_][k]))),
                "ci95": [float(x) for x in np.nanpercentile(arr(vals[a_][k]) - arr(vals[b_][k]), [2.5, 97.5])]} for k in HEAD}
    return (res, diff, extra) if extra else (res, diff)


def load_preds(tag, ids):
    rows = {}
    for i in ids:
        p = os.path.join(PREDS, tag, i + ".json")
        rows[i] = json.load(open(p)) if os.path.exists(p) else {"id": i, "text": "", "n_tokens": 0, "seconds": 0,
                                                                  "error": "missing"}
    return rows


def cmd_score(a):
    global SCORES
    if getattr(a, "scores_dir", None):
        SCORES = a.scores_dir
    ids = data.load_split()["eval"]
    recs = {r["id"]: r for r in data.load_records()}
    T = {i: teacher_info(recs[i]) for i in ids}
    os.makedirs(SCORES, exist_ok=True)
    rows_by_tag, summary = {}, {}
    for tag in a.tags:
        preds = load_preds(tag, ids)
        use = [i for i in ids if not (a.skip_missing and preds[i].get("error") == "missing")]
        rows = [score_episode(preds[i], T[i]) for i in use]
        rows_by_tag[tag] = rows
        with open(os.path.join(SCORES, f"{tag}_per_episode.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        summary[tag] = {"all": agg(rows), "parseable": agg(rows, lambda r: r["parsed"]),
                        "confident_teacher": agg(rows, lambda r: r["teacher_confident"])}
    if len({len(v) for v in rows_by_tag.values()}) == 1 and rows_by_tag[a.tags[0]]:
        _b = boot(rows_by_tag, a.tags)
        res, diff = _b[0], _b[1]
        for _k, _v in (_b[2].items() if len(_b) > 2 else []):
            summary[_k] = _v
        for t in a.tags:
            summary[t]["bootstrap_ci95"] = res[t]
        if diff:
            summary["sft_minus_base"] = diff
    json.dump(summary, open(os.path.join(SCORES, "summary.json"), "w"), indent=1)
    fmt = lambda v: "-" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))
    md = []
    for sub in ("all", "parseable", "confident_teacher"):
        md.append(f"\n### Subset: {sub}\n")
        keys = [k for k in summary[a.tags[0]][sub] if k != "n"]
        md.append("| metric | " + " | ".join(f"{t} (n={summary[t][sub]['n']})" for t in a.tags) + " |")
        md.append("|---|" + "---|" * len(a.tags))
        for k in keys:
            cells = []
            for t in a.tags:
                s = fmt(summary[t][sub].get(k))
                if sub == "all" and k in HEAD and "bootstrap_ci95" in summary[t]:
                    lo, hi = summary[t]["bootstrap_ci95"][k]
                    s += f" [{lo:.3f}, {hi:.3f}]"
                cells.append(s)
            md.append(f"| {k} | " + " | ".join(cells) + " |")
    if "sft_minus_base" in summary:
        md.append("\n### Paired bootstrap sft - base (all)\n\n| metric | diff | 95% CI |\n|---|---|---|")
        for k, v in summary["sft_minus_base"].items():
            md.append(f"| {k} | {v['mean']:+.3f} | [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] |")
    for _k in [k for k in summary if k.endswith("_minus_base") and k != "sft_minus_base" or k == "sft_unmasked_minus_sft"]:
        md.append(f"\n### Paired bootstrap {_k.replace('_minus_', ' - ')} (all)\n\n| metric | diff | 95% CI |\n|---|---|---|")
        for k, v in summary[_k].items():
            md.append(f"| {k} | {v['mean']:+.3f} | [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] |")
    open(os.path.join(SCORES, "summary.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    g = sp.add_parser("generate")
    g.add_argument("--adapter", default="none")
    g.add_argument("--tag", required=True)
    g.add_argument("--limit", type=int, default=0)
    g.add_argument("--max-new-tokens", type=int, default=5120)
    s = sp.add_parser("score")
    s.add_argument("--tags", nargs="+", required=True)
    s.add_argument("--skip-missing", action="store_true")
    s.add_argument("--scores-dir", default="")
    args = ap.parse_args()
    cmd_generate(args) if args.cmd == "generate" else cmd_score(args)
