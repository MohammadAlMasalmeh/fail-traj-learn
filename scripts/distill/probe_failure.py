"""Teacher-forced probe of P("failure": true) at every segment's failure field.
  probe_failure.py own     --tag T --adapter PATH   # eval episodes, continuation = model's greedy text
  probe_failure.py teacher --tag T --adapter PATH   # 40 train episodes (seed 0, half w/ failure), continuation = teacher JSON
  probe_failure.py score   --tag T                  # thresholds from teacher_train, metrics on eval (own)
  probe_failure.py fake    --tag T --probe-dir D    # CPU test: no model, random p_true (writes D/*.jsonl)
Add --probe-dir to redirect outputs (default outputs/distill/probe)."""
import argparse, bisect, gc, json, os, random, re, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data
from eval import parse_pred, usable_segments, load_preds
from score_failures import near, prf, frame_set, cats, episode_metrics, boot_f1, VERIFIED, TOL

PROBE = os.path.join(data.OUT_DIR, "probe")
FAIL_RE = re.compile(r'(?<!\\)"failure"\s*:\s*(true|false)')
# first differing token after `"failure":` -- compact style and spaced style: (true, false); verified at runtime
CAND = {"compact": (3397, 4530), "spaced": (1847, 2416)}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def check_cands(tok):
    for style, sep in (("compact", ""), ("spaced", " ")):
        t = tok('"failure":' + sep + "true", add_special_tokens=False)["input_ids"]
        f = tok('"failure":' + sep + "false", add_special_tokens=False)["input_ids"]
        assert t[:-1] == f[:-1] and (t[-1], f[-1]) == CAND[style], (style, t, f)


def find_occurrences(text, tok):
    """-> (assistant token ids, [(token_idx_of_value, style, value_str, token_is_candidate)]); predicting position = idx-1."""
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    starts = [s for s, e in offs]
    occ = []
    for m in FAIL_RE.finditer(text):
        pos = m.start(1)
        k = bisect.bisect_right(starts, pos) - 1  # token covering pos
        style = "spaced" if text[pos - 1] == " " else "compact"
        occ.append((k, style, m.group(1), ids[k] in CAND[style]))
    return ids, occ


def map_segments(text, occ):
    """(list of (segment, occ idx) for usable segs, mismatch flag, greedy-flagged usable segs)."""
    o = parse_pred(text)
    if o is None:
        return None, True, []
    allsegs = [s for s in (o.get("segments") if isinstance(o.get("segments"), list) else []) if isinstance(s, dict)]
    use = {id(s) for s in usable_segments(o)}
    gflag = [s for s in allsegs if id(s) in use and s.get("failure") is True]
    if len(allsegs) != len(occ):
        return None, True, gflag
    return [(s, j) for j, s in enumerate(allsegs) if id(s) in use], False, gflag


def seg_dict(s, **kw):
    return dict(start_frame=s["start_frame"], end_frame=s["end_frame"],
                categories=s.get("categories") if isinstance(s.get("categories"), list) else [], **kw)


def make_row(eid, text, p_list, occ, extra=None):
    mp = float(max(p_list)) if p_list else None
    pairs, mism, gflag = map_segments(text, occ) if text else (None, True, [])
    segs = None
    if pairs is not None:
        segs = [seg_dict(s, greedy_failure=occ[j][2] == "true", p_true=p_list[j] if p_list else None) for s, j in pairs]
    r = dict(id=eid, n_occ=len(occ), n_cand_ok=sum(o[3] for o in occ), mismatch=bool(mism), max_p=mp, segments=segs,
             greedy_segments=[seg_dict(s) for s in gflag], greedy_any=any(o[2] == "true" for o in occ))
    r.update(extra or {})
    return r


def teacher_ids(recs, split, n=40, seed=0):
    rnd = random.Random(seed)
    tr = set(split["train"])
    pos = sorted(r["id"] for r in recs if r["id"] in tr and data.teacher_has_failure(r))
    neg = sorted(r["id"] for r in recs if r["id"] in tr and not data.teacher_has_failure(r))
    rnd.shuffle(pos); rnd.shuffle(neg)
    return sorted(pos[: n // 2] + neg[: n - n // 2])


def jobs(mode, tag, recs):
    """-> list of (id, continuation text, extra)"""
    split = data.load_split()
    if mode == "own":
        P = load_preds(tag, split["eval"])
        return [(i, P[i].get("text", ""), {"gen_error": P[i].get("error")}) for i in split["eval"]]
    byid = {r["id"]: r for r in recs}
    out = []
    for i in teacher_ids(recs, split):
        t = byid[i]["messages"][1]["content"]
        segs = json.loads(t)["segments"]
        out.append((i, t, {"teacher_has_failure": any(s["failure"] for s in segs),
                           "teacher_labels": [bool(s["failure"]) for s in segs]}))
    return out


def out_path(pdir, tag, mode):
    return os.path.join(pdir, f"{tag}_own.jsonl" if mode == "own" else f"{tag}_teacher_train.jsonl")


def done_ids(path):
    s = set()
    if os.path.exists(path):
        for l in open(path):
            try:
                r = json.loads(l)
                if not r.get("oom"):
                    s.add(r["id"])
            except Exception:
                pass
    return s


# ------------------------------------------------------------------ model run
def cmd_run(a):
    import torch
    from model import load_model, load_processor
    from lossfn import _core
    os.makedirs(a.probe_dir, exist_ok=True)
    recs = data.load_records()
    rbyid = {r["id"]: r for r in recs}
    path = out_path(a.probe_dir, a.tag, a.mode)
    done = done_ids(path)
    todo = [j for j in jobs(a.mode, a.tag, recs) if j[0] not in done]
    log(f"[probe] mode={a.mode} tag={a.tag} todo={len(todo)} done={len(done)}")
    if not todo:
        return
    proc = load_processor()
    tok = proc.tokenizer
    check_cands(tok)
    adapter = None if a.adapter in (None, "none", "") else a.adapter
    model = load_model(adapter_path=adapter).eval()
    inner, lm_head, softcap = _core(model)
    f = open(path, "a")
    for n, (eid, text, extra) in enumerate(todo):
        t0 = time.time()
        try:
            ids_a, occ = find_occurrences(text, tok) if text else ([], [])
            p_list, ntok = [], 0
            if occ:
                msgs = data.build_student_messages(rbyid[eid], cfg=data.DEFAULT_CFG)
                enc = data.encode_prompt(proc, msgs, cfg=data.DEFAULT_CFG)
                P = enc["input_ids"].shape[1]
                ids = torch.cat([enc["input_ids"][0], torch.tensor(ids_a, dtype=enc["input_ids"].dtype)])[None].cuda()
                mm = torch.cat([enc["mm_token_type_ids"][0], torch.zeros(len(ids_a), dtype=enc["mm_token_type_ids"].dtype)])[None].cuda()
                with torch.inference_mode():
                    out = inner(input_ids=ids, attention_mask=torch.ones_like(ids), mm_token_type_ids=mm,
                                pixel_values=enc["pixel_values"].cuda().to(torch.bfloat16),
                                image_position_ids=enc["image_position_ids"].cuda(), use_cache=False)
                    pos = torch.tensor([P + o[0] - 1 for o in occ], device="cuda")
                    h = out.last_hidden_state[0][pos]
                    for o, hh in zip(occ, h):
                        cid = torch.tensor(CAND[o[1]], device="cuda")
                        lg = torch.nn.functional.linear(hh[None], lm_head.weight[cid]).float()[0]
                        if softcap:
                            lg = torch.tanh(lg / softcap) * softcap
                        p_list.append(float(torch.softmax(lg, 0)[0]))
                ntok = int(ids.shape[1])
                del out, ids, mm, h
            row = make_row(eid, text, p_list, occ, dict(extra, n_tokens=ntok, mode=a.mode))
        except torch.cuda.OutOfMemoryError as e:
            row = dict(id=eid, oom=True, error="OOM " + str(e)[:150])
        except Exception as e:
            row = dict(id=eid, error=f"{type(e).__name__}: {str(e)[:200]}", max_p=None, segments=None, mismatch=True, n_occ=0)
        gc.collect(); torch.cuda.empty_cache()
        row["seconds"] = round(time.time() - t0, 2)
        f.write(json.dumps(row) + "\n"); f.flush()
        log(f"[probe] {n+1}/{len(todo)} {eid} occ={row.get('n_occ')} maxp={row.get('max_p')} mismatch={row.get('mismatch')} err={row.get('error')} {row['seconds']}s")


# ------------------------------------------------------------------- CPU test
def cmd_fake(a):
    """No model: tokenizer-only; random p_true correlated with labels. Writes into --probe-dir."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(data.BASE_ID)
    check_cands(tok)
    os.makedirs(a.probe_dir, exist_ok=True)
    rng = random.Random(1)
    recs = data.load_records()
    for mode in ("own", "teacher"):
        rows = []
        for eid, text, extra in jobs(mode, a.tag, recs):
            if mode == "own" and not text:
                continue
            ids_a, occ = find_occurrences(text, tok)
            p = [min(1, max(0, (0.4 if o[2] == "true" else 0.1) + rng.random() * 0.5)) for o in occ]
            r = make_row(eid, text, p, occ, dict(extra, n_tokens=len(ids_a), mode=mode))
            rows.append(r)
            log(mode, eid, "occ", len(occ), "cand_ok", r["n_cand_ok"], "mismatch", r["mismatch"], "tok", len(ids_a))
        with open(out_path(a.probe_dir, a.tag, mode), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")


# --------------------------------------------------------------------- score
def auroc(scores, labels):
    s = np.asarray(scores, float); y = np.asarray(labels, bool)
    npos, nneg = int(y.sum()), int((~y).sum())
    if npos == 0 or nneg == 0:
        return None
    order = s.argsort(kind="mergesort")
    ranks = np.empty(len(s)); ss = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def best_thr(scores, labels):
    """threshold maximizing F1 of (score > thr); candidates = midpoints between sorted unique scores (+extremes)."""
    s = np.asarray(scores, float); y = np.asarray(labels, bool)
    u = np.unique(s)
    cands = np.concatenate([[u[0] - 1e-6], (u[:-1] + u[1:]) / 2, [u[-1] + 1e-6]])
    best = (-1, 0.5, None)
    for t in cands:
        p = s > t
        P, R, F = prf(int((p & y).sum()), int((p & ~y).sum()), int((~p & y).sum()))
        if F > best[0] + 1e-12 or (abs(F - best[0]) <= 1e-12 and abs(t - 0.5) < abs(best[1] - 0.5)):
            best = (F, float(t), dict(p=P, r=R, f1=F))
    return best[1], best[2]


def load_jsonl(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def seg_metrics(pred_by_ep, V, ids):
    tp_ref = n_ref = tp_pred = n_pred = cat_ok = 0
    ious = []
    for i in ids:
        refs = V[i]["ref_segments"]; pred = pred_by_ep[i]
        hit = [any(near(q, p) for p in pred) for q in refs]
        n_ref += len(refs); tp_ref += sum(hit)
        n_pred += len(pred); tp_pred += sum(any(near(p, q) for q in refs) for p in pred)
        for q, h in zip(refs, hit):
            if h and any(cats(q) & cats(p) for p in pred if near(q, p)):
                cat_ok += 1
        if V[i]["segments_known"] and (refs or pred):
            A, B = frame_set(refs), frame_set(pred)
            ious.append(len(A & B) / len(A | B) if A | B else 1.0)
    p, r, f = prf(tp_pred, n_pred - tp_pred, n_ref - tp_ref)
    return dict(n_pred=n_pred, n_ref=n_ref, recall=tp_ref / n_ref if n_ref else 0.0, precision=tp_pred / n_pred if n_pred else 0.0,
                f1=f, category_match_on_hits=cat_ok / tp_ref if tp_ref else 0.0,
                frame_iou_mean=float(np.mean(ious)) if ious else 0.0)


def cmd_score(a):
    V = json.load(open(VERIFIED))["episodes"]
    tr = [r for r in load_jsonl(out_path(a.probe_dir, a.tag, "teacher")) if r.get("max_p") is not None]
    ow = {r["id"]: r for r in load_jsonl(out_path(a.probe_dir, a.tag, "own")) if not r.get("oom")}
    S = {"tag": a.tag}
    seg_p, seg_y = [], []
    for r in tr:
        if r.get("segments") is not None:
            seg_p += [s["p_true"] for s in r["segments"]]
            seg_y += [bool(s["greedy_failure"]) for s in r["segments"]]  # teacher string: stated value == teacher label
    tau_seg, m_seg = best_thr(seg_p, seg_y) if seg_p else (0.5, None)
    ep_p = [r["max_p"] for r in tr]; ep_y = [r["teacher_has_failure"] for r in tr]
    tau_ep, m_ep = best_thr(ep_p, ep_y) if tr else (0.5, None)
    sp, sy = np.asarray(seg_p), np.asarray(seg_y, bool)
    f05 = prf(int(((sp > .5) & sy).sum()), int(((sp > .5) & ~sy).sum()), int((~(sp > .5) & sy).sum()))[2] if seg_p else None
    S["teacher_train"] = dict(n_episodes=len(tr), n_segments=len(seg_p), n_fail_segments=int(sy.sum()), tau_seg=tau_seg,
                              seg_f1_at_tau=m_seg, seg_f1_at_0p5=f05, seg_auroc=auroc(seg_p, seg_y) if seg_p else None,
                              tau_ep=tau_ep, ep_f1_at_tau=m_ep, ep_auroc=auroc(ep_p, ep_y) if tr else None)
    ids = sorted(i for i in V if i in ow)
    sc = [ow[i].get("max_p") or 0.0 for i in ids]
    E = S["eval"] = dict(n=len(ids), n_missing=len(V) - len(ids), n_mismatch=sum(bool(ow[i].get("mismatch")) for i in ids),
                         n_no_score=sum(ow[i].get("max_p") is None for i in ids))
    greedy = [bool(ow[i].get("greedy_any")) for i in ids]
    for ref in ("verified", "simulator"):
        y = [V[i][ref + "_has_failure"] for i in ids]
        E["auroc_vs_" + ref] = auroc(sc, y)
        for name, thr in (("tau_ep", tau_ep), ("0.5", 0.5)):
            pred = [s > thr for s in sc]
            m = episode_metrics(pred, y); m["f1_ci95"] = boot_f1(pred, y)
            E[f"episode_vs_{ref}_at_{name}"] = m
        m = episode_metrics(greedy, y); m["f1_ci95"] = boot_f1(greedy, y)
        E[f"episode_vs_{ref}_greedy"] = m
    E["max_p_mean_by_verified"] = {str(k): float(np.mean([s for s, i in zip(sc, ids) if V[i]["verified_has_failure"] == k] or [np.nan]))
                                   for k in (True, False)}

    def preds(thr):  # mapping failed -> fall back to the model's own greedy flags
        return {i: ([s for s in ow[i]["segments"] if s["p_true"] is not None and s["p_true"] > thr]
                    if ow[i].get("segments") is not None else ow[i].get("greedy_segments", [])) for i in ids}
    E["segment_greedy"] = seg_metrics({i: (ow[i].get("greedy_segments") or []) for i in ids}, V, ids)
    E["segment_at_tau_seg"] = seg_metrics(preds(tau_seg), V, ids)
    E["segment_at_0p5"] = seg_metrics(preds(0.5), V, ids)
    ov, nov = [], []
    for i in ids:
        for s in ow[i].get("segments") or []:
            if s["p_true"] is not None:
                (ov if any(near(s, q) for q in V[i]["ref_segments"]) else nov).append(s["p_true"])
    st = lambda x: dict(n=len(x), mean=float(np.mean(x)) if x else None, median=float(np.median(x)) if x else None,
                        q25=float(np.percentile(x, 25)) if x else None, q75=float(np.percentile(x, 75)) if x else None)
    E["p_true_overlapping_verified_refs"] = st(ov)
    E["p_true_not_overlapping"] = st(nov)
    E["segment_auroc_overlap_vs_not"] = auroc(ov + nov, [True] * len(ov) + [False] * len(nov)) if ov and nov else None
    json.dump(S, open(os.path.join(a.probe_dir, "summary.json"), "w"), indent=1)

    f3 = lambda v: "-" if v is None else f"{v:.3f}"
    T = S["teacher_train"]
    L = [f"# Failure-field probe ({a.tag})", "",
         f"Teacher-forced P(true) at the `\"failure\":` value position (softmax over true/false tokens). Calibration: {T['n_episodes']} train episodes, {T['n_segments']} segments ({T['n_fail_segments']} failing).", "",
         f"- tau_seg (max seg F1 on teacher_train) = **{T['tau_seg']:.3f}**, F1 there {f3((T['seg_f1_at_tau'] or {}).get('f1'))} (F1 at 0.5: {f3(T['seg_f1_at_0p5'])}), seg AUROC {f3(T['seg_auroc'])}",
         f"- tau_ep (max episode F1 on teacher_train) = **{T['tau_ep']:.3f}**, F1 there {f3((T['ep_f1_at_tau'] or {}).get('f1'))}, ep AUROC {f3(T['ep_auroc'])}", "",
         f"## Eval (own greedy continuation), n={E['n']} (missing {E['n_missing']}, seg-mapping mismatches {E['n_mismatch']}, no score {E['n_no_score']})", "",
         f"- Episode AUROC (max p_true): vs verified **{f3(E['auroc_vs_verified'])}**, vs simulator **{f3(E['auroc_vs_simulator'])}**",
         f"- mean max_p: verified-failure {f3(E['max_p_mean_by_verified']['True'])}, verified-clean {f3(E['max_p_mean_by_verified']['False'])}", "",
         "| episode-level | P | R | F1 [95% CI] | TP/FP/FN/TN |", "|---|---|---|---|---|"]
    for ref in ("verified", "simulator"):
        for name in ("at_tau_ep", "at_0.5", "greedy"):
            m = E[f"episode_vs_{ref}_{name}"]; ci = m["f1_ci95"]
            L.append(f"| vs {ref}, {name} | {m['precision']:.2f} | {m['recall']:.2f} | {m['f1']:.2f} [{ci[0]:.2f}, {ci[1]:.2f}] | {m['tp']}/{m['fp']}/{m['fn']}/{m['tn']} |")
    L += ["", f"## Segment level vs verified ref segments (tol {TOL} fr)", "",
          "| flagging | #pred | #ref | recall | precision | F1 | cat match on hits | frame IoU |", "|---|---|---|---|---|---|---|---|"]
    for k, n in (("segment_greedy", "greedy (model)"), ("segment_at_0p5", "p_true>0.5"), ("segment_at_tau_seg", f"p_true>{tau_seg:.3f}")):
        m = E[k]
        L.append(f"| {n} | {m['n_pred']} | {m['n_ref']} | {m['recall']:.2f} | {m['precision']:.2f} | {m['f1']:.2f} | {m['category_match_on_hits']:.2f} | {m['frame_iou_mean']:.2f} |")
    o, n_ = E["p_true_overlapping_verified_refs"], E["p_true_not_overlapping"]
    L += ["", "## p_true by segment", "", "| group | n | mean | median | q25 | q75 |", "|---|---|---|---|---|---|",
          f"| overlapping verified failure refs | {o['n']} | {f3(o['mean'])} | {f3(o['median'])} | {f3(o['q25'])} | {f3(o['q75'])} |",
          f"| not overlapping | {n_['n']} | {f3(n_['mean'])} | {f3(n_['median'])} | {f3(n_['q25'])} | {f3(n_['q75'])} |", "",
          f"Segment AUROC (overlap vs not): {f3(E['segment_auroc_overlap_vs_not'])}", ""]
    open(os.path.join(a.probe_dir, "summary.md"), "w").write("\n".join(L))
    print("\n".join(L), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["own", "teacher", "score", "fake"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--adapter", default="none")
    ap.add_argument("--probe-dir", default=PROBE)
    a = ap.parse_args()
    {"own": cmd_run, "teacher": cmd_run, "score": cmd_score, "fake": cmd_fake}[a.mode](a)
