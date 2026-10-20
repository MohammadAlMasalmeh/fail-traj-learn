"""CPU check of --unmask-reasons review_required_segment mode (tokenizer+images, no model)."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data
from model import load_processor
UN = ["review_required_segment"]
proc = load_processor(); recs = {r["id"]: r for r in data.load_records()}
cfg = data.DEFAULT_CFG
tot = so = sn = 0; viol = 0; eps_fail = eps_fail_sup = 0; fail_tok = fail_sup = 0; mism = 0
for i, rid in enumerate(data.load_split()["train"]):
    r = recs[rid]; asst = r["messages"][1]["content"]
    ex = data.encode_train(proc, r, cfg=cfg, unmask_reasons=UN)
    P, A = ex["n_prompt"], ex["n_asst"]; lab = ex["labels"][P:]; offs = ex["offsets"]
    rs = r["supervision"]["assistant_excluded_char_ranges"]
    ui = [(x["start"], x["end"]) for x in rs if x["reason"] == "uncertain_identity"]
    allr = [(x["start"], x["end"]) for x in rs]
    for j, (s, e) in enumerate(offs):
        sup_new = lab[j].item() != -100
        old = not any(s < b and e > a for a, b in allr)
        tot += 1; so += old; sn += sup_new
        if sup_new and any(s < b and e > a for a, b in ui): viol += 1
        exp = not any(s < b and e > a for a, b in ui)
        if exp != sup_new: mism += 1
    segs = json.loads(asst)["segments"]; fidx = {k for k, sg in enumerate(segs) if sg.get("failure")}
    if fidx:
        eps_fail += 1; any_s = False
        for x in rs:
            if x["reason"] == "review_required_segment" and x.get("segment_index") in fidx:
                for j, (s, e) in enumerate(offs):
                    if s < x["end"] and e > x["start"]:
                        fail_tok += 1; ok = lab[j].item() != -100; fail_sup += ok; any_s |= ok
        eps_fail_sup += any_s
    if i % 100 == 0: print(i, flush=True)
print(f"old supervised {so}/{tot}={100*so/tot:.1f}% ; new {sn}/{tot}={100*sn/tot:.1f}%")
print("uncertain_identity supervised-overlap tokens (must be 0):", viol, "; mismatches vs expected:", mism)
print(f"episodes with failure=true: {eps_fail}; with failure-segment text supervised: {eps_fail_sup}; failure-range tokens supervised {fail_sup}/{fail_tok}")
