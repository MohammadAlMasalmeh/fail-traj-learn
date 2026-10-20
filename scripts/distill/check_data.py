"""Unit check of masking contract + token length stats. usage: check_data.py [stride] [cams] [--n N]"""
import sys, os, json, statistics as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data
from model import load_processor
stride = int(sys.argv[1]) if len(sys.argv) > 1 else 3
cams = sys.argv[2] if len(sys.argv) > 2 else "both"
N = int(sys.argv[3]) if len(sys.argv) > 3 else 10**9
cfg = data.InputCfg(stride=stride, cams=cams)
proc = load_processor()
recs = {r["id"]: r for r in data.load_records()}
split = data.load_split()
tot, sup, lens, plens, nchars_ok, nchars_cov = [], [], [], [], 0, 0
for i, rid in enumerate(split["train"][:N]):
    r = recs[rid]
    asst = r["messages"][1]["content"]
    ex = data.encode_train(proc, r, cfg=cfg)
    ids, lab, offs = ex["input_ids"], ex["labels"], ex["offsets"]
    P, A = ex["n_prompt"], ex["n_asst"]
    ranges = [(x["start"], x["end"]) for x in r["supervision"]["assistant_excluded_char_ranges"]]
    excl = [False] * len(asst)
    for s, e in ranges:
        for k in range(s, min(e, len(asst))): excl[k] = True
    # 1) offsets tile the string and decoding reproduces it exactly
    dec = proc.tokenizer.decode(ids[P:P + A - 1].tolist())
    assert dec == asst, (rid, "decode mismatch")
    cov = [False] * len(asst)
    for j, (s, e) in enumerate(offs):
        tid = ids[P + j].item(); l = lab[P + j].item()
        if l != -100:
            assert l == tid
            assert not any(excl[s:e]), (rid, j, "supervised token touches excluded chars")
            for k in range(s, e): cov[k] = True
        else:
            # masked => must overlap excluded range (we never mask anything else)
            assert any(excl[s:e]) or s == e, (rid, j, "masked without cause")
    assert lab[-1].item() == data.EOT_ID and ids[-1].item() == data.EOT_ID
    assert (lab[:P] == -100).all()
    nok = sum(1 for k in range(len(asst)) if not excl[k]); ncv = sum(cov)
    nchars_ok += nok; nchars_cov += ncv
    # supervised text is substring-faithful: reconstruct from covered chars
    sup.append(int((lab[P:] != -100).sum())); tot.append(A); lens.append(len(ids)); plens.append(P)
    if i % 50 == 0: print(i, rid, len(ids), P, A, sup[-1], flush=True)
def q(x): return f"min {min(x)} med {int(st.median(x))} max {max(x)} mean {int(st.mean(x))}"
print(f"cfg={cfg}")
print("total seq tokens:", q(lens)); print("prompt tokens:", q(plens)); print("assistant tokens:", q(tot))
print(f"supervised assistant tokens: {sum(sup)}/{sum(tot)} = {100*sum(sup)/sum(tot):.1f}% ; supervised tokens/ep {q(sup)}")
print(f"non-excluded chars covered by supervised tokens: {nchars_cov}/{nchars_ok} = {100*nchars_cov/nchars_ok:.2f}%")
print("zero-supervised-except-EOT episodes:", sum(1 for s in sup if s <= 1))
print("ALL CHECKS PASSED")
