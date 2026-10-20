import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data
from model import load_processor
proc = load_processor(); cfg = data.InputCfg()
recs = {r["id"]: r for r in data.load_records()}
out = {}
for rid in recs:
    msgs = data.build_student_messages(recs[rid], cfg=cfg)
    nimg = sum(e["type"] == "image" for e in msgs[0]["content"])
    txt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    P = len(proc.tokenizer(txt, add_special_tokens=False)["input_ids"]) - nimg + nimg * 66
    A = len(proc.tokenizer(recs[rid]["messages"][1]["content"], add_special_tokens=False)["input_ids"]) + 1
    out[rid] = [P, A, nimg]
json.dump(out, open("outputs/distill/lens.json", "w"))
import numpy as np
T = np.array([sum(v[:2]) for v in out.values()])
for th in (8000, 9000, 9500, 10000): print(th, (T > th).sum())
