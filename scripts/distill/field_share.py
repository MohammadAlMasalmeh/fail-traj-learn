"""Share of generated tokens in free-text fields of existing v3 predictions."""
import sys, os, json, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data, eval as ev
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(data.BASE_ID)
FREE = ["evidence", "uncertainty", "recovery", "boundary_reason", "observed_result", "what_happened"]
cnt = collections.Counter(); total = 0; n = 0
d = os.path.join(data.OUT_DIR, "preds", "sft_v3")
for f in sorted(os.listdir(d)):
    r = json.load(open(os.path.join(d, f))); o = ev.parse_pred(r["text"])
    if not o: continue
    n += 1; total += len(tok(r["text"], add_special_tokens=False)["input_ids"])
    L = lambda k, v: len(tok(json.dumps(k) + ":" + json.dumps(v, separators=(",", ":")) + ",", add_special_tokens=False)["input_ids"])
    ic = o.get("identity_check", {})
    for k, v in ic.items():
        cnt["identity_check." + k] += L(k, v)
    for s in o.get("segments", []):
        for k, v in s.items():
            cnt[k] += L(k, v)
free = sum(cnt[k] for k in FREE) + cnt["identity_check.evidence"] + cnt["identity_check.uncertainty"]
print("episodes", n, "total tokens", total, "mean/ep", total / n)
for k, v in cnt.most_common(14): print(f"{k:28s}{v:8d} {v/total:6.1%}")
print("segment free-text (6 fields)", sum(cnt[k] for k in FREE), f"{sum(cnt[k] for k in FREE)/total:.1%}")
print("incl identity_check evidence/uncertainty", free, f"{free/total:.1%}")
