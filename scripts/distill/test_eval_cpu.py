import argparse, json, os, random, shutil, sys, statistics
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data, eval as ev
ids = data.load_split()["eval"]; recs = {r["id"]: r for r in data.load_records()}
rnd = random.Random(0)
garb = lambda t: rnd.choice([t[: len(t) // 2], "no json here", "```json\n" + t[10:] + "\n```", "{}"])
for tag, f in (("_teacher", lambda t: t), ("_garbage", garb)):
    d = os.path.join(ev.PREDS, tag); os.makedirs(d, exist_ok=True)
    for i in ids:
        json.dump({"id": i, "text": f(recs[i]["messages"][1]["content"]), "n_tokens": 100, "seconds": 1.0,
                   "hit_max": False, "error": None}, open(os.path.join(d, i + ".json"), "w"))
t = recs[ids[0]]["messages"][1]["content"]
assert ev.parse_pred("```json\n" + t + "\n```") == json.loads(t)
ev.cmd_score(argparse.Namespace(tags=["_teacher", "_garbage"], skip_missing=False))
S = json.load(open(os.path.join(ev.SCORES, "summary.json")))
T = S["_teacher"]["all"]
for k in ("parse_rate", "schema_valid_rate", "fail_f1", "fail_acc", "frame_action_acc", "boundary_f1_tol5",
          "failure_iou", "category_micro_f1", "category_macro_f1"):
    assert abs(T[k] - 1.0) < 1e-9, (k, T[k])
assert T["action_edit_dist_norm"] == 0 and T["seg_count_mae"] == 0
print("garbage:", S["_garbage"]["all"])
print("TEACHER-VS-TEACHER PERFECT: OK")
for tag in ("_teacher", "_garbage"):
    shutil.rmtree(os.path.join(ev.PREDS, tag)); os.remove(os.path.join(ev.SCORES, tag + "_per_episode.jsonl"))
for f in ("summary.json", "summary.md"):
    os.remove(os.path.join(ev.SCORES, f))
from model import load_processor
proc = load_processor(); L = []
for i in ids:
    L.append(data.encode_prompt(proc, data.build_student_messages(recs[i]))["input_ids"].shape[1])
print("prompt tokens min/median/max", min(L), statistics.median(L), max(L))
