"""Score annotations against visually verified dev-set events (reference_v1 notes).

Each check is (episode, description, expected, predicate). A predicate looks for a
failure segment with a given category overlapping a frame window. Expected=True
means the event is real and should be labeled; False means it is a known false
positive pattern that should NOT be labeled.
"""
import json, sys
from pathlib import Path

def hit(ann, cat, a, b):
    return any(s["failure"] and (cat is None or cat in s["categories"])
               and s["start_frame"] < b and s["end_frame"] > a for s in ann["segments"])

CHECKS = [
    ("R001", "wrong object labeled (60-135)", True, lambda x: hit(x, "sequencing_semantic", 60, 135)),
    ("R001", "delayed separation (170-260)", True, lambda x: hit(x, "manipulation", 170, 260)),
    ("R001", "false slip at 135-145", False, lambda x: hit(x, "grasp", 135, 145)),
    ("R002", "false failure on clean success", False, lambda x: hit(x, None, 0, 150)),
    ("R004", "delayed separation (155-170)", True, lambda x: hit(x, "manipulation", 150, 170)),
    ("R004", "false slip at 135-145", False, lambda x: hit(x, "grasp", 135, 145)),
    ("R005", "drop outside basket (155-185)", True, lambda x: hit(x, None, 150, 185)),
    ("R005", "regrasp miss ~210", True, lambda x: hit(x, "grasp", 205, 225)),
    ("R005", "regrasp miss ~270", True, lambda x: hit(x, "grasp", 262, 280)),
    ("R005", "false wrong-object", False, lambda x: hit(x, "sequencing_semantic", 0, 280)),
    ("R005", "false attachment after drop (190-260)", False, lambda x: hit(x, "manipulation", 190, 260)),
    # Held-out events verified by frame strips (R007 lid shrinks as gripper rises; R008 upright bottle toppled).
    ("R007", "first grasp misses (60-85)", True, lambda x: hit(x, "grasp", 60, 85)),
    ("R007", "no false release/drop failure (75-90)", False, lambda x: any(
        s["failure"] and s["action"] == "release" and s["start_frame"] < 90 and s["end_frame"] > 75
        for s in x["segments"])),
    ("R008", "toppling collision (100-140)", True, lambda x: hit(x, "collision", 95, 140)),
]

def main(runs):
    rows = []
    for run in runs:
        for tag in ("haiku", "sonnet", "opus", "ens"):
            res, n = [], 0
            for ep, desc, expected, pred in CHECKS:
                f = Path(run) / "answers" / f"{ep}_{tag}.json"
                if not f.exists():
                    res.append("·"); continue
                ok = pred(json.loads(f.read_text(encoding="utf-8"))) == expected
                res.append("✓" if ok else "✗"); n += ok
            scored = sum(r != "·" for r in res)
            if scored:
                rows.append((f"{Path(run).name}/{tag}", "".join(res), f"{n}/{scored}"))
    print("checks:", *[f"{i+1}={c[0]} {c[1]}" for i, c in enumerate(CHECKS)], sep="\n  ")
    for r in rows:
        print(f"{r[0]:<28} {r[1]}  {r[2]}")

if __name__ == "__main__":
    main(sys.argv[1:])
