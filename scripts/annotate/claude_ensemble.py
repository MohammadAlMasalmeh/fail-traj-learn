"""Frame-level majority vote over several annotations of one episode.

Usage: python claude_ensemble.py --out RUN_DIR --tag TAG EP=path1,path2,path3 [EP=...]

Writes RUN_DIR/answers/<EP>_<TAG>.json for scoring with claude_events.py /
claude_metrics.py. A frame is failure when a strict majority of annotations mark
it failure; its categories are those used by a strict majority of the annotations
that mark it failure. Segments are runs of identical (failure, categories);
action/object fields come from the first annotation at the segment start. The
output is a scoring artifact, not a schema-validated annotation.
"""
import argparse
import json
from collections import Counter
from pathlib import Path


def frame_labels(ann):
    n = ann["segments"][-1]["end_frame"]
    out = [None] * n
    for s in ann["segments"]:
        for f in range(s["start_frame"], s["end_frame"]):
            out[f] = s
    return out


def vote(anns):
    per = [frame_labels(a) for a in anns]
    n = min(len(p) for p in per)
    k = len(anns)
    labels = []
    for f in range(n):
        segs = [p[f] for p in per]
        fail = [s for s in segs if s["failure"]]
        if 2 * len(fail) > k:
            cats = Counter(c for s in fail for c in set(s["categories"]))
            cs = frozenset(c for c, m in cats.items() if 2 * m > len(fail)) or frozenset(cats.most_common(1)[0][:1])
            labels.append((True, cs, segs[0]))
        else:
            labels.append((False, frozenset(), segs[0]))
    segments, start = [], 0
    for f in range(1, n + 1):
        if f == n or labels[f][:2] != labels[start][:2]:
            fail, cats, src = labels[start]
            segments.append(dict(src, start_frame=start, end_frame=f, failure=fail, categories=sorted(cats)))
            start = f
    base = anns[0]
    statuses = Counter(a["identity_check"]["match_status"] for a in anns).most_common(1)[0][0]
    return dict(episode_id=base["episode_id"], identity_check=dict(base["identity_check"], match_status=statuses),
                segments=segments)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("items", nargs="+")
    a = p.parse_args()
    (a.out / "answers").mkdir(parents=True, exist_ok=True)
    for item in a.items:
        ep, paths = item.split("=", 1)
        anns = [json.loads(Path(x).read_text(encoding="utf-8")) for x in paths.split(",")]
        (a.out / "answers" / f"{ep}_{a.tag}.json").write_text(json.dumps(vote(anns), indent=1), encoding="utf-8")
        print(ep, len(anns), "annotations")
