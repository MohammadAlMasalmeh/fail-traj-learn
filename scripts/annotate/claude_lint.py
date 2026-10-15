#!/usr/bin/env python3
"""Lint Claude episode annotations against gripper telemetry (stdlib only).

Usage: python claude_lint.py --runs RUN [RUN ...] [--out path]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

EMPTY_CLOSED = 0.012  # summed finger aperture (m): fingers closed on nothing
OPEN = 0.07           # summed finger aperture (m): fully open
CHECKS = ["L1", "L2", "L3", "L4", "L5", "L6"]
L1_PHRASES = ("establishes control", "remains under gripper control", "under control")
L2_PHRASES = ("does not acquire", "not acquired", "misses", "fails to grasp", "closes on nothing", "empty")
L3_PHRASES = ("separates from the gripper", "released", "releases")
L5_PHRASES = ("slips", "drops", "dropped", "loses control", "lost control", "does not acquire",
              "is not the requested", "collision", "knocks")
NEGATION = re.compile(r"(?<![a-z])(?:no|not|without) ")
NEG_WINDOW = 25
L6_MARGIN = 5
DETAIL_WIDTH = 120


def aperture(row: dict) -> float:
    left, right = row["finger_qpos_m"]
    return abs(left) + abs(right)


def seg_text(seg: dict) -> str:
    return f"{seg.get('what_happened') or ''} {seg.get('observed_result') or ''}".lower()


def seg_samples(seg: dict, telemetry: list[dict]) -> list[dict]:
    start, end = seg["start_frame"], seg["end_frame"]
    rows = [r for r in telemetry if start <= r["frame"] < end]
    if rows:
        return rows
    later = [r for r in telemetry if r["frame"] >= start]
    if not later:
        return []
    nearest = min(r["frame"] for r in later)
    return [r for r in later if r["frame"] == nearest]


def first_phrase(text: str, phrases) -> str | None:
    return next((p for p in phrases if p in text), None)


def unnegated_phrase(text: str, phrases) -> str | None:
    """Return a phrase whose occurrence is not preceded within NEG_WINDOW chars by a negator."""
    for phrase in phrases:
        idx = text.find(phrase)
        while idx != -1:
            window = text[max(0, idx - NEG_WINDOW):idx]
            if not NEGATION.search(window):
                return phrase
            idx = text.find(phrase, idx + 1)
    return None


def lint_segment(seg: dict, idx: int, telemetry: list[dict]) -> list[tuple[str, str]]:
    """Return (check, detail) pairs for one segment."""
    hits: list[tuple[str, str]] = []
    text = seg_text(seg)
    samples = seg_samples(seg, telemetry)
    last = samples[-1] if samples else None
    last_ap = aperture(last) if last else None

    phrase = first_phrase(text, L1_PHRASES)
    if phrase and last_ap is not None and last_ap < EMPTY_CLOSED:
        hits.append(("L1", f"claims control ('{phrase}') but last aperture {last_ap:.4f} m < {EMPTY_CLOSED}"))

    phrase = first_phrase(text, L2_PHRASES)
    if (phrase and last is not None and last.get("gripper_command") == "close"
            and EMPTY_CLOSED <= last_ap < OPEN):
        hits.append(("L2", f"miss claim ('{phrase}') but fingers closed on object, aperture {last_ap:.4f} m"))

    phrase = first_phrase(text.replace("attempts to release", ""), L3_PHRASES)
    if (phrase and samples and all(r.get("gripper_command") == "close" and aperture(r) >= EMPTY_CLOSED
                                   for r in samples)):
        hits.append(("L3", f"separation/release ('{phrase}') while gripper commanded closed and not empty"))

    if seg.get("failure") is False:
        phrase = unnegated_phrase(text, L5_PHRASES)
        if phrase:
            hits.append(("L5", f"failure words ('{phrase}') on non-failure segment"))
    return hits


def lint_annotation(ann: dict, telemetry: list[dict], run: str, tag: str, ep: str) -> dict:
    segments = ann.get("segments") or []
    findings: list[dict] = []
    counts = {c: 0 for c in CHECKS}

    def add(check: str, idx: int | None, frames: str, detail: str) -> None:
        counts[check] += 1
        findings.append({"episode": ep, "tag": tag, "run": run, "segment_index": idx,
                         "frames": frames, "check": check, "detail": detail})

    match_status = (ann.get("identity_check") or {}).get("match_status")
    any_semantic = any("sequencing_semantic" in (s.get("categories") or []) for s in segments)
    if match_status == "supported_match" and any_semantic:
        add("L4", None, "-", "match_status=supported_match but a segment has sequencing_semantic")
    elif match_status == "supported_mismatch" and not any_semantic:
        add("L4", None, "-", "match_status=supported_mismatch but no segment has sequencing_semantic")

    for idx, seg in enumerate(segments):
        frames = f"{seg['start_frame']}-{seg['end_frame']}"
        for check, detail in lint_segment(seg, idx, telemetry):
            add(check, idx, frames, detail)
        evidence_frames = [e["frame"] for e in (seg.get("evidence") or []) if isinstance(e, dict) and "frame" in e]
        lo, hi = seg["start_frame"] - L6_MARGIN, seg["end_frame"] + L6_MARGIN
        if not any(lo <= f <= hi for f in evidence_frames):
            add("L6", idx, frames, f"no evidence frame in [{lo},{hi}] (evidence frames: {evidence_frames})")

    failures = sum(1 for s in segments if s.get("failure"))
    reviews = sum(1 for s in segments if s.get("review_required"))
    return {"run": run, "tag": tag, "episode": ep, "n_segments": len(segments),
            "failure_segments": failures,
            "review_required_fraction": round(reviews / len(segments), 3) if segments else 0.0,
            "counts": counts, "findings": findings}


def load_json(path: Path):
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def lint_run(run_dir: Path) -> list[dict]:
    run = run_dir.name
    records = []
    telemetry_cache: dict[str, list[dict]] = {}
    for ann_path in sorted((run_dir / "answers").glob("*.json")):
        if ann_path.name.endswith("_validation.json"):
            continue
        try:
            ann = load_json(ann_path)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"warning: skipping {ann_path}: {exc}", file=sys.stderr)
            continue
        ep = ann.get("episode_id") or ann_path.stem.rsplit("_", 1)[0]
        tag = ann_path.stem[len(ep) + 1:] if ann_path.stem.startswith(ep + "_") else ann_path.stem
        if ep not in telemetry_cache:
            inp = run_dir / ep / "input.json"
            try:
                telemetry_cache[ep] = sorted(load_json(inp)["telemetry"], key=lambda r: r["frame"])
            except (OSError, json.JSONDecodeError, KeyError) as exc:
                print(f"warning: no telemetry for {ep} ({exc}); sample-based checks skipped", file=sys.stderr)
                telemetry_cache[ep] = []
        records.append(lint_annotation(ann, telemetry_cache[ep], run, tag, ep))
    return records


def print_report(records: list[dict]) -> None:
    head = f"{'run/tag':<34}{'episode':<9}{'seg':>4}{'fail':>5} " + " ".join(f"{c:>3}" for c in CHECKS) + f" {'review':>7}"
    print(head)
    for r in records:
        cells = " ".join(f"{r['counts'][c]:>3}" for c in CHECKS)
        print(f"{r['run'] + '/' + r['tag']:<34}{r['episode']:<9}{r['n_segments']:>4}{r['failure_segments']:>5} "
              f"{cells} {r['review_required_fraction']:>7.3f}")
    print()
    for r in records:
        for f in r["findings"]:
            seg = "-" if f["segment_index"] is None else f"s{f['segment_index']}"
            detail = f["detail"][:DETAIL_WIDTH]
            print(f"{r['run']}/{r['tag']} {f['episode']} {seg} {f['frames']} {f['check']} {detail}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", required=True, help="run directories")
    parser.add_argument("--out", help="output JSON (default: <first run>/lint.json)")
    args = parser.parse_args()

    run_dirs = [Path(r).resolve() for r in args.runs]
    records: list[dict] = []
    for run_dir in run_dirs:
        if not (run_dir / "answers").is_dir():
            print(f"warning: {run_dir} has no answers/ directory", file=sys.stderr)
            continue
        records.extend(lint_run(run_dir))

    out_path = Path(args.out).resolve() if args.out else run_dirs[0] / "lint.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump({"thresholds": {"EMPTY_CLOSED": EMPTY_CLOSED, "OPEN": OPEN},
                   "annotations": records}, fh, indent=2)

    print_report(records)
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
