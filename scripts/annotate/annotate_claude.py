"""Claude-subscription backend for the frozen atomic-action annotation protocol.

The OpenAI runner (annotate_trajectories.py) sends one HTTP request per
episode/model. This backend instead prepares a self-contained request folder per
episode that a Claude agent (Claude Code subagent on the user's subscription)
reads with its file tools, then writes a JSON answer that is validated here with
the unchanged ``validate_atomic`` contract.

Differences from the OpenAI runner (documented in docs/claude_backend_experiments.md):
- Images: each sampled frame's agent and wrist views are tiled into labelled
  contact sheets instead of ~114 separate images (Claude request image limits
  and per-image overhead). Pixels are not resized; frame labels sit in a header
  strip above each tile so no evidence pixels are covered.
- Transport: no HTTP, token preflight, or USD reservation; usage is subscription.
- Repair: ``validate`` may be called more than once; every attempt is logged so
  first-pass validity is still measurable.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import annotation_runtime as common
import annotate_trajectories as base

ROOT = common.ROOT
PER_SHEET = 8          # sampled frames per contact sheet
PER_ROW = 2            # frame pairs (agent|wrist) per sheet row
HEADER = 22            # label strip height in pixels


def sheet_layout(frames):
    return [frames[i:i + PER_SHEET] for i in range(0, len(frames), PER_SHEET)]


def draw_sheet(arrays, frames, path):
    from PIL import Image, ImageDraw, ImageFont
    h, w = arrays["agent"].shape[1:3]
    rows = (len(frames) + PER_ROW - 1) // PER_ROW
    sheet = Image.new("RGB", (PER_ROW * 2 * w + (PER_ROW - 1) * 8, rows * (h + HEADER)), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    for i, f in enumerate(frames):
        r, c = divmod(i, PER_ROW)
        x0, y0 = c * (2 * w + 8), r * (h + HEADER)
        for j, camera in enumerate(("agent", "wrist")):
            sheet.paste(Image.fromarray(arrays[camera][f]), (x0 + j * w, y0 + HEADER))
            draw.text((x0 + j * w + 4, y0 + 2), f"frame {f} | {camera}", fill="black", font=font)
    sheet.save(path, quality=92)


def gripper_digest(ep, empty=0.012, open_=0.07):
    """Summarize model-visible gripper telemetry into runs of command/aperture state."""
    def state(r):
        ap = abs(r["finger_qpos_m"][0]) + abs(r["finger_qpos_m"][1])
        if ap >= open_:
            return "fingers open"
        if ap < empty:
            return "fingers fully closed (nothing between them)"
        return "fingers stopped partly open"
    runs = []
    for r in ep["telemetry"]:
        key = (r["gripper_command"], state(r))
        if runs and runs[-1][0] == key:
            runs[-1][2] = r["frame"]
        else:
            runs.append([key, r["frame"], r["frame"]])
    return [f"frames {a}-{b}: command {cmd}; {st}" for (cmd, st), a, b in runs]


def request_text(ep, prompt_name, sheets, digest=False):
    lines = [
        f"# Annotation request: episode {ep['id']}",
        "",
        f"Follow the protocol in `{prompt_name}` exactly (read the whole file first).",
        "Then view every contact sheet listed below with the Read tool. Each tile is",
        "labelled 'frame N | agent' or 'frame N | wrist'; the agent (third-person) view",
        "is on the left of each pair and the wrist view on its right. These are the",
        "only images supplied; cite only these frame indices and camera names.",
        "",
        "## Episode metadata",
        "```json",
        json.dumps({"episode_id": ep["id"], **{k: ep[k] for k in ("task", "n_frames", "fps", "sampled_frames")}}),
        "```",
        "",
        "## Contact sheets",
    ]
    for path, frames in sheets:
        lines.append(f"- `{path}`: frames {frames}")
    lines += ["", "## Telemetry (one row per sampled frame; commands at t follow observation t)", "```"]
    lines += [json.dumps(row) for row in ep["telemetry"]]
    lines += ["```", ""]
    if digest:
        lines += ["## Gripper telemetry digest (derived from the rows above; a cue, not proof)",
                  "Aperture = sum of the two finger positions. A run where the gripper is commanded",
                  "closed but the fingers stop partly open is consistent with an object between them;",
                  "fully closed fingers under a close command are consistent with an empty gripper.",
                  "Confirm every physical claim in the images; telemetry alone never establishes contact.", ""]
        lines += ["- " + line for line in gripper_digest(ep)] + [""]
    return "\n".join(lines)


def event_frames(telemetry, n, offsets=range(-4, 11, 2), tail=12):
    """Extra frames around gripper command changes and over the recording's end.

    Release, slip, and placement outcomes often resolve within a few frames of a
    command change or in the last frames, which a fixed stride samples once or twice.
    """
    cmd = {r["frame"]: r["gripper_command"] for r in telemetry}
    changes = [f for f in sorted(cmd) if f - 1 in cmd and cmd[f] != cmd[f - 1]]
    extra = {f + k for f in changes for k in offsets} | set(range(max(0, n - tail), n, 2))
    return {f for f in extra if 0 <= f < n}


def prepare(out, source, ids, prompt, stride, addendum=None, digest=False, events=False):
    import numpy as np
    out, source = Path(out).resolve(), Path(source).resolve()
    if out.exists():
        raise ValueError("Fresh output directory required; never overwrite a run")
    out.mkdir(parents=True)
    text = Path(prompt).read_text(encoding="utf-8")
    if addendum:
        text += Path(addendum).read_text(encoding="utf-8")
    (out / "prompt.txt").write_text(text, encoding="utf-8")
    episodes = []
    for eid in ids:
        observed = common.load(source / eid / "observable.json")
        n, fps = observed["n_frames"], observed["fps"]
        frames = set(range(0, n, stride)) | {n - 1}
        if events:
            frames |= event_frames(observed["telemetry"], n)
        frames = sorted(frames)
        rows = {r["frame"]: r for r in observed["telemetry"]}
        ep = dict(id=eid, task=observed["task"], n_frames=n, fps=fps, sampled_frames=frames,
                  telemetry=[dict(frame=f, seconds=f / fps, **{k: rows[f][k] for k in (
                      "eef_xyz_m", "finger_qpos_m", "gripper_command")}) for f in frames],
                  source_sha256={name: common.digest(source / eid / name)
                                 for name in ("cameras.npz", "observable.json")})
        folder = out / eid
        folder.mkdir()
        sheets = []
        with np.load(source / eid / "cameras.npz", allow_pickle=False) as arrays:
            arrays = {k: arrays[k] for k in ("agent", "wrist")}
            for k, chunk in enumerate(sheet_layout(frames)):
                path = folder / f"sheet_{k:02d}.jpg"
                draw_sheet(arrays, chunk, path)
                sheets.append((path.as_posix(), chunk))
        (folder / "request.md").write_text(
            request_text(ep, (out / "prompt.txt").as_posix(), sheets, digest), encoding="utf-8")
        common.write_json(folder / "input.json", ep)
        episodes.append(ep)
    common.write_json(out / "manifest.json", dict(
        name="claude_atomic", prepared_utc=datetime.now(timezone.utc).isoformat(),
        settings=dict(sample_stride=stride, base_prompt=Path(prompt).name,
                      addendum=Path(addendum).name if addendum else None, gripper_digest=digest,
                      event_frames=events, per_sheet=PER_SHEET, per_row=PER_ROW,
                      prompt_sha256=common.digest(out / "prompt.txt")),
        episodes=[e["id"] for e in episodes]))
    print(json.dumps({e["id"]: len(e["sampled_frames"]) for e in episodes}))


def validate(out, eid, model):
    out = Path(out).resolve()
    ep = common.load(out / eid / "input.json")
    answer = out / "answers" / f"{eid}_{model}.json"
    log_path = out / "answers" / f"{eid}_{model}_validation.json"
    log = common.load(log_path) if log_path.exists() else []
    try:
        value = json.loads(answer.read_text(encoding="utf-8"))
        base.validate_atomic(value, ep, label_mode="failure_binary")
        status, error = "validated", ""
    except Exception as exc:
        status, error = "invalid", str(exc)[:500]
    log.append(dict(attempt=len(log) + 1, status=status, error=error,
                    utc=datetime.now(timezone.utc).isoformat()))
    common.write_json(log_path, log)
    print(status + (": " + error if error else ""))
    return status == "validated"


def report(out):
    out = Path(out).resolve()
    rows = []
    for log_path in sorted((out / "answers").glob("*_validation.json")):
        eid, model = log_path.stem.removesuffix("_validation").split("_", 1)
        log = common.load(log_path)
        rows.append(dict(episode=eid, model=model, attempts=len(log),
                         first_pass=log[0]["status"] == "validated", final=log[-1]["status"],
                         first_error=log[0]["error"]))
    common.write_json(out / "summary.json", rows)
    print(json.dumps(rows, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("prepare", "validate", "report"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--source", type=Path, default=common.SOURCE)
    p.add_argument("--episodes", nargs="+", default=["R001", "R002", "R003", "R004"])
    p.add_argument("--prompt", type=Path, default=ROOT / "scripts/annotate/prompts/atomic_actions/annotation.txt")
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--addendum", type=Path)
    p.add_argument("--gripper-digest", action="store_true")
    p.add_argument("--event-frames", action="store_true")
    p.add_argument("--episode")
    p.add_argument("--model")
    a = p.parse_args()
    if a.command == "prepare":
        prepare(a.out, a.source, a.episodes, a.prompt, a.stride, a.addendum, a.gripper_digest, a.event_frames)
    elif a.command == "validate":
        raise SystemExit(0 if validate(a.out, a.episode, a.model) else 1)
    else:
        report(a.out)
