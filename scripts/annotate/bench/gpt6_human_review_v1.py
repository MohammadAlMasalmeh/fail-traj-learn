#!/usr/bin/env python3
"""Prepare blinded GPT-6 conversation episodes and freeze human-reviewed labels.

The model sees only ``episodes/<id>/manifest.json``, ``observable.json``, the
rendered images named by the manifest, and the three prompt files for that ID.
Private selection facts remain outside the episode folders.

Commands:
  select         choose 12 disjoint, factually stratified episodes
  prepare        decode cameras, render pages/prompts, and freeze model inputs
  status         report GPT and human-review progress
  validate-gpt   validate one or all gpt6_annotation.json files
  freeze-gpt     freeze all twelve GPT candidates
  freeze-human   validate and freeze all twelve human_review.json files
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / "outputs" / "gpt6_human_review_v1"
DATA = Path("/home/aliu/projects/fail-traj-learn/data")
REFS = DATA.parent / "bench" / "references"
SEED = 20260927
FPS = 20
MODEL = "GPT-6 Astra"
IDS = [f"R{i:03d}" for i in range(1, 13)]

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

STRATA = {
    "successful_completion": 2,
    "closed_loss": 2,
    "release_failure": 2,
    "supported_failed_pickup": 2,
    "wrong_object_control": 2,
    "held_ending": 2,
}
MODE_HINTS = {
    "successful_completion": {"success"},
    "closed_loss": {"drop_transport"},
    "release_failure": {"release_miss"},
    "supported_failed_pickup": {"missed_grasp", "drop_transport"},
    "wrong_object_control": {"wrong_object"},
    "held_ending": {"hold_no_release", "stall_after_grasp"},
}
EXCLUSION_MANIFESTS = [
    "gpt6_pilot_v1/private_manifest.json",
    "oracle_audit_r6/manifest.json",
    "observable_sparse_v1/private_manifest.json",
    "oopsie_transfer_v1/private_manifest.json",
    "preprocessing_ablation_v2/private_manifest.json",
    "preprocessing_ablation_v3/holdout_private_manifest.json",
]
CLASSIFICATIONS = {"progress", "failure", "recovery", "neutral"}
FAILURE_TYPES = {"unknown", "missed_target", "lost_control", "wrong_action", "collision", "timeout", "other"}
EVENT_TYPES = {"grasp", "bilateral_loss", "release", "loss_closed", "loss_unknown", "settled"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def identities(value):
    if isinstance(value, dict):
        if "dataset" in value and "episode_index" in value:
            yield str(value["dataset"]), int(value["episode_index"])
        for child in value.values():
            yield from identities(child)
    elif isinstance(value, list):
        for child in value:
            yield from identities(child)


def factual(row: dict) -> dict:
    """Private physical facts used only to stratify episodes."""
    import pyarrow.parquet as pq
    from score_preprocessing_ablation_v2 import contact_events, separation_events, sustained_airborne

    folder = DATA / row["dataset"]
    sidepath = folder / "sidecar" / f"episode_{int(row['episode_index']):06d}.json"
    side = read(sidepath)
    paths = sorted((folder / "data").glob("chunk-*/*.parquet"))
    columns = ["frame_index", "action", "priv.obj_grasped", "priv.obj_gripper_contact",
               "priv.obj_support_contact", "priv.obj_obj_contact"]
    table = pq.read_table(paths, columns=columns,
                          filters=[("episode_index", "=", int(row["episode_index"]))])
    table = table.sort_by([("frame_index", "ascending")])
    values = table.to_pydict()
    target = side["goal_state"][0][1]
    slot = side["object_slots"].index(target)
    all_grasped = np.asarray(values["priv.obj_grasped"]) > 0
    held = all_grasped[:, slot]
    contact = np.asarray(values["priv.obj_gripper_contact"])[:, slot] > 0
    support = ((np.asarray(values["priv.obj_support_contact"])[:, slot] > 0) |
               (np.asarray(values["priv.obj_obj_contact"])[:, slot] > 0))
    command = np.asarray(values["action"])[:, 6]
    _, sustained_grasps = contact_events(held, command)
    events = separation_events(held, contact, command, sustained_grasps)
    airborne = sustained_airborne(held, support)
    other = all_grasped.copy()
    other[:, slot] = False
    event_types = {event["type"] for event in events}
    eligible = []
    if bool(row.get("success")) and "release" in event_types:
        eligible.append("successful_completion")
    if not bool(row.get("success")) and "loss_closed" in event_types and airborne.any():
        eligible.append("closed_loss")
    if not bool(row.get("success")) and "release" in event_types:
        eligible.append("release_failure")
    if (held & support).sum() >= 3 and not airborne.any() and not held[-1]:
        eligible.append("supported_failed_pickup")
    if not held.any() and other.sum() >= 5:
        eligible.append("wrong_object_control")
    if len(held) >= 5 and held[-5:].all() and airborne.any():
        eligible.append("held_ending")
    return {
        "target_name": target,
        "n_frames": int(len(held)),
        "task": side["task_language"],
        "recorded_success": bool(row.get("success")),
        "failure_mode": row.get("failure_mode"),
        "eligible": eligible,
        "primary_events": events,
        "target_held_frames": int(held.sum()),
        "other_object_grasp_frames": int(other.sum()),
        "source_sha256": {str(path): sha(path) for path in paths + [sidepath]},
    }


def select_episodes() -> list[dict]:
    path = OUT / "private_manifest.json"
    if path.exists():
        raise RuntimeError(f"selection already exists: {path}")
    excluded: set[tuple[str, int]] = set()
    exclusion_hashes = {}
    for relative in EXCLUSION_MANIFESTS:
        manifest = ROOT / "outputs" / relative
        if not manifest.exists():
            raise FileNotFoundError(f"required exclusion manifest is missing: {manifest}")
        excluded.update(identities(read(manifest)))
        exclusion_hashes[relative] = sha(manifest)

    rows = []
    for reference in sorted(REFS.glob("*/episode_*.json")):
        row = read(reference)
        identity = (str(row.get("dataset")), int(row.get("episode_index", -1)))
        mode = "success" if row.get("success") else row.get("failure_mode")
        if (row.get("suite") == "libero_object" and row.get("goal_slot") == "basket_1"
                and identity not in excluded
                and any(mode in modes for modes in MODE_HINTS.values())):
            rows.append(row)

    rng = random.Random(SEED)
    rng.shuffle(rows)
    chosen = []
    used = set()
    target_counts = Counter()
    fact_cache = {}
    candidate_counts = {}
    for stratum, count in STRATA.items():
        candidates = [row for row in rows
                      if ("success" if row.get("success") else row.get("failure_mode")) in MODE_HINTS[stratum]]
        candidate_counts[stratum] = len(candidates)
        for _ in range(count):
            candidates.sort(key=lambda row: (target_counts[str(row.get("target_slot"))],
                                             str(row["dataset"]), int(row["episode_index"])))
            selected = None
            for row in candidates:
                identity = (str(row["dataset"]), int(row["episode_index"]))
                if identity in used:
                    continue
                facts = fact_cache.setdefault(identity, factual(row))
                if stratum in facts["eligible"]:
                    selected = {
                        "dataset": identity[0], "episode_index": identity[1], "stratum": stratum,
                        "n_frames": facts["n_frames"], "task": facts["task"], "facts": facts,
                    }
                    break
            if selected is None:
                raise RuntimeError(f"insufficient unused factual candidates for {stratum}")
            chosen.append(selected)
            identity = (selected["dataset"], selected["episode_index"])
            used.add(identity)
            target_counts[selected["facts"]["target_name"]] += 1
            print("selected", stratum, identity, selected["facts"]["target_name"], flush=True)

    rng.shuffle(chosen)
    for ident, item in zip(IDS, chosen):
        item["id"] = ident
    assert len(chosen) == len(IDS) and not used.intersection(excluded)
    OUT.mkdir(parents=True, exist_ok=True)
    save(path, chosen)
    save(OUT / "selection_provenance.json", {
        "version": "gpt6_human_review_v1", "seed": SEED, "strata": STRATA,
        "excluded_identities": len(excluded), "exclusion_sha256": exclusion_hashes,
        "candidate_counts_before_factual_filter": candidate_counts,
        "target_counts": dict(target_counts), "private_manifest_sha256": sha(path),
    })
    return chosen


def font(size: int = 13):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def render_pages(agent: np.ndarray, wrist: np.ndarray, telemetry: list[dict], folder: Path, ident: str) -> list[str]:
    pages = []
    page_dir = folder / "pages"
    page_dir.mkdir(parents=True, exist_ok=True)
    size, cols, per_page, caption_h = 192, 3, 12, 46
    cell_w, cell_h = size * 2, size + caption_h
    for offset in range(0, len(agent), per_page):
        frames = list(range(offset, min(len(agent), offset + per_page)))
        name = f"{ident}_{frames[0]:04d}_{frames[-1]:04d}.jpg"
        page = Image.new("RGB", (cell_w * cols, cell_h * 4), "white")
        draw = ImageDraw.Draw(page)
        for j, frame in enumerate(frames):
            x, y = (j % cols) * cell_w, (j // cols) * cell_h
            draw.text((x + 3, y + 2), f"{ident} frame {frame} | external / wrist", font=font(), fill="black")
            page.paste(Image.fromarray(agent[frame]).resize((size, size), Image.Resampling.LANCZOS), (x, y + 22))
            page.paste(Image.fromarray(wrist[frame]).resize((size, size), Image.Resampling.LANCZOS), (x + size, y + 22))
            current = telemetry[frame]
            previous = telemetry[max(0, frame - 1)]
            dz_mm = (current["eef_xyz_m"][2] - previous["eef_xyz_m"][2]) * 1000
            caption = (f"{current['gripper_command']} q=({current['finger_qpos_m'][0]*1000:.1f},"
                       f"{current['finger_qpos_m'][1]*1000:.1f})mm "
                       f"z={current['eef_xyz_m'][2]*100:.1f}cm dz={dz_mm:+.1f}mm")
            draw.text((x + 3, y + size + 25), caption, font=font(11), fill="black")
        page.save(page_dir / name, quality=96)
        pages.append(f"pages/{name}")
    return pages


def render_initial(agent: np.ndarray, wrist: np.ndarray, path: Path, ident: str) -> None:
    image = Image.new("RGB", (512, 286), "white")
    draw = ImageDraw.Draw(image)
    draw.text((5, 4), f"{ident} initial scene, frame 0 | external / wrist", font=font(15), fill="black")
    image.paste(Image.fromarray(agent[0]), (0, 30))
    image.paste(Image.fromarray(wrist[0]), (256, 30))
    image.save(path, quality=96)


SCHEMA = """# GPT-6 candidate schema

The final file must be one JSON object with these fields:

```json
{
  "version": "gpt6_human_review_v1",
  "model_family": "GPT-6 Astra",
  "id": "R001",
  "task": "exact task from observable.json",
  "fps": 20,
  "episode_outcome": "success | failure | unknown",
  "target_description": "visible identity and uncertainty",
  "summary": "short observable episode summary",
  "segments": [
    {
      "start": 0,
      "end": 20,
      "text": "observable facts",
      "classification": "progress | failure | recovery | neutral",
      "failure_type": "unknown | missed_target | lost_control | wrong_action | collision | timeout | other",
      "confidence": 0.0,
      "note": "uncertainty or supporting detail",
      "evidence_frames": [0, 19]
    }
  ],
  "events": [
    {
      "frame": 20,
      "type": "grasp | bilateral_loss | release | loss_closed | loss_unknown | settled",
      "confidence": 0.0,
      "reason": "visible evidence and any telemetry consistency",
      "evidence_frames": [19, 20]
    }
  ],
  "trajectory_note": "remaining ambiguity"
}
```

Segments must be sorted, non-overlapping, and cover every frame exactly with half-open
ranges `[start,end)`. The first starts at 0 and the last ends at `n_frames`. Use
`failure_type: unknown` for every non-failure segment. A failure segment must use a
specific non-unknown failure type. Evidence frames and event frames must be integers
within the episode. Empty events are valid. Do not add dense per-frame state arrays.
"""


def prompt_one(ident: str) -> str:
    base = f"outputs/gpt6_human_review_v1/episodes/{ident}"
    return f"""# Turn 1 of 3: observable evidence pass for {ident}

This is a blinded robot-trajectory annotation run. Complete this evidence pass now;
do not ask questions, delegate, browse the web, or inspect unrelated repository files.

You may read only:
- `{base}/manifest.json`
- `{base}/observable.json`
- `{base}/initial.jpg`
- every image listed in that manifest under `{base}/pages/`
- this prompt file

Do not read private manifests, references, prior predictions, human annotations,
evaluation results, documentation, source datasets, or another episode. The episode
outcome is intentionally withheld. Opening/closing is a command, not proof of contact.

Inspect every listed page with the image-viewing tool at full available detail. Identify
the task target independently from the object the robot happens to manipulate. Build an
evidence ledger containing target identity, observable interaction phases, possible
grasp/contact/loss/release/settling transitions, useful task progress, harmful actions,
attempted recovery, and unresolved ambiguity. Check images against telemetry without
treating telemetry as physical contact truth.

Write a JSON object to `{base}/gpt6_observation.json` with keys `id`,
`inspected_pages`, `target_description`, `phase_ledger`, `transition_candidates`, and
`uncertainties`. Each phase and transition must cite frame numbers. `inspected_pages`
must exactly equal the manifest page list. Do not write final segments or
`gpt6_annotation.json` in this turn. After saving, respond only with a short completion
message asking the user to send turn 2.
"""


def prompt_two(ident: str) -> str:
    base = f"outputs/gpt6_human_review_v1/episodes/{ident}"
    return f"""# Turn 2 of 3: skeptical audit for {ident}

Continue the same blinded conversation. Re-read `{base}/gpt6_observation.json`,
`{base}/manifest.json`, and `{base}/observable.json`. Reinspect relevant supplied image pages. The
same read restrictions from turn 1 remain in force; do not inspect hidden references,
other episodes, prior studies, or human labels.

Audit the evidence ledger rather than endorsing it. Examine the trajectory forwards
and backwards around every gripper-command change and every proposed physical
transition. Specifically test for missed brief supported pinches, object squeeze-out,
residual one-sided contact after bilateral loss, closed-command loss, opening-associated
release, wrong-object interaction, regrasp, and local settling. Separate physical facts
from task-level judgments. Remove transitions supported only by command or finger
position, and retain `unknown` when occlusion prevents a decision.

Write `{base}/gpt6_audit.json` with keys `id`, `checked_command_runs`,
`supported_transitions`, `rejected_transitions`, `possible_omissions`,
`semantic_questions`, and `remaining_uncertainties`. Cite frames and visible evidence.
Do not write the final annotation yet. Respond only with a short completion message
asking the user to send turn 3.
"""


def prompt_three(ident: str) -> str:
    base = f"outputs/gpt6_human_review_v1/episodes/{ident}"
    return f"""# Turn 3 of 3: final candidate annotation for {ident}

Continue the same blinded conversation. Reconcile `{base}/gpt6_observation.json` and
`{base}/gpt6_audit.json` against the supplied images and telemetry. Follow
`outputs/gpt6_human_review_v1/prompts/OUTPUT_SCHEMA.md` exactly.

Produce a concise full-episode segmentation. Classify each interval by its task-level
effect: `progress` advances the instructed task; `failure` introduces or worsens an
observable error; `recovery` attempts to restore progress after an error; `neutral`
has no clear task effect. Do not call all approach motion progress when target identity
or intent is unclear. An attempted correction can remain failure if it repeats the
error. Infer episode success only when the visible task result supports it; otherwise
use failure or unknown as warranted. Do not invent hidden physical events.

Write the final object to `{base}/gpt6_annotation.json`. Validate its IDs, exact
half-open coverage from frame 0 through `n_frames`, enums, event bounds, and evidence
frames before finishing. Do not alter the observation or audit files. Respond with the
output path and a one-sentence statement of whether validation succeeded.
"""


def write_prompts() -> None:
    prompt_dir = OUT / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    (prompt_dir / "OUTPUT_SCHEMA.md").write_text(SCHEMA, encoding="utf-8")
    (prompt_dir / "README.md").write_text(
        "# GPT-6 conversation prompts\n\nStart a fresh GPT-6 Astra Codex conversation for each episode. "
        "Send its three prompt files in numeric order. Never reuse a conversation across episode IDs.\n",
        encoding="utf-8",
    )
    for ident in IDS:
        for number, content in ((1, prompt_one(ident)), (2, prompt_two(ident)), (3, prompt_three(ident))):
            (prompt_dir / f"{ident}_turn{number}.md").write_text(content, encoding="utf-8")


def expected_input_files() -> list[Path]:
    files = sorted(path for path in (OUT / "prompts").iterdir() if path.is_file())
    for ident in IDS:
        episode = OUT / "episodes" / ident
        files.extend([episode / "cameras.npz", episode / "observable.json", episode / "manifest.json", episode / "initial.jpg"])
        files.extend(sorted((episode / "pages").glob("*.jpg")))
    return files


def verify_input_freeze() -> None:
    frozen = read(OUT / "input_freeze.json")
    current = {str(path.relative_to(OUT)): sha(path) for path in expected_input_files()}
    if frozen["sha256"] != current:
        missing = sorted(set(frozen["sha256"]) - set(current))
        added = sorted(set(current) - set(frozen["sha256"]))
        changed = sorted(key for key in set(current) & set(frozen["sha256"])
                         if current[key] != frozen["sha256"][key])
        raise RuntimeError(f"input freeze mismatch: missing={missing}, added={added}, changed={changed}")


def prepare() -> None:
    private_path = OUT / "private_manifest.json"
    if not private_path.exists():
        select_episodes()
    items = read(private_path)
    if [item["id"] for item in items] != IDS:
        raise ValueError("private manifest IDs do not match the fixed twelve-episode protocol")
    if (OUT / "input_freeze.json").exists():
        verify_input_freeze()
        print("Existing frozen pack verified; no inputs rewritten", flush=True)
        return
    write_prompts()
    from common import Episode, open_dataset
    from review_oracle import decode_episode

    datasets = {}
    for item in items:
        folder = OUT / "episodes" / item["id"]
        folder.mkdir(parents=True, exist_ok=True)
        if (folder / "cameras.npz").exists():
            with np.load(folder / "cameras.npz") as arrays:
                agent, wrist = arrays["agent"], arrays["wrist"]
            observable = read(folder / "observable.json")
            telemetry = observable["telemetry"]
        else:
            dataset = datasets.get(item["dataset"])
            if dataset is None:
                dataset = datasets[item["dataset"]] = open_dataset(item["dataset"])
            episode = Episode(dataset, item["dataset"], item["episode_index"])
            if episode.n != item["n_frames"] or episode.task != item["task"]:
                raise ValueError(f"source identity mismatch for {item['id']}")
            agent, wrist, _ = decode_episode(dataset, episode)
            np.savez_compressed(folder / "cameras.npz", agent=agent, wrist=wrist)
            state, action = episode.col("observation.state"), episode.col("action")
            telemetry = [{
                "frame": frame,
                "eef_xyz_m": state[frame, :3].round(5).tolist(),
                "finger_qpos_m": state[frame, 6:8].round(5).tolist(),
                "gripper_command": "close" if action[frame, 6] > 0 else "open",
            } for frame in range(episode.n)]
            observable = {
                "id": item["id"], "task": item["task"], "n_frames": episode.n, "fps": episode.fps,
                "frame_bounds": "zero-based, end-exclusive",
                "notes": "Finger positions and gripper commands are not proof of contact or possession.",
                "telemetry": telemetry,
            }
            save(folder / "observable.json", observable)
        if agent.shape != wrist.shape or agent.shape != (item["n_frames"], 256, 256, 3):
            raise ValueError(f"unexpected camera shape for {item['id']}: {agent.shape}, {wrist.shape}")
        render_initial(agent, wrist, folder / "initial.jpg", item["id"])
        pages = render_pages(agent, wrist, telemetry, folder, item["id"])
        save(folder / "manifest.json", {
            "version": "gpt6_human_review_v1", "id": item["id"], "task": item["task"],
            "n_frames": item["n_frames"], "fps": FPS, "initial_image": "initial.jpg",
            "pages": pages, "supplied_frames": list(range(item["n_frames"])),
            "model_read_allowlist": ["manifest.json", "observable.json", "initial.jpg", *pages],
            "notes": "Every frame is supplied once in chronological pages; external view is left and wrist view right.",
        })
        print(item["id"], item["n_frames"], "frames", len(pages), "pages", flush=True)

    hashes = {str(path.relative_to(OUT)): sha(path) for path in expected_input_files()}
    freeze_path = OUT / "input_freeze.json"
    save(freeze_path, {
        "version": "gpt6_human_review_v1", "created_from_private_manifest_sha256": sha(private_path),
        "model": MODEL, "conversation_policy": "fresh three-turn Codex conversation per episode",
        "sha256": hashes,
    })
    total = sum(item["n_frames"] for item in items)
    (OUT / "README.md").write_text(
        "# GPT-6 candidate labels for human review\n\n"
        f"Twelve frozen observable-only episodes ({total} frames). Start a fresh GPT-6 Astra Codex conversation "
        "for each episode and send `prompts/<ID>_turn1.md`, turn 2, then turn 3. The model writes its immutable "
        "candidate beside the source as `gpt6_annotation.json`. After all candidates are frozen, reviewers use "
        "`human_annotator.py --seed-name gpt6_annotation.json --annotation-name human_review.json` so edits never "
        "overwrite the candidate. Private selection files at this directory's root must never be read by the model.\n",
        encoding="utf-8",
    )
    print("Prepared", len(items), "episodes and", total, "frames", flush=True)


def validate_gpt_annotation(value: dict, meta: dict) -> dict:
    if not isinstance(value, dict):
        raise ValueError("candidate must be a JSON object")
    if value.get("version") != "gpt6_human_review_v1":
        raise ValueError("wrong or missing version")
    if value.get("model_family") != MODEL:
        raise ValueError(f"model_family must be {MODEL!r}")
    if value.get("id") != meta["id"] or value.get("task") != meta["task"] or value.get("fps") != meta["fps"]:
        raise ValueError("candidate identity/task/fps does not match observable.json")
    if value.get("episode_outcome") not in {"success", "failure", "unknown"}:
        raise ValueError("invalid episode_outcome")
    if not str(value.get("target_description", "")).strip() or not str(value.get("summary", "")).strip():
        raise ValueError("target_description and summary are required")
    segments = value.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError("segments must be a non-empty list")
    expected = 0
    for index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            raise ValueError(f"segment {index} is not an object")
        start, end = segment.get("start"), segment.get("end")
        if isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, int) or not isinstance(end, int):
            raise ValueError(f"segment {index} bounds must be integers")
        if start != expected or end <= start or end > meta["n_frames"]:
            raise ValueError(f"segment {index} breaks exact contiguous coverage")
        classification = segment.get("classification")
        failure_type = segment.get("failure_type")
        if classification not in CLASSIFICATIONS or failure_type not in FAILURE_TYPES:
            raise ValueError(f"segment {index} has an invalid class or failure type")
        if classification == "failure" and failure_type == "unknown":
            raise ValueError(f"segment {index} failure requires a specific failure_type")
        if classification != "failure" and failure_type != "unknown":
            raise ValueError(f"segment {index} non-failure must use failure_type=unknown")
        if not str(segment.get("text", "")).strip():
            raise ValueError(f"segment {index} needs observable text")
        confidence = segment.get("confidence")
        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not 0 <= confidence <= 1:
            raise ValueError(f"segment {index} confidence must be in [0,1]")
        evidence = segment.get("evidence_frames")
        if not isinstance(evidence, list) or not evidence:
            raise ValueError(f"segment {index} requires evidence_frames")
        if any(isinstance(frame, bool) or not isinstance(frame, int) or not start <= frame < end for frame in evidence):
            raise ValueError(f"segment {index} evidence must fall inside its interval")
        expected = end
    if expected != meta["n_frames"]:
        raise ValueError("segments do not end at n_frames")
    events = value.get("events")
    if not isinstance(events, list):
        raise ValueError("events must be a list")
    for index, event in enumerate(events):
        frame = event.get("frame") if isinstance(event, dict) else None
        if (event.get("type") not in EVENT_TYPES or isinstance(frame, bool) or not isinstance(frame, int)
                or not 0 <= frame < meta["n_frames"]):
            raise ValueError(f"event {index} has an invalid type or frame")
        confidence = event.get("confidence")
        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not 0 <= confidence <= 1:
            raise ValueError(f"event {index} confidence must be in [0,1]")
        evidence = event.get("evidence_frames")
        if not isinstance(evidence, list) or not evidence or any(
                isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < meta["n_frames"]
                for value in evidence):
            raise ValueError(f"event {index} has invalid evidence_frames")
        if not str(event.get("reason", "")).strip():
            raise ValueError(f"event {index} needs a reason")
    return value


def selected_ids(requested: str | None) -> list[str]:
    if requested is None:
        return IDS
    if requested not in IDS:
        raise ValueError(f"unknown episode ID {requested!r}")
    return [requested]


def validate_conversation_artifacts(observation: dict, audit: dict, manifest: dict) -> None:
    ident = manifest["id"]
    if observation.get("id") != ident:
        raise ValueError("observation artifact has the wrong episode ID")
    if observation.get("inspected_pages") != manifest["pages"]:
        raise ValueError("observation inspected_pages must exactly match the manifest page list")
    for key in ("phase_ledger", "transition_candidates", "uncertainties"):
        if not isinstance(observation.get(key), list):
            raise ValueError(f"observation {key} must be a list")
    if not str(observation.get("target_description", "")).strip():
        raise ValueError("observation target_description is required")
    if audit.get("id") != ident:
        raise ValueError("audit artifact has the wrong episode ID")
    for key in ("checked_command_runs", "supported_transitions", "rejected_transitions",
                "possible_omissions", "semantic_questions", "remaining_uncertainties"):
        accepted = (list, dict) if key == "remaining_uncertainties" else (list,)
        if not isinstance(audit.get(key), accepted):
            kinds = "a list or object" if key == "remaining_uncertainties" else "a list"
            raise ValueError(f"audit {key} must be {kinds}")


def validate_gpt(requested: str | None = None) -> list[Path]:
    verify_input_freeze()
    paths = []
    for ident in selected_ids(requested):
        episode = OUT / "episodes" / ident
        observation_path = episode / "gpt6_observation.json"
        audit_path = episode / "gpt6_audit.json"
        candidate_path = episode / "gpt6_annotation.json"
        for path in (observation_path, audit_path, candidate_path):
            if not path.exists():
                raise FileNotFoundError(path)
        validate_conversation_artifacts(read(observation_path), read(audit_path),
                                        read(episode / "manifest.json"))
        validate_gpt_annotation(read(candidate_path), read(episode / "observable.json"))
        paths.extend((observation_path, audit_path, candidate_path))
        print(ident, "three-turn GPT artifacts valid", flush=True)
    return paths


def freeze_gpt() -> None:
    paths = validate_gpt()
    freeze = OUT / "gpt_freeze.json"
    value = {
        "version": "gpt6_human_review_v1", "input_freeze_sha256": sha(OUT / "input_freeze.json"),
        "conversation_artifact_sha256": {str(path.relative_to(OUT)): sha(path) for path in paths},
    }
    if freeze.exists() and read(freeze) != value:
        raise RuntimeError("GPT freeze exists and does not match current candidates")
    save(freeze, value)
    print("Frozen", len(paths) // 3, "GPT conversations", flush=True)


def validate_human_review(value: dict, meta: dict, candidate_path: Path) -> None:
    outcome = value.get("episode_outcome")
    if value.get("id") != meta["id"] or outcome not in {"success", "failure", "unknown"}:
        raise ValueError("human review has invalid identity or outcome")
    if outcome == "unknown" and not str(value.get("trajectory_note", "")).strip():
        raise ValueError("unknown human outcome requires a terminal uncertainty note")
    review_of = value.get("review_of", {})
    if review_of.get("sha256") != sha(candidate_path):
        raise ValueError("human review does not record the frozen GPT candidate hash")
    if value.get("candidate_decision") not in {"accepted", "corrected"}:
        raise ValueError("human review must explicitly accept or correct the GPT candidate")
    if value["candidate_decision"] == "corrected" and not str(value.get("candidate_review_note", "")).strip():
        raise ValueError("a corrected candidate requires a reviewer note")
    segments = value.get("segments")
    expected = 0
    if not isinstance(segments, list) or not segments:
        raise ValueError("human review needs semantic segments")
    for index, segment in enumerate(segments):
        start, end = segment.get("start"), segment.get("end")
        if start != expected or not isinstance(end, int) or end <= start or end > meta["n_frames"]:
            raise ValueError(f"human segment {index} breaks coverage")
        if segment.get("classification") not in CLASSIFICATIONS or not str(segment.get("text", "")).strip():
            raise ValueError(f"human segment {index} has an invalid class or empty text")
        if segment["classification"] == "failure" and segment.get("failure_type") in {None, "", "unknown"}:
            raise ValueError(f"human failure segment {index} requires a failure type")
        expected = end
    if expected != meta["n_frames"]:
        raise ValueError("human review does not cover the full episode")


def freeze_human() -> None:
    if not (OUT / "gpt_freeze.json").exists():
        raise RuntimeError("freeze GPT candidates before human review")
    freeze_gpt()
    paths = []
    for ident in IDS:
        episode = OUT / "episodes" / ident
        review_path = episode / "human_review.json"
        if not review_path.exists():
            raise FileNotFoundError(review_path)
        validate_human_review(read(review_path), read(episode / "observable.json"), episode / "gpt6_annotation.json")
        paths.append(review_path)
        print(ident, "human review valid", flush=True)
    freeze = OUT / "human_freeze.json"
    value = {
        "version": "gpt6_human_review_v1", "gpt_freeze_sha256": sha(OUT / "gpt_freeze.json"),
        "human_review_sha256": {str(path.relative_to(OUT)): sha(path) for path in paths},
    }
    if freeze.exists() and read(freeze) != value:
        raise RuntimeError("human freeze exists and does not match current reviews")
    save(freeze, value)
    print("Frozen", len(paths), "human reviews", flush=True)


def status() -> None:
    rows = []
    for ident in IDS:
        episode = OUT / "episodes" / ident
        rows.append({
            "id": ident,
            "observation": (episode / "gpt6_observation.json").exists(),
            "audit": (episode / "gpt6_audit.json").exists(),
            "candidate": (episode / "gpt6_annotation.json").exists(),
            "human_review": (episode / "human_review.json").exists(),
        })
    print(json.dumps({
        "input_frozen": (OUT / "input_freeze.json").exists(),
        "gpt_frozen": (OUT / "gpt_freeze.json").exists(),
        "human_frozen": (OUT / "human_freeze.json").exists(),
        "episodes": rows,
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["select", "prepare", "status", "validate-gpt", "freeze-gpt", "freeze-human"])
    parser.add_argument("--id", help="Single episode ID for validate-gpt")
    args = parser.parse_args()
    if args.command == "select":
        select_episodes()
    elif args.command == "prepare":
        prepare()
    elif args.command == "status":
        status()
    elif args.command == "validate-gpt":
        validate_gpt(args.id)
    elif args.command == "freeze-gpt":
        freeze_gpt()
    elif args.command == "freeze-human":
        freeze_human()


if __name__ == "__main__":
    main()
