"""Data utilities for SFT distillation of the sol_v8 teacher labels into Gemma 4.

Public API (reused by eval):
  load_records(), build_student_messages(record, export_dir, cfg) -> messages
  encode_prompt(processor, messages, export_dir, cfg) -> BatchFeature (for generation)
  encode_train(processor, record, export_dir, cfg) -> dict(input_ids, labels, mm_token_type_ids, pixel_values, ...)
  make_split(), load_split()
"""
import json, os, random, re, sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompt_short import SHORT_INSTRUCTIONS

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EXPORT_DIR = os.path.join(REPO, "outputs", "hf_sol_v8_distillation_500", "sol_v8_distillation_500_export_v1")
OUT_DIR = os.path.join(REPO, "outputs", "distill")
BASE_ID = os.environ.get("DISTILL_BASE", "google/gemma-4-E4B-it")  # 12B (google/gemma-4-12B-it) does not fit; see SETUP_NOTES.md
EOT_ID = 106  # "<turn|>"
CAM_TXT = {"both": "agent, wrist", "agent": "agent", "wrist": "wrist"}


@dataclass
class InputCfg:
    stride: int = 3             # show images for every `stride`-th sampled frame (first & last always kept)
    cams: str = "both"          # both | agent | wrist
    max_soft_tokens: int = 70   # per-image visual token budget (70 -> 64 soft tokens for 256x256)
    all_telemetry: bool = True  # include compact telemetry line for frames whose images are dropped
    events: bool = False        # append a "Derived gripper events" block computed from telemetry + n_frames only


DEFAULT_CFG = InputCfg()


def load_records(export_dir=EXPORT_DIR):
    with open(os.path.join(export_dir, "train.jsonl"), encoding="utf8") as f:
        return [json.loads(l) for l in f]


def _frames(record):
    """Parse the user content into [{frame, tele(dict), images{cam: relpath}}]."""
    content = record["messages"][0]["content"]
    frames, cur = [], None
    for e in content[1:]:
        if e["type"] == "text":
            tele = json.loads(e["text"].split("; agent camera")[0])
            cur = {"frame": tele["frame"], "tele": tele, "images": {}}
            frames.append(cur)
        else:
            cam = "agent" if "_agent" in e["image"] else "wrist"
            cur["images"][cam] = e["image"]
    return frames


def _task_info(record):
    last = record["messages"][0]["content"][0]["text"].strip().split("\n")[-1]
    j = json.loads(last)
    return {k: j[k] for k in ("episode_id", "task", "n_frames", "fps")}


def _tele_line(t):
    """Compact telemetry: frame|seconds|eef xyz (mm)|finger qpos (mm)|gripper command."""
    x, y, z = (round(v * 1000) for v in t["eef_xyz_m"])
    a, b = (round(v * 1000) for v in t["finger_qpos_m"])
    return f"{t['frame']}|{t['seconds']:.2f}|{x},{y},{z}|{a},{b}|{t['gripper_command']}"


EMPTY_MM = 6.0      # fingers <= this while closed => closed on nothing
RISE_MM = 30.0      # eef z rise during a close run => lifted
TIME_LIMIT = 280


def _opening_mm(t):
    a, b = t["finger_qpos_m"]
    return (abs(a) + abs(b)) / 2 * 1000


def close_runs(frames):
    """Maximal runs of consecutive sampled frames with gripper_command == close -> list of dicts (telemetry only)."""
    T = [f["tele"] for f in frames]
    runs, i, n = [], 0, len(T)
    while i < n:
        if T[i]["gripper_command"] != "close":
            i += 1
            continue
        j = i
        while j + 1 < n and T[j + 1]["gripper_command"] == "close":
            j += 1
        reopen = T[j + 1] if j + 1 < n and T[j + 1]["gripper_command"] == "open" else None
        span = T[i:j + 1] + ([reopen] if reopen else [])  # finger state lags the command by a frame
        mn = min(_opening_mm(t) for t in span)
        rise = (max(t["eef_xyz_m"][2] for t in T[i:j + 1]) - T[i]["eef_xyz_m"][2]) * 1000
        runs.append({"start": T[i]["frame"], "end": T[j]["frame"], "min_mm": mn, "end_mm": _opening_mm(span[-1]),
                     "rise_mm": rise, "reopen": reopen["frame"] if reopen else None, "empty": mn <= EMPTY_MM})
        i = j + 1
    return runs


def events_block(frames, n_frames):
    """Compact derived gripper events from telemetry only (no labels). frames = _frames(record)."""
    lines = ["Derived gripper events from the telemetry (min = smallest finger opening during the close; "
             "empty = closed on nothing, <= 6 mm; hold = fingers > 6 mm; rise = end-effector rose > 30 mm; reopen = frame the gripper re-opened):"]
    for r in close_runs(frames):
        lines.append(f"close {r['start']}-{r['end']} min{r['min_mm']:.0f}mm {'empty' if r['empty'] else 'hold'}"
                     + (" rise" if r["rise_mm"] > RISE_MM else "")
                     + (f" reopen{r['reopen']}" if r["reopen"] is not None else " no-reopen"))
    if len(lines) == 1:
        lines.append("no close commands")
    lines.append(f"len {n_frames} " + ("(time limit)" if n_frames >= TIME_LIMIT else "(early end)"))
    return "\n".join(lines) + "\n"


def kept_indices(n, stride):
    keep = set(range(0, n, stride))
    keep.add(n - 1)
    return keep


def build_student_messages(record, export_dir=EXPORT_DIR, cfg: InputCfg = DEFAULT_CFG):
    """Compressed student input. Image entries are {'type':'image','image': <relative path>};
    encode_prompt()/encode_train() load them. The assistant target is untouched (see record)."""
    info = _task_info(record)
    frames = _frames(record)
    keep = kept_indices(len(frames), cfg.stride)
    head = (SHORT_INSTRUCTIONS + "\nEpisode: " + json.dumps(info) +
            "\nTelemetry, one line per sampled frame in time order: frame|seconds|end-effector x,y,z (mm)|finger joint positions (mm)|gripper command. "
            "Lines marked +img are followed by the agent-camera image then the wrist-camera image (a subset of frames). "
            "Frame numbers are original source-frame indices.\n")
    content = [{"type": "text", "text": head}]
    for i, fr in enumerate(frames):
        shown = i in keep
        if not shown and not cfg.all_telemetry:
            continue
        line = _tele_line(fr["tele"]) + (" +img" if shown else "") + "\n"
        content.append({"type": "text", "text": line})
        if shown:
            for cam in (["agent", "wrist"] if cfg.cams == "both" else [cfg.cams]):
                if cam in fr["images"]:
                    content.append({"type": "image", "image": fr["images"][cam]})
    if cfg.events:
        content.append({"type": "text", "text": events_block(frames, info["n_frames"])})
    return [{"role": "user", "content": content}]


def _load_images(messages, export_dir):
    from PIL import Image
    imgs = []
    for m in messages:
        for e in m["content"]:
            if e["type"] == "image":
                p = e["image"]
                imgs.append(Image.open(os.path.join(export_dir, p)).convert("RGB") if isinstance(p, str) else p)
    return imgs


def _prompt_text(processor, messages):
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def encode_prompt(processor, messages, export_dir=EXPORT_DIR, cfg: InputCfg = DEFAULT_CFG):
    processor.image_processor.max_soft_tokens = cfg.max_soft_tokens
    text = _prompt_text(processor, messages)
    return processor(text=text, images=_load_images(messages, export_dir), return_tensors="pt")


def encode_train(processor, record, export_dir=EXPORT_DIR, cfg: InputCfg = DEFAULT_CFG, with_pixels=True, unmask_reasons=(), fail_token_weight=1.0, max_len=0):
    """Dict of tensors with labels. The assistant string is tokenized separately with char offsets (exact contract
    coordinates); the prompt ends with the generation prefix (special token), so no tokenization merge at the seam."""
    import torch
    import dataclasses
    msgs = build_student_messages(record, export_dir, cfg)
    asst = record["messages"][1]["content"]
    assert isinstance(asst, str)
    processor.image_processor.max_soft_tokens = cfg.max_soft_tokens
    text = _prompt_text(processor, msgs)
    imgs = _load_images(msgs, export_dir)
    enc = processor(text=text, images=imgs, return_tensors="pt")
    a = processor.tokenizer(asst, add_special_tokens=False, return_offsets_mapping=True)
    img_stride = cfg.stride
    if max_len and cfg.events:  # training only: shed images (keep telemetry + events) until it fits
        k = 1
        while enc["input_ids"].shape[1] + len(a["input_ids"]) + 1 > max_len and k < 6:
            k += 1
            img_stride = cfg.stride * k
            msgs = build_student_messages(record, export_dir, dataclasses.replace(cfg, stride=img_stride))
            text = _prompt_text(processor, msgs)
            enc = processor(text=text, images=_load_images(msgs, export_dir), return_tensors="pt")
    ids_a, offs = list(a["input_ids"]), list(a["offset_mapping"])
    ranges = [(r["start"], r["end"]) for r in record["supervision"]["assistant_excluded_char_ranges"]
              if r.get("reason") not in set(unmask_reasons)]
    lab_a = []
    for tid, (s, e) in zip(ids_a, offs):
        bad = any(s < re_ and e > rs for rs, re_ in ranges)
        lab_a.append(-100 if bad else tid)
    wts = [1.0] * len(ids_a)
    if fail_token_weight != 1.0:
        spans = weighted_spans(asst)
        for t, (s, e) in enumerate(offs):
            if any(s < re_ and e > rs for rs, re_ in spans):
                wts[t] = float(fail_token_weight)
    ids_a.append(EOT_ID)
    lab_a.append(EOT_ID)
    wts.append(1.0)
    P = enc["input_ids"].shape[1]
    out = {
        "input_ids": torch.cat([enc["input_ids"][0], torch.tensor(ids_a)]),
        "labels": torch.cat([torch.full((P,), -100), torch.tensor(lab_a)]),
        "mm_token_type_ids": torch.cat([enc["mm_token_type_ids"][0],
                                        torch.zeros(len(ids_a), dtype=enc["mm_token_type_ids"].dtype)]),
        "weights": torch.cat([torch.ones(P), torch.tensor(wts)]),
        "n_prompt": P, "img_stride": img_stride, "n_asst": len(ids_a), "offsets": offs,
        "pixel_values": enc["pixel_values"], "image_position_ids": enc["image_position_ids"],
    }
    return out


_WSPAN = re.compile(r'"failure":\s*(true|false)|"categories":\s*\[[^\]]*\]')


def weighted_spans(asst):
    """Char spans of the value of every "failure" key and of every "categories" list value."""
    out = []
    for m in _WSPAN.finditer(asst):
        if m.group(1):
            out.append((m.start(1), m.end(1)))
        else:
            out.append((m.start() + m.group().index("["), m.end()))
    return out


def teacher_has_failure(record):
    return any(s["failure"] for s in json.loads(record["messages"][1]["content"])["segments"])


def make_split(records=None, n_eval=50, seed=0, path=os.path.join(OUT_DIR, "split.json")):
    records = records or load_records()
    rnd = random.Random(seed)
    strata = {True: [], False: []}
    for r in records:
        strata[teacher_has_failure(r)].append(r["id"])
    ev = []
    for k in (True, False):
        ids = sorted(strata[k])
        rnd.shuffle(ids)
        ev += ids[: round(n_eval * len(ids) / len(records))]
    ev = sorted(ev)
    if len(ev) != n_eval:
        rest = sorted(set(r["id"] for r in records) - set(ev))
        rnd.shuffle(rest)
        ev = sorted(ev + rest[: n_eval - len(ev)]) if len(ev) < n_eval else ev[:n_eval]
    evs = set(ev)
    tr = sorted(r["id"] for r in records if r["id"] not in evs)
    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump({"seed": seed, "train": tr, "eval": ev, "n_failure_total": len(strata[True]),
               "n_failure_eval": sum(1 for i in ev if i in set(strata[True]))}, open(path, "w"), indent=1)
    return tr, ev


def load_split(path=os.path.join(OUT_DIR, "split.json")):
    return json.load(open(path))


def save_cfg(cfg, *dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        json.dump(cfg.__dict__, open(os.path.join(d, "input_cfg.json"), "w"), indent=1)


def load_cfg(path):
    """path: json file or an adapter dir containing input_cfg.json."""
    if os.path.isdir(path):
        path = os.path.join(path, "input_cfg.json")
    return InputCfg(**json.load(open(path)))
