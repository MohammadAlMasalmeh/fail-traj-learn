"""Data utilities for SFT distillation of the sol_v8 teacher labels into Gemma 4.

Public API (reused by eval):
  load_records(), build_student_messages(record, export_dir, cfg) -> messages
  encode_prompt(processor, messages, export_dir, cfg) -> BatchFeature (for generation)
  encode_train(processor, record, export_dir, cfg) -> dict(input_ids, labels, mm_token_type_ids, pixel_values, ...)
  make_split(), load_split()
"""
import json, os, random, sys
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


def encode_train(processor, record, export_dir=EXPORT_DIR, cfg: InputCfg = DEFAULT_CFG, with_pixels=True, unmask_reasons=()):
    """Dict of tensors with labels. The assistant string is tokenized separately with char offsets (exact contract
    coordinates); the prompt ends with the generation prefix (special token), so no tokenization merge at the seam."""
    import torch
    msgs = build_student_messages(record, export_dir, cfg)
    asst = record["messages"][1]["content"]
    assert isinstance(asst, str)
    processor.image_processor.max_soft_tokens = cfg.max_soft_tokens
    text = _prompt_text(processor, msgs)
    imgs = _load_images(msgs, export_dir)
    enc = processor(text=text, images=imgs, return_tensors="pt")
    a = processor.tokenizer(asst, add_special_tokens=False, return_offsets_mapping=True)
    ids_a, offs = list(a["input_ids"]), list(a["offset_mapping"])
    ranges = [(r["start"], r["end"]) for r in record["supervision"]["assistant_excluded_char_ranges"]
              if r.get("reason") not in set(unmask_reasons)]
    lab_a = []
    for tid, (s, e) in zip(ids_a, offs):
        bad = any(s < re_ and e > rs for rs, re_ in ranges)
        lab_a.append(-100 if bad else tid)
    ids_a.append(EOT_ID)
    lab_a.append(EOT_ID)
    P = enc["input_ids"].shape[1]
    out = {
        "input_ids": torch.cat([enc["input_ids"][0], torch.tensor(ids_a)]),
        "labels": torch.cat([torch.full((P,), -100), torch.tensor(lab_a)]),
        "mm_token_type_ids": torch.cat([enc["mm_token_type_ids"][0],
                                        torch.zeros(len(ids_a), dtype=enc["mm_token_type_ids"].dtype)]),
        "n_prompt": P, "n_asst": len(ids_a), "offsets": offs,
        "pixel_values": enc["pixel_values"], "image_position_ids": enc["image_position_ids"],
    }
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
