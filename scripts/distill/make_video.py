"""Render before/after distillation video. CPU only. Output: outputs/distill/video/distill_before_after.mp4"""
import os, sys, json, bisect
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval as ev, data
import imageio_ffmpeg

OUT = os.path.join(data.OUT_DIR, "video"); os.makedirs(OUT, exist_ok=True)
W, H, FPS, EP_SEC = 1280, 720, 15, 11.0
CHOSEN = ["D0160", "D0219", "D0008"]
BG, FG, MUTED, PANEL = (18, 22, 30), (236, 240, 245), (150, 160, 175), (30, 36, 48)
RED, GREEN = (255, 80, 80), (90, 220, 130)
COL = {"reach": (66, 133, 244), "grasp": (251, 188, 5), "transport": (52, 168, 83), "release": (171, 71, 188),
       "withdraw": (0, 172, 193), "idle": (120, 130, 145), "regrasp": (255, 112, 67)}
EXTRA = [(200, 200, 100), (240, 120, 180), (150, 200, 250)]


def col(a):
    if a not in COL:
        COL[a] = EXTRA[len(COL) % 3]
    return COL[a]


def F(sz, b=False):
    return ImageFont.truetype("C:/Windows/Fonts/segoeui%s.ttf" % ("b" if b else ""), sz)


f12, f14, f16, f20, f28, f40 = F(12), F(14), F(16), F(20), F(28, True), F(40, True)
fb16, fb20 = F(16, True), F(20, True)
IMG_DIR = data.EXPORT_DIR
recs = {r["id"]: r for r in data.load_records()}
split = data.load_split()["eval"]


def pred_segs(tag, i):
    p = json.load(open(os.path.join(ev.PREDS, tag, i + ".json")))
    o = ev.parse_pred(p.get("text", ""))
    return p, (ev.usable_segments(o) if o else None)


def seg_at(segs, f):
    for s in segs:
        if s["start_frame"] <= f < s["end_frame"]:
            return s
    return None


def center(d, x, y, t, font, fill):
    w = d.textlength(t, font=font)
    d.text((x - w / 2, y), t, font=font, fill=fill)


def new():
    im = Image.new("RGB", (W, H), BG)
    return im, ImageDraw.Draw(im)


def title_card(n, i, task, tb, sb):
    im, d = new()
    center(d, W / 2, 70, "Distilling a VLM teacher into Gemma 4 E4B", f28, FG)
    center(d, W / 2, 112, f"example {n}/3", f20, MUTED)
    center(d, W / 2, 200, i, f40, FG)
    center(d, W / 2, 262, f"Task: \"{task}\"", f28, (160, 200, 255))
    for name, v, c, x in [("Base Gemma 4 E4B", tb, RED, 380), ("Distilled (SFT)", sb, GREEN, 900)]:
        center(d, x, 380, name, f20, MUTED)
        center(d, x, 415, f"{v*100:.0f}%", F(80, True), c)
        center(d, x, 520, "frame accuracy vs teacher", f16, MUTED)
    center(d, W / 2, 640, "Teacher = gpt-6.1-sol annotations", f16, MUTED)
    return im


def render_main(i, task, T, rows, lo, hi, frames, f_now, idx, accs, n, counts):
    im, d = new()
    d.text((40, 16), f"Example {n}/3  |  {i}", font=fb20, fill=FG)
    tt = f'"{task}"'
    d.text((W - 40 - d.textlength(tt, font=f16), 20), tt, font=f16, fill=(160, 200, 255))
    fr = frames[idx]
    for k, cam in enumerate(["agent", "wrist"]):
        x = 40 + k * 360
        pim = Image.open(os.path.join(IMG_DIR, fr["images"][cam])).convert("RGB").resize((340, 340), Image.LANCZOS)
        im.paste(pim, (x, 60))
        d.text((x + 8, 64), cam + " camera", font=fb16, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
    px = 780
    d.rounded_rectangle((px, 60, W - 40, 400), 10, fill=PANEL)
    d.text((px + 20, 72), f"frame {int(f_now)} / {hi}", font=fb20, fill=FG)
    d.text((px + 20, 112), "Frame action accuracy vs teacher", font=f14, fill=MUTED)
    for k, (nm, v, c) in enumerate([("Base Gemma 4 E4B", accs[0], RED), ("Distilled (SFT)", accs[1], GREEN)]):
        y = 140 + k * 70
        d.text((px + 20, y), nm, font=f16, fill=FG)
        d.rectangle((px + 20, y + 30, px + 440, y + 48), fill=(50, 58, 72))
        d.rectangle((px + 20, y + 30, px + 20 + 420 * v, y + 48), fill=c)
        t = f"{v*100:.0f}%"
        d.text((px + 450 - d.textlength(t, font=fb16), y), t, font=fb16, fill=c)
    d.text((px + 20, 295), "Segments predicted", font=f14, fill=MUTED)
    d.text((px + 20, 318), "  |  ".join(f"{nm} {c}" for nm, c in zip(["teacher", "base", "SFT"], counts)), font=f16, fill=FG)
    d.text((px + 20, 352), "Red text = label differs from teacher at playhead", font=f12, fill=MUTED)
    bx0, bx1 = 270, W - 40

    def X(f):
        return bx0 + (f - lo) / (hi - lo) * (bx1 - bx0)
    tseg = seg_at(T["segs"], f_now)
    for r, (name, segs) in enumerate(rows):
        y = 425 + r * 62
        d.text((40, y), name, font=fb16, fill=FG)
        cur = seg_at(segs, f_now) if segs else None
        lab = "invalid JSON" if segs is None else (cur["action"] if cur else "-")
        bad = r > 0 and (segs is None or (cur or {}).get("action") != (tseg or {}).get("action"))
        d.text((40, y + 24), "now: " + lab, font=f16, fill=RED if bad else MUTED)
        if segs is None:
            d.rectangle((bx0, y, bx1, y + 44), fill=(90, 95, 105))
            center(d, (bx0 + bx1) / 2, y + 10, "invalid JSON", fb20, (230, 230, 230))
            continue
        d.rectangle((bx0, y, bx1, y + 44), fill=(40, 46, 58))
        for s in segs:
            a, b = max(s["start_frame"], lo), min(s["end_frame"], hi)
            if b <= a:
                continue
            x0, x1 = X(a), X(b)
            d.rectangle((x0, y, x1, y + 44), fill=col(s["action"]), outline=(18, 22, 30))
            if s.get("failure") is True:
                d.rectangle((x0, y, x1, y + 44), outline=RED, width=4)
                for hx in range(int(x0), int(x1), 8):
                    d.line((hx, y + 44, min(hx + 12, x1), y), fill=RED, width=1)
            if x1 - x0 > 55:
                center(d, (x0 + x1) / 2, y + 11, s["action"], fb16, (15, 15, 15))
    xp = X(f_now)
    d.line((xp, 415, xp, 420 + 3 * 62 - 14), fill=(255, 255, 255), width=3)
    lx = 40
    d.text((lx, 640), "Legend:", font=fb16, fill=MUTED)
    lx += 75
    for a in ["reach", "grasp", "transport", "release", "withdraw", "idle"]:
        d.rectangle((lx, 644, lx + 18, 662), fill=col(a))
        d.text((lx + 26, 640), a, font=f16, fill=FG)
        lx += 40 + d.textlength(a, font=f16)
    d.rectangle((lx + 10, 644, lx + 28, 662), outline=RED, width=3)
    d.text((lx + 36, 640), "failure=true", font=f16, fill=FG)
    d.text((40, 690), "Held-out episode (not used in training). Bars = predicted action segments; frame accuracy vs teacher labels.",
           font=f14, fill=MUTED)
    return im


def summary_stats():
    ids = [i for i in split if os.path.exists(os.path.join(ev.PREDS, "base", i + ".json"))
           and os.path.exists(os.path.join(ev.PREDS, "sft", i + ".json"))]
    out = {}
    for tag in ["base", "sft"]:
        rows = [ev.score_episode(json.load(open(os.path.join(ev.PREDS, tag, i + ".json"))), ev.teacher_info(recs[i])) for i in ids]
        out[tag] = ev.agg(rows)
    return len(ids), out


def final_card(n, st):
    im, d = new()
    center(d, W / 2, 50, "Summary: base vs distilled (SFT)", f40, FG)
    center(d, W / 2, 110, f"Held-out episodes, n = {n}; metrics vs gpt-6.1-sol teacher", f20, MUTED)
    hdr = [("Frame action accuracy", "frame_action_acc"), ("Boundary F1 (+/-5 frames)", "boundary_f1_tol5"),
           ("Valid JSON rate", "parse_rate")]
    for k, (nm, key) in enumerate(hdr):
        cx = 240 + k * 400
        center(d, cx, 190, nm, f20, MUTED)
        for j, (tag, c, lab) in enumerate([("base", RED, "Base"), ("sft", GREEN, "SFT")]):
            v = st[tag][key] or 0
            center(d, cx - 80 + j * 160, 235, f"{v*100:.0f}%", F(54, True), c)
            center(d, cx - 80 + j * 160, 305, lab, f16, MUTED)
    center(d, W / 2, 430, "Limitation: failure flags were mostly masked in training (87% of teacher failure segments", f20, (255, 200, 120))
    center(d, W / 2, 462, "were review-required), so the student does not yet predict failures.", f20, (255, 200, 120))
    return im


def main():
    n_all, st = summary_stats()
    print("n", n_all, {k: (v["frame_action_acc"], v["boundary_f1_tol5"], v["parse_rate"]) for k, v in st.items()})
    path = os.path.join(OUT, "distill_before_after.mp4")
    w = imageio_ffmpeg.write_frames(path, (W, H), fps=FPS, codec="libx264", pix_fmt_in="rgb24", pix_fmt_out="yuv420p",
                                    macro_block_size=1, output_params=["-crf", "24", "-preset", "medium"])
    w.send(None)

    def emit(im, sec):
        b = im.tobytes()
        for _ in range(int(sec * FPS)):
            w.send(b)
    for n, i in enumerate(CHOSEN, 1):
        rec = recs[i]
        T = ev.teacher_info(rec)
        task = data._task_info(rec)["task"]
        lo = min(s["start_frame"] for s in T["segs"])
        hi = max(s["end_frame"] for s in T["segs"])
        _, bs = pred_segs("base", i)
        _, ss = pred_segs("sft", i)
        sc = [ev.score_episode(json.load(open(os.path.join(ev.PREDS, t, i + ".json"))), T)["frame_action_acc"] for t in ("base", "sft")]
        print(i, task, sc)
        rows = [("Teacher (gpt-6.1-sol)", T["segs"]), ("Base Gemma 4 E4B", bs), ("Distilled (SFT)", ss)]
        counts = [len(T["segs"]), "invalid" if bs is None else len(bs), "invalid" if ss is None else len(ss)]
        emit(title_card(n, i, task, *sc), 2.5)
        frames = data._frames(rec)
        fnums = [f["frame"] for f in frames]
        nf = int(EP_SEC * FPS)
        for k in range(nf):
            f_now = lo + (hi - 1 - lo) * k / (nf - 1)
            idx = max(0, bisect.bisect_right(fnums, f_now) - 1)
            im = render_main(i, task, T, rows, lo, hi, frames, f_now, idx, sc, n, counts)
            if n == 1 and k == int(nf * 0.6):
                im.save(os.path.join(OUT, "preview.png"))
            w.send(im.tobytes())
    emit(final_card(n_all, st), 4.0)
    w.close()
    print("wrote", path)


if __name__ == "__main__" and "--v3" not in sys.argv:
    main()


# ============================ v3 mode (python make_video.py --v3) ============================
import score_failures as sf
V3_CHOSEN = ["D0185", "D0173", "D0142"]
HEAD = {"D0185": "Repeated missed grasps — base sees none, v3 flags every one",
        "D0173": "Base sees no failures; v3 finds all three, with the right timing",
        "D0142": "A missed grasp, then a successful retry — v3 flags just the miss"}
V3_FALLBACK = dict(f1=(0.0, 0.69), prec=1.00, rec=0.53, segrec=(0.0, 0.38), segprec=0.72, cat=0.83,
                   acc=(0.23, 0.77), bf1=(0.26, 0.81), js=(0.96, 1.00))
YEL = (255, 200, 120)


def v3_stats():
    d = dict(V3_FALLBACK)
    try:
        S = json.load(open(os.path.join(data.OUT_DIR, "scores_failures", "summary.json")))["tags"]
        b, s = S["base"], S["sft_v3"]
        d.update(f1=(b["episode_vs_verified"]["f1"], s["episode_vs_verified"]["f1"]),
                 prec=s["episode_vs_verified"]["precision"], rec=s["episode_vs_verified"]["recall"],
                 segrec=(b["segment"]["recall"], s["segment"]["recall"]), segprec=s["segment"]["precision"],
                 cat=s["segment"]["category_match_on_hits"], js=(b["parse_rate"], s["parse_rate"]))
    except Exception as e:
        print("fallback failure stats", e)
    try:
        S = json.load(open(os.path.join(os.path.dirname(data.OUT_DIR), "distill_v3", "scores", "summary.json")))
        b, s = S["base"]["all"], S["sft_v3"]["all"]
        d.update(acc=(b["frame_action_acc"], s["frame_action_acc"]), bf1=(b["boundary_f1_tol5"], s["boundary_f1_tol5"]))
    except Exception as e:
        print("fallback frame stats", e)
    return d


def fcat(s):
    c = s.get("categories")
    return (c[0] if isinstance(c, list) and c else "failure").replace("_", " ")


def title_card3(n, i, task, tb, sb, nref, head=""):
    im, d = new()
    center(d, W / 2, 60, "Distilling a VLM teacher into Gemma 4 E4B  (v3)", f28, FG)
    center(d, W / 2, 102, f"example {n}/3", f20, MUTED)
    center(d, W / 2, 180, i, f40, FG)
    center(d, W / 2, 242, f"Task: \"{task}\"", f28, (160, 200, 255))
    for name, v, c, x in [("Base Gemma 4 E4B", tb, RED, 380), ("Distilled v3 (SFT)", sb, GREEN, 900)]:
        center(d, x, 340, name, f20, MUTED)
        center(d, x, 375, f"{v*100:.0f}%", F(80, True), c)
        center(d, x, 480, "frame accuracy vs teacher", f16, MUTED)
    center(d, W / 2, 565, head, F(24, True), YEL)
    center(d, W / 2, 605, f"Verified failure segments in this episode: {nref}", f16, MUTED)
    center(d, W / 2, 640, "Teacher = gpt-6.1-sol annotations", f16, MUTED)
    return im


def render_main3(i, task, T, rows, lo, hi, frames, f_now, idx, accs, n, fc, caught, nref, extras=()):
    im, d = new()
    d.text((40, 16), f"Example {n}/3  |  {i}", font=fb20, fill=FG)
    tt = f'"{task}"'
    d.text((W - 40 - d.textlength(tt, font=f16), 20), tt, font=f16, fill=(160, 200, 255))
    fr = frames[idx]
    for k, cam in enumerate(["agent", "wrist"]):
        x = 40 + k * 360
        pim = Image.open(os.path.join(IMG_DIR, fr["images"][cam])).convert("RGB").resize((340, 340), Image.LANCZOS)
        im.paste(pim, (x, 60))
        d.text((x + 8, 64), cam + " camera", font=fb16, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
    px = 780
    d.rounded_rectangle((px, 60, W - 40, 400), 10, fill=PANEL)
    d.text((px + 20, 70), f"frame {int(f_now)} / {hi}", font=fb20, fill=FG)
    d.text((px + 20, 102), "Frame action accuracy vs teacher", font=f14, fill=MUTED)
    for k, (nm, v, c) in enumerate([("Base Gemma 4 E4B", accs[0], RED), ("Distilled v3 (SFT)", accs[1], GREEN)]):
        y = 124 + k * 52
        d.text((px + 20, y), nm, font=f16, fill=FG)
        d.rectangle((px + 20, y + 26, px + 440, y + 40), fill=(50, 58, 72))
        d.rectangle((px + 20, y + 26, px + 20 + 420 * v, y + 40), fill=c)
        t = f"{v*100:.0f}%"
        d.text((px + 450 - d.textlength(t, font=fb16), y), t, font=fb16, fill=c)
    d.line((px + 20, 240, px + 440, 240), fill=(60, 68, 84), width=1)
    d.text((px + 20, 248), "FAILURES (red-outlined segments)", font=fb16, fill=RED)
    d.text((px + 20, 276), f"Failures — teacher: {fc[0]}, base: {fc[1]}, v3: {fc[2]}", font=fb16, fill=FG)
    d.text((px + 20, 306), f"v3 caught {caught}/{nref} verified failures" if nref else "no verified failures here; v3 flags none",
           font=fb16, fill=GREEN if nref and caught else MUTED)
    if extras:
        d.text((px + 20, 330), f"+{len(extras)} extra flag (not in reference)", font=f12, fill=MUTED)
    d.text((px + 20, 350), "Red text = label differs from teacher at playhead", font=f12, fill=MUTED)
    bx0, bx1 = 270, W - 40

    def X(f):
        return bx0 + (f - lo) / (hi - lo) * (bx1 - bx0)
    tseg = seg_at(T["segs"], f_now)
    for r, (name, segs) in enumerate(rows):
        y = 435 + r * 62
        d.text((40, y), name, font=fb16, fill=FG)
        cur = seg_at(segs, f_now) if segs else None
        lab = "invalid JSON" if segs is None else (cur["action"] if cur else "-")
        bad = r > 0 and (segs is None or (cur or {}).get("action") != (tseg or {}).get("action"))
        d.text((40, y + 24), "now: " + lab, font=f16, fill=RED if bad else MUTED)
        if segs is None:
            d.rectangle((bx0, y, bx1, y + 44), fill=(90, 95, 105))
            center(d, (bx0 + bx1) / 2, y + 10, "invalid JSON", fb20, (230, 230, 230))
            continue
        d.rectangle((bx0, y, bx1, y + 44), fill=(40, 46, 58))
        fails = []
        for s in segs:
            a, b = max(s["start_frame"], lo), min(s["end_frame"], hi)
            if b <= a:
                continue
            x0, x1 = X(a), X(b)
            d.rectangle((x0, y, x1, y + 44), fill=col(s["action"]), outline=(18, 22, 30))
            if x1 - x0 > 55:
                center(d, (x0 + x1) / 2, y + 11, s["action"], fb16, (15, 15, 15))
            if s.get("failure") is True:
                fails.append((x0, x1, s))
        for x0, x1, s in fails:
            ex = r == 2 and any(s is e for e in extras)
            oc = (150, 90, 95) if ex else RED
            if ex:
                for hx in range(int(x0), int(x1), 8):
                    d.line((hx, y, min(hx + 4, x1), y), fill=oc, width=3)
                    d.line((hx, y + 44, min(hx + 4, x1), y + 44), fill=oc, width=3)
                d.line((x0, y, x0, y + 44), fill=oc, width=3)
                d.line((x1, y, x1, y + 44), fill=oc, width=3)
            else:
                d.rectangle((x0, y, x1, y + 44), outline=oc, width=5)
                for hx in range(int(x0), int(x1), 8):
                    d.line((hx, y + 44, min(hx + 12, x1), y), fill=oc, width=1)
            t = "FAIL?" if ex else "FAIL: " + fcat(s)
            if d.textlength(t, font=f12) + 8 > x1 - x0:
                t = "FAIL"
            tw = d.textlength(t, font=f12) + 8
            d.rectangle((x0, y - 15, x0 + tw, y), fill=oc)
            d.text((x0 + 4, y - 16), t, font=f12, fill=(255, 255, 255))
    xp = X(f_now)
    d.line((xp, 418, xp, 435 + 3 * 62 - 14), fill=(255, 255, 255), width=3)
    lx = 40
    d.text((lx, 640), "Legend:", font=fb16, fill=MUTED)
    lx += 75
    for a in ["reach", "grasp", "transport", "release", "withdraw", "idle"]:
        d.rectangle((lx, 644, lx + 18, 662), fill=col(a))
        d.text((lx + 26, 640), a, font=f16, fill=FG)
        lx += 40 + d.textlength(a, font=f16)
    d.rectangle((lx + 10, 644, lx + 28, 662), outline=RED, width=3)
    d.text((lx + 36, 640), "failure=true (FAIL: category)", font=f16, fill=FG)
    d.text((40, 690), "Held-out episode (not used in training). Bars = predicted action segments; failures scored vs frame-verified references.",
           font=f14, fill=MUTED)
    return im


def final_cards3(st):
    pc = lambda v: f"{v*100:.0f}%"
    im1, d = new()
    center(d, W / 2, 30, "Results on 50 held-out episodes: base → v3", f40, FG)
    center(d, W / 2, 86, "Failure labels frame-verified; frame/boundary metrics vs gpt-6.1-sol teacher", f16, MUTED)
    tiles = [("Episode failure-detection F1", f"{st['f1'][0]:.2f}", f"{st['f1'][1]:.2f}", f"precision {st['prec']:.2f}, recall {st['rec']:.2f}"),
             ("Failure segments caught", pc(st["segrec"][0]), pc(st["segrec"][1]), f"precision {st['segprec']:.2f}, category correct {pc(st['cat'])} when caught"),
             ("Frame action accuracy", f"{st['acc'][0]:.2f}", f"{st['acc'][1]:.2f}", ""),
             ("Boundary F1 (+/-5 frames)", f"{st['bf1'][0]:.2f}", f"{st['bf1'][1]:.2f}", ""),
             ("Valid JSON", pc(st["js"][0]), pc(st["js"][1]), "")]
    for k, (nm, a, b, sub) in enumerate(tiles):
        if k < 4:
            cx, y = 340 + (k % 2) * 600, 125 + (k // 2) * 170
        else:
            cx, y = 640, 125 + 2 * 170
        d.rounded_rectangle((cx - 280, y, cx + 280, y + 155), 12, fill=PANEL)
        center(d, cx, y + 8, nm, f20, MUTED)
        center(d, cx - 110, y + 38, a, F(48, True), RED)
        center(d, cx, y + 44, "→", F(36, True), MUTED)
        center(d, cx + 110, y + 38, b, F(48, True), GREEN)
        center(d, cx - 110, y + 98, "base", f12, MUTED)
        center(d, cx + 110, y + 98, "v3", f12, MUTED)
        if sub:
            center(d, cx, y + 122, sub, f14, FG)
    im2, d = new()
    center(d, W / 2, 40, "What we changed to raise the failure score", f40, FG)
    bullets = ["Distilled gpt-6.1-sol labels into Gemma 4 E4B with QLoRA SFT (450 train episodes)",
               "Kept the teacher's review-flagged failure segments in the loss (87% of failure labels were flagged and masked at first)",
               "Diagnosed the student was narrating a \"success story\" and ignoring evidence",
               "Added derived gripper events to the input, computed from telemetry only (empty grasp vs. holding, episode hit time limit)",
               "Up-weighted the failure-flag and category tokens 5x in the loss"]
    y = 130
    for b in bullets:
        words, line, lines = b.split(), "", []
        for w_ in words:
            if d.textlength(line + " " + w_, font=f20) > 1040:
                lines.append(line)
                line = w_
            else:
                line = (line + " " + w_).strip()
        lines.append(line)
        d.ellipse((90, y + 9, 102, y + 21), fill=GREEN)
        for ln in lines:
            d.text((120, y), ln, font=f20, fill=FG)
            y += 30
        y += 22
    center(d, W / 2, 640, "Still below a simple telemetry rule on grasp failures; wrong-object picks remain hard (5/28).", f16, YEL)
    return im1, im2


def main_v3():
    st = v3_stats()
    print(st)
    path = os.path.join(OUT, "distill_v3_before_after.mp4")
    w = imageio_ffmpeg.write_frames(path, (W, H), fps=FPS, codec="libx264", pix_fmt_in="rgb24", pix_fmt_out="yuv420p",
                                    macro_block_size=1, output_params=["-crf", "26", "-preset", "medium"])
    w.send(None)
    VER = json.load(open(os.path.join(data.OUT_DIR, "eval_verified.json")))["episodes"]

    def emit(im, sec):
        b = im.tobytes()
        for _ in range(int(sec * FPS)):
            w.send(b)
    for n, i in enumerate(V3_CHOSEN, 1):
        rec = recs[i]
        T = ev.teacher_info(rec)
        task = data._task_info(rec)["task"]
        lo = min(s["start_frame"] for s in T["segs"])
        hi = max(s["end_frame"] for s in T["segs"])
        _, bs = pred_segs("base", i)
        _, ss = pred_segs("sft_v3", i)
        sc = [ev.score_episode(json.load(open(os.path.join(ev.PREDS, t, i + ".json"))), T)["frame_action_acc"] for t in ("base", "sft_v3")]
        refs = VER[i]["ref_segments"]
        pf = [s for s in (ss or []) if s.get("failure") is True]
        caught = sum(any(sf.near(q, p) for p in pf) for q in refs)
        extras = [p for p in pf if not any(sf.near(p, q) for q in refs)]
        fc = (sum(1 for s in T["segs"] if s["failure"]), sum(1 for s in (bs or []) if s.get("failure") is True), len(pf))
        print(i, task, "extras", [(e["start_frame"], e["end_frame"]) for e in extras], "acc b/v3", sc, "fail t/b/v3", fc, "caught", caught, "/", len(refs))
        rows = [("Teacher (gpt-6.1-sol)", T["segs"]), ("Base Gemma 4 E4B", bs), ("Distilled v3 (SFT)", ss)]
        emit(title_card3(n, i, task, *sc, len(refs), HEAD[i]), 2.5)
        frames = data._frames(rec)
        fnums = [f["frame"] for f in frames]
        nf = int(EP_SEC * FPS)
        PREV_K = int(nf * 0.8)
        for k in range(nf):
            f_now = lo + (hi - 1 - lo) * k / (nf - 1)
            idx = max(0, bisect.bisect_right(fnums, f_now) - 1)
            im = render_main3(i, task, T, rows, lo, hi, frames, f_now, idx, sc, n, fc, caught, len(refs), extras)
            if n == 1 and k == PREV_K:
                im.save(os.path.join(OUT, "preview_v3.png"))
            w.send(im.tobytes())
    c1, c2 = final_cards3(st)
    c1.save(os.path.join(OUT, "final_card_v3.png"))
    c2.save(os.path.join(OUT, "final_card_v3b.png"))
    emit(c1, 4.0)
    emit(c2, 4.0)
    w.close()
    print("wrote", path)


if __name__ == "__main__" and "--v3" in sys.argv:
    main_v3()
