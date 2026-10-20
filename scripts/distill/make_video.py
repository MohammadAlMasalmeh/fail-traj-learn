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


if __name__ == "__main__":
    main()
