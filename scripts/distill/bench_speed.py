"""Inference speed benchmark for v3. python bench_speed.py --mode {base,merged} --bs N --n 8 [--static] [--tag X]
base   = current load_model path (PLE on CPU, unmerged LoRA).
merged = NF4 base + LoRA merged into weights (merge_and_unload), PLE kept on GPU (bf16 unless --ple8).
Results appended to outputs/distill/speed/results.jsonl, texts in outputs/distill/speed/<tag>/."""
import argparse, json, os, sys, time, gc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, data
import eval as ev

ap = argparse.ArgumentParser()
ap.add_argument("--mode", default="base"); ap.add_argument("--bs", type=int, default=1)
ap.add_argument("--n", type=int, default=8); ap.add_argument("--max_new", type=int, default=3072)
ap.add_argument("--static", action="store_true"); ap.add_argument("--compile", action="store_true")
ap.add_argument("--tag", default=None)
a = ap.parse_args()
tag = a.tag or f"{a.mode}_bs{a.bs}" + ("_static" if a.static else "") + ("_compile" if a.compile else "")
ADAPTER = os.path.join(data.OUT_DIR, "..", "distill_v3", "adapter_final")
ADAPTER = os.path.normpath(ADAPTER)
MERGED = os.environ.get("MERGED_DIR", os.path.expanduser("~/AppData/Local/Temp/claude/merged_v3"))
cfg = data.load_cfg(os.path.join(ADAPTER, "input_cfg.json"))
ids = data.load_split()["eval"][: a.n]
recs = {r["id"]: r for r in data.load_records()}
from model import load_model, load_processor
import model as M
proc = load_processor()
proc.tokenizer.padding_side = "left"
t0 = time.time()
if a.mode == "base":
    model = load_model(adapter_path=ADAPTER).eval()
elif a.mode == "ple_gpu":  # unmerged LoRA, PLE on GPU
    M.offload_ple = lambda m: None
    model = load_model(adapter_path=ADAPTER).eval()
elif a.mode == "merged_exact":  # merged in bf16 on CPU (make_merged.py), then NF4, PLE on GPU
    M.offload_ple = lambda m: None
    model = load_model(base_id=MERGED).eval()
elif a.mode == "merged_cpuple":  # exact merge, NF4, PLE on CPU (frees 5.6 GB for KV/batch)
    model = load_model(base_id=MERGED).eval()
else:
    orig = M.offload_ple
    M.offload_ple = lambda m: None  # keep PLE on GPU
    model = load_model(adapter_path=ADAPTER).eval()
    M.offload_ple = orig
    model = model.merge_and_unload()
    gc.collect(); torch.cuda.empty_cache()
print(f"[bench] loaded in {time.time()-t0:.0f}s, vram alloc {torch.cuda.memory_allocated()/2**30:.2f} GB", flush=True)
EOT = data.EOT_ID

def make_batch(batch_ids):
    texts, imgs = [], []
    proc.image_processor.max_soft_tokens = cfg.max_soft_tokens
    for e in batch_ids:
        msgs = data.build_student_messages(recs[e], cfg=cfg)
        texts.append(data._prompt_text(proc, msgs)); imgs.append(data._load_images(msgs, data.EXPORT_DIR))
    if len(batch_ids) == 1:
        inp = proc(text=texts, images=imgs[0], return_tensors="pt")
    else:
        inp = proc(text=texts, images=imgs, return_tensors="pt", padding=True)
    inp = inp.to("cuda"); inp["pixel_values"] = inp["pixel_values"].to(torch.bfloat16)
    return inp

gk = dict(max_new_tokens=a.max_new, do_sample=False, eos_token_id=EOT)
if a.static: gk["cache_implementation"] = "static"
if a.compile:
    model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=False)
od = os.path.join(data.OUT_DIR, "speed", tag); os.makedirs(od, exist_ok=True)
torch.cuda.reset_peak_memory_stats()
rows = []; T0 = time.time()
for i in range(0, len(ids), a.bs):
    b = ids[i:i + a.bs]
    try:
        inp = make_batch(b); P = inp["input_ids"].shape[1]
        torch.cuda.synchronize(); t = time.time()
        with torch.no_grad():
            out = model.generate(**inp, **gk)
        torch.cuda.synchronize(); dt = time.time() - t
    except Exception as ex:
        print(f"[bench] FAIL {type(ex).__name__}: {str(ex)[:300]}", flush=True)
        rows.append(dict(ids=b, error=f"{type(ex).__name__}: {str(ex)[:200]}")); break
    ntok = 0
    for j, e in enumerate(b):
        g = out[j, P:]
        g = g[: ((g == EOT).nonzero()[0].item() + 1) if (g == EOT).any() else len(g)]
        ntok += len(g)
        txt = proc.tokenizer.decode(g, skip_special_tokens=True)
        json.dump(dict(id=e, text=txt, n_tokens=len(g)), open(os.path.join(od, e + ".json"), "w"))
    rows.append(dict(ids=b, batch_seconds=dt, tokens=ntok, P=P, steps=int(out.shape[1] - P)))
    print(f"[bench] {b} P={P} steps={out.shape[1]-P} real_tokens={ntok} {dt:.0f}s -> {ntok/dt:.2f} tok/s agg, {out.shape[0]*(out.shape[1]-P)/dt:.2f} slot-tok/s", flush=True)
    del out, inp; gc.collect(); torch.cuda.empty_cache()
ok = [r for r in rows if "error" not in r]
res = dict(tag=tag, mode=a.mode, bs=a.bs, n_done=sum(len(r["ids"]) for r in ok), peak_vram_gb=round(torch.cuda.max_memory_allocated()/2**30, 2),
           tok_s=round(sum(r["tokens"] for r in ok)/max(1e-9, sum(r["batch_seconds"] for r in ok)), 2) if ok else None,
           sec_per_ep=round(sum(r["batch_seconds"] for r in ok)/max(1, sum(len(r["ids"]) for r in ok)), 1) if ok else None,
           errors=[r["error"] for r in rows if "error" in r])
# compare vs v3 preds
same = 0; segdiff = 0; faildiff = 0; cmp = 0
def summ(t):
    o = ev.parse_pred(t)
    if not o or not isinstance(o.get("segments"), list): return None
    return [bool(s.get("failure")) for s in o["segments"] if isinstance(s, dict)]
for r in ok:
    for e in r["ids"]:
        ref = json.load(open(os.path.join(data.OUT_DIR, "preds", "sft_v3", e + ".json")))["text"]
        new = json.load(open(os.path.join(od, e + ".json")))["text"]
        cmp += 1; same += ref == new
        sa, sb = summ(ref), summ(new)
        if sa is None or sb is None: segdiff += (sa is None) != (sb is None); continue
        segdiff += len(sa) != len(sb); faildiff += sa != sb
res.update(compared=cmp, identical_text=same, seg_count_differs=segdiff, failure_flags_differ=faildiff)
print("[bench] RESULT", json.dumps(res), flush=True)
open(os.path.join(data.OUT_DIR, "speed", "results.jsonl"), "a").write(json.dumps(res) + "\n")
