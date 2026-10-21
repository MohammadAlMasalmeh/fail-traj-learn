"""QLoRA SFT of Gemma 4 on the sol_v8 teacher labels (masked per dataset contract).

python scripts/distill/train.py --base google/gemma-4-E4B-it --epochs 2 --stride 3 --max_hours 5
"""
import argparse, json, math, os, random, sys, threading, queue, time
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data
from model import load_model, load_processor
from lossfn import sft_loss

ap = argparse.ArgumentParser()
ap.add_argument("--base", default=data.BASE_ID)
ap.add_argument("--out", "--out-dir", dest="out", default=data.OUT_DIR)
ap.add_argument("--unmask-reasons", nargs="*", default=[], help="excluded-range reasons that are NOT masked (e.g. review_required_segment)")
ap.add_argument("--epochs", type=float, default=2.0)
ap.add_argument("--accum", type=int, default=8)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--warmup", type=float, default=0.03)
ap.add_argument("--stride", type=int, default=data.DEFAULT_CFG.stride)
ap.add_argument("--cams", default=data.DEFAULT_CFG.cams)
ap.add_argument("--max_soft_tokens", type=int, default=data.DEFAULT_CFG.max_soft_tokens)
ap.add_argument("--max_hours", type=float, default=5.0, help="wall-clock cap; schedule is NOT stretched, training just stops (adapter saved)")
ap.add_argument("--max_steps", type=int, default=0, help="debug: stop after N optimizer steps")
ap.add_argument("--max_len", type=int, default=10000, help="skip examples longer than this (0 = none)")
ap.add_argument("--resume", default="", help="adapter dir from a previous run's ckpt (weights only; optimizer restarts)")
ap.add_argument("--resume_examples", type=int, default=0)
ap.add_argument("--fail-token-weight", type=float, default=1.0, help="loss weight for failure true/false and categories-list tokens")
ap.add_argument("--events", action="store_true", help="append derived gripper events block to the student input")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

cfg = data.InputCfg(stride=args.stride, cams=args.cams, max_soft_tokens=args.max_soft_tokens, events=args.events)
os.makedirs(os.path.join(args.out, "ckpt"), exist_ok=True)
log_path = os.path.join(args.out, "train_log.jsonl")
data.save_cfg(cfg, args.out)
logf = open(log_path, "a", encoding="utf8")


def log(d):
    d["time"] = time.time()
    logf.write(json.dumps(d) + "\n"); logf.flush()
    print(json.dumps(d), flush=True)


recs = {r["id"]: r for r in data.load_records()}
split = data.load_split()
train_ids = split["train"]
assert not set(train_ids) & set(split["eval"])
proc = load_processor(args.base)
n_per_epoch = len(train_ids)
total_examples = int(args.epochs * n_per_epoch)
total_steps = total_examples // args.accum
warm = max(1, int(round(args.warmup * total_steps)))
rnd = random.Random(args.seed)
order = []
for e in range(math.ceil(args.epochs)):
    ids = list(train_ids); rnd.shuffle(ids); order += ids
order = order[:total_examples]
log({"event": "start", "args": vars(args), "total_examples": total_examples, "total_steps": total_steps, "warmup_steps": warm})

model = load_model(args.base, adapter_path=args.resume or None, trainable=True, grad_ckpt=True)
model.train()
params = [p for p in model.parameters() if p.requires_grad]
import bitsandbytes as bnb
opt = bnb.optim.PagedAdamW8bit(params, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0)


def lr_at(step):  # step is 0-based optimizer step
    if step < warm:
        return args.lr * (step + 1) / warm
    prog = (step - warm) / max(1, total_steps - warm)
    return args.lr * (0.5 * (1 + math.cos(math.pi * prog)) * 0.9 + 0.1)  # cosine to 10% of peak


# background prefetch of encoded examples
Q = queue.Queue(maxsize=6)
start_idx = args.resume_examples


def producer():
    for i in range(start_idx, len(order)):
        ex = data.encode_train(proc, recs[order[i]], cfg=cfg, unmask_reasons=args.unmask_reasons,
                             fail_token_weight=args.fail_token_weight, max_len=args.max_len)
        if ex["img_stride"] != cfg.stride:
            log({"event": "img_drop", "id": order[i], "img_stride": ex["img_stride"], "seq": len(ex["input_ids"])})
        ex["id"] = order[i]
        Q.put(ex)
    Q.put(None)


threading.Thread(target=producer, daemon=True).start()

ckpt_every = max(1, total_steps // 4)
t0 = time.time()
step = start_idx // args.accum
done = start_idx
group = []


def save(path):
    model.save_pretrained(path)
    data.save_cfg(cfg, path)
    json.dump({"step": step, "examples": done, "args": vars(args)}, open(os.path.join(path, "state.json"), "w"))


def run_group(group):
    global step
    N = sum(float(e["weights"][1:][e["labels"][1:] != -100].sum()) for e in group)  # weighted token count (= #tokens when weight is 1)
    tot = 0.0
    tg = time.time()
    for e in group:
        try:
            ls, n = sft_loss(model, e)
            (ls / N).backward()
        except torch.cuda.OutOfMemoryError:
            log({"event": "oom_skip", "id": e["id"], "seq": len(e["input_ids"])})
            ls = n = None
            import gc; gc.collect(); torch.cuda.empty_cache()
            continue
        tot += ls.item()
        log({"event": "example", "id": e["id"], "seq": len(e["input_ids"]), "n_sup": n, "loss": ls.item() / max(1, n),
             "peak_gb": torch.cuda.max_memory_allocated() / 1e9})
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    for g in opt.param_groups:
        g["lr"] = lr_at(step)
    opt.step(); opt.zero_grad(set_to_none=True)
    step += 1
    log({"event": "step", "step": step, "of": total_steps, "loss": tot / N, "lr": lr_at(step - 1),
         "sec": time.time() - tg, "elapsed_h": (time.time() - t0) / 3600})


stop = False
while True:
    ex = Q.get()
    if ex is None:
        break
    if args.max_len and len(ex["input_ids"]) > args.max_len:
        done += 1
        continue
    group.append(ex); done += 1
    if len(group) == args.accum:
        run_group(group); group = []
        if step % ckpt_every == 0 and step < total_steps:
            save(os.path.join(args.out, "ckpt", f"step{step}"))
        if args.max_steps and step >= args.max_steps:
            stop = True
        if (time.time() - t0) / 3600 > args.max_hours:
            log({"event": "time_cap_hit", "step": step}); stop = True
        if stop:
            break
if group and not stop:
    run_group(group)
save(os.path.join(args.out, "adapter_final"))
log({"event": "finished", "step": step, "hours": (time.time() - t0) / 3600})
