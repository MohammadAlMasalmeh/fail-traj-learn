import sys, os, time, json, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data
from model import load_model, load_processor
from lossfn import sft_loss
stride = int(sys.argv[1]); cams = sys.argv[2]; nstep = int(sys.argv[3]) if len(sys.argv) > 3 else 3
cfg = data.InputCfg(stride=stride, cams=cams)
proc = load_processor()
recs = {r["id"]: r for r in data.load_records()}
split = data.load_split()
rid = max(split["train"], key=lambda i: len(recs[i]["messages"][1]["content"]))
ex = data.encode_train(proc, recs[rid], cfg=cfg)
print("longest-asst episode", rid, "seq", len(ex["input_ids"]), "prompt", ex["n_prompt"], "asst", ex["n_asst"], "sup", int((ex["labels"] != -100).sum()), flush=True)
m = load_model(trainable=True, grad_ckpt=True)
m.train()
params = [p for p in m.parameters() if p.requires_grad]
import bitsandbytes as bnb
opt = bnb.optim.PagedAdamW8bit(params, lr=1e-4)
torch.cuda.reset_peak_memory_stats()
for s in range(nstep):
    t = time.time()
    print("fwd start", flush=True); loss, n = sft_loss(m, ex); print("fwd done", torch.cuda.max_memory_allocated()/1e9, flush=True)
    loss.backward()
    opt.step(); opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    print(f"step {s} loss {loss.item():.4f} n_sup {n} {time.time()-t:.1f}s peak_alloc {torch.cuda.max_memory_allocated()/1e9:.2f}GB peak_reserved {torch.cuda.max_memory_reserved()/1e9:.2f}GB", flush=True)
