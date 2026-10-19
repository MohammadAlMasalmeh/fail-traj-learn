# Sol V8 Distillation 500: download, training contract, and reproduction

Run commands from the repository root. The public dataset is
[andaloo23/sol-v8-distillation-500](https://huggingface.co/datasets/andaloo23/sol-v8-distillation-500).
Release v1 is pinned to Hugging Face commit
`a8590e59f6fa1589de61bea83a7b82a888aed83c`.

## Download and inspect

Python 3.12 is sufficient to download, verify, extract, and inspect the release.
No Hugging Face account or API key is needed for this public download:

```bash
python scripts/annotate/download_distillation_v8.py
```

The helper downloads the pinned commit, checks the 795,088,474-byte archive against
SHA-256 `844b68b5727d5a95412f82a815190aa308141f1f5d8b2cef59c10ac54280ba3e`,
then extracts it under `outputs/hf_sol_v8_distillation_500/`. It refuses to replace
an existing extracted dataset. Use `--out /path/to/new-directory` to select another
destination. The archive stays available for subsequent integrity checks.

```python
import json
from pathlib import Path

root = Path("outputs/hf_sol_v8_distillation_500/sol_v8_distillation_500_export_v1")
with (root / "train.jsonl").open(encoding="utf-8") as stream:
    row = json.loads(next(stream))
target = json.loads(row["messages"][1]["content"])
images = [root / item["image"] for item in row["messages"][0]["content"]
          if item["type"] == "image"]
print(row["id"], len(images), len(target["segments"]))
```

Each JSONL record has `id`, `split`, `messages`, `annotation`, `supervision`, and
`metadata`. User content interleaves prompt/task metadata, sampled telemetry,
agent-camera JPEGs, and wrist-camera JPEGs in the original teacher-input order.
Resolve image paths against the extracted directory, not your current directory.
The assistant content is a JSON string. `annotation` points to its unchanged
parsed JSON counterpart under `annotations/`.

The [dataset card](../datasets/sol_v8_distillation_500/README.md) lists all contents,
counts, review limits, and evaluation reservations. The archive also retains the
exact prompt, original responses, frozen inputs, file hashes, and provenance.
`manifest.json` records per-file hashes; `verification.json` records the original
integrity checks.

For a fresh verification of image hashes, annotations, split disjointness, and
uncertainty masks, use the repository's exporter verification function:

```python
import sys
from pathlib import Path
sys.path.insert(0, "scripts/annotate")
from export_distillation_v8 import verify_export
print(verify_export(Path("outputs/hf_sol_v8_distillation_500/sol_v8_distillation_500_export_v1")))
```

## Using the labels for training

This release provides teacher-generated supervision, not human ground truth.
The user accepted the batch after reviewing 20 random episodes; the remaining
episodes were not independently human-adjudicated. Structural validation does
not establish semantic correctness.

There are 500 training episodes, 44,652 images, and 4,093 segments. Only 1,410
segments are unflagged; 2,683 segments and 260 uncertain identity objects are
masked. Keep all records for provenance. The 121 episodes with no unflagged
segment targets must not become negative training examples merely because their
segments are flagged.

A model-specific multimodal collator must:

1. Load the referenced images and preserve the message order.
2. Apply the model's chat template and tokenizer without rewriting the assistant
   JSON string. Preserve non-ASCII text and its character offsets.
3. Set labels to `-100` for user, image, padding, and other non-target tokens.
4. Also set labels to `-100` for every assistant text token overlapping any
   `supervision.assistant_excluded_char_ranges` interval. These cover whole
   uncertain identity objects and whole `review_required` segments.

Ranges are half-open Unicode character offsets relative to the assistant content
string before chat-template delimiters. They are not UTF-8 byte offsets or token
indices. For a token span `[start, end)` in that same character coordinate system,
overlap with an excluded range `r` means
`start < r["end"] and end > r["start"]`. A token touching both allowed and excluded
text must be masked. Account for template prefixes and image-token expansion when
mapping the real model sequence back to assistant text. Do not apply these
offsets directly to a tokenization of the entire formatted conversation.

`identity_teacher_loss_mask`, per-segment `teacher_loss_mask`,
`has_unflagged_segment_targets`, and `human_verified` provide additional audit
information. The character ranges are the supplied text-loss exclusion contract.
Unflagged teacher targets are not certified positive or negative human labels.

There is no universal collator or model-training command in this release. The
existing failure-representation learner consumes a different annotation format;
these JSONL records require an explicit adapter. A generic SFT trainer that
ignores the masks does not satisfy this dataset's training contract.

`splits.json` reserves 50 development and 50 test episode identities, but includes
no examples or labels for those reservations. They are episode-disjoint within
the same collecting policy, not task-held-out or cross-policy test sets. Sampling
balanced recorded episode outcomes; it does not represent natural failure
frequencies or verify local failure labels.

## Source files and release provenance

| Stage | Source |
| --- | --- |
| Select 500 train / 50 development / 50 test identities | `scripts/annotate/prepare_distillation_v8.py` |
| Decode original cameras and observable telemetry | `scripts/annotate/common.py`, `scripts/annotate/bench/review_oracle.py` (`decode_episode` only) |
| Sampling corrections | `scripts/analysis/exclusions.json` |
| Freeze sampled inputs, prompt, and validate labels | `scripts/annotate/annotate_trajectories.py`, `annotation_runtime.py`, `annotation_states.py` |
| Frozen instruction and output schema | `scripts/annotate/prompts/atomic_actions/annotation.txt` |
| Prepare, submit, reconcile, collect Batch API jobs | `scripts/annotate/batch_trajectories.py` |
| Serialize queue submission under token limits | `scripts/annotate/queue_batch_trajectories.py` |
| Preserve failures and retry unsuccessful episodes | `scripts/annotate/resume_distillation_v8.py` |
| Random review pack and browser template | `scripts/annotate/review_distillation_sample.py`, `review_trajectory_annotations.html` |
| Preserve labels, construct masks, verify and archive | `scripts/annotate/export_distillation_v8.py` |
| Publish archive/card/checksum and verify public readback | `scripts/annotate/publish_distillation_v8.py` |

The [release directory](../datasets/sol_v8_distillation_500/) preserves the original
selection, random-review sample, subsequent approval, verification summary, and
dataset card. `release.json` records the immutable Hub revision, archive hash,
counts, original settings, and frozen code hashes. Paths inside historical JSON
snapshots refer to the original build machine; they are provenance, not portable
commands. `review_sample.json` was written before approval, so its pending status
must be read alongside the later `review_approval.json`.

The frozen Python fingerprints include original line endings. Scoped
`.gitattributes` rules preserve those source and snapshot bytes across clones;
normalizing them would invalidate the old run manifests even if Python behavior
stayed the same.

The teacher used `gpt-6.1-sol`, low reasoning, high image detail, every fifth frame
plus the final frame, and maximum output 12,288 tokens. The historical primary
run had 50 shards of 10 requests, validated 496 episodes, and retained four stopped
outputs. A separate recovery run validated those four episodes. The merged
generation estimate was $17.8752715. These saved prices and costs are historical
provenance, not current price quotes or a budget guarantee.

Git holds source and small provenance snapshots. The approved ZIP, labels, images,
and frozen teacher inputs are distributed through the pinned Hugging Face release.
Original full-rate recordings, LeRobot/video environments, earlier exclusion
packs, and paid API job journals remain external. They are not needed to consume
the release. Byte-identical download is supported; a fresh teacher generation may
produce different labels, timestamps, archive bytes, and costs.

## Rebuilding from the original recordings

Use the original Linux/WSL LeRobot environment with its compatible PyTorch, video
decoder, NumPy, and Pillow dependencies. The local annotation environment only
needs `pip install -r scripts/annotate/requirements-current-annotation.txt`.
`common.py` loads recorded datasets from `$FTL_PROJ/data`; set `FTL_PROJ` before
source export. Raw `full_*` datasets must contain `episodes.jsonl`, `meta/`, video
files, and `sidecar/` metadata. Their identities and inventory hashes are in the
frozen selection. No new simulation is necessary.

Use a new work directory. To retain the published identity selection while
relocating the data root, copy the snapshot and update only its machine path:

```bash
export FTL_PROJ=/path/to/original/project
export DISTILLATION_WORK="$PWD/outputs/sol_v8_rebuild"
mkdir -p "$DISTILLATION_WORK"
python - <<'PY'
import json, os
from pathlib import Path
p = Path("datasets/sol_v8_distillation_500/selection.json")
selection = json.loads(p.read_text())
selection["data"] = str((Path(os.environ["FTL_PROJ"]) / "data").resolve())
out = Path(os.environ["DISTILLATION_WORK"]) / "selection.json"
with out.open("x") as stream:
    json.dump(selection, stream, indent=2)
    stream.write("\n")
PY
python scripts/annotate/prepare_distillation_v8.py export \
  --data "$FTL_PROJ/data" --out "$DISTILLATION_WORK" --workers 4
```

This creates a derivative selection hash because its machine path changed. The
committed original stays unchanged. To select a different population instead,
run `prepare_distillation_v8.py select` in a fresh directory with `--count 500
--seed 20261015`. Selection also depends on the current exclusion packs and raw
inventory; the seed alone cannot reconstruct the historical selection.

Prepare frozen requests offline:

```bash
python scripts/annotate/batch_trajectories.py prepare \
  --source "$DISTILLATION_WORK/source" \
  --selection "$DISTILLATION_WORK/selection.json" --out "$DISTILLATION_WORK/run"
python scripts/annotate/batch_trajectories.py verify --out "$DISTILLATION_WORK/run"
```

The CLI defaults to at most 50 requests per shard. To match the historical
10-request shard layout, call `freeze_shards(out, max_requests=10)` when preparing
a new run, after `annotate_trajectories.prepare()` and copying its selection;
never re-shard a submitted run. Shard size does not change the per-episode request.

The following command submits paid work. Set `OPENAI_API_KEY` in your environment
or ignored `.env`, review the frozen requests and current model availability and
costs, and explicitly choose to launch it:

```bash
python scripts/annotate/queue_batch_trajectories.py --out "$DISTILLATION_WORK/run"
```

The queue preserves unresolved submissions and does not blindly resend them.
`batch_trajectories.py reconcile --out ...` and `collect --out ...` retrieve remote
state without submitting new generation. After inspecting unsuccessful records,
the historical recovery pattern was:

```bash
python scripts/annotate/resume_distillation_v8.py prepare \
  --main "$DISTILLATION_WORK/run" --retry "$DISTILLATION_WORK/retry"
python scripts/annotate/resume_distillation_v8.py run \
  --main "$DISTILLATION_WORK/run" --retry "$DISTILLATION_WORK/retry"
```

Recovery submits new paid attempts for unsuccessful episodes and writes
`completion.json` at the work root. It preserves prior answers and rejection
journals. The helper was written for the original billing-recovery run; inspect
its checks rather than clearing journals or locks to force a retry.

If all 500 primary episodes validate and no recovery is necessary, create the
completion index directly from that single run:

```bash
python - <<'PY'
import os, sys
from pathlib import Path
sys.path.insert(0, "scripts/annotate")
import batch_trajectories as batch
work = Path(os.environ["DISTILLATION_WORK"]).resolve()
run = work / "run"
summary = batch.io.load(run / "batch_summary.json")
if summary["validated"] != 500 or summary["stopped"]:
    raise ValueError("Require 500 validated primary outputs")
manifest = batch.io.load(run / "manifest.json")
rows = []
for ep in manifest["episodes"]:
    path = run / (ep["id"] + "_sol_annotation.json")
    rows.append(dict(episode=ep["id"], annotation=str(path),
                     sha256=batch.io.digest(path), source_run=str(run),
                     quality="schema_validated_teacher_proposal"))
batch.io.write_json(work / "completion.json", dict(planned=500, validated=500,
    missing=[], generation_estimate_usd=summary["generation_estimate_usd"], episodes=rows))
PY
```

Build the random review pack, serve it locally, and inspect it before export:

```bash
python scripts/annotate/review_distillation_sample.py \
  --root "$DISTILLATION_WORK" --out "$DISTILLATION_WORK/random20_review" \
  --count 20 --seed 7845393249386976415
python -m http.server 8787 --bind 127.0.0.1 --directory "$DISTILLATION_WORK/random20_review"
```

Open `http://127.0.0.1:8787`. The exporter records the original project's approval
statement and does not implement an interactive approval gate. Its approval text
is historical: a new dataset requires its own real review and accurate approval
record, not an assumed reuse of the original user's statement.

For the original approved artifacts, export and archive with:

```bash
python scripts/annotate/export_distillation_v8.py --root "$DISTILLATION_WORK" \
  --out "$DISTILLATION_WORK/export_v1" \
  --archive "$DISTILLATION_WORK/sol_v8_distillation_500_export_v1.zip"
```

Both destinations must be new. The exporter verifies source hashes, partition
coverage, label schema, uncertainty masks, all supplied images, and ZIP CRCs.
Do not edit frozen answers to obtain a passing export. To reconstruct the exact
published artifact, download the pinned release instead of regenerating it.

## Publishing and checking the source

`publish_distillation_v8.py` is the publication script used for this release,
with its local paths made configurable for a clone. It uploads only the ZIP,
original `.zip.json` receipt, and committed dataset card to the fixed public
repository. It refuses a conflicting existing archive and verifies anonymous
metadata, size, remote SHA-256, card, and checksum readback. It requires the
`andaloo23` account with write access:

```bash
python -m pip install -r scripts/annotate/requirements-distillation-publish.txt
hf auth login
python scripts/annotate/publish_distillation_v8.py --root /path/to/approved/archive-directory
```

Authenticate locally; never commit tokens or paste them into scripts. Downloading
the release is independent of this maintainer-only publication step.

Run the focused offline tests without spending API credits:

```bash
python -m unittest discover -s scripts/annotate -p 'test_batch_trajectories.py' -v
python -m unittest discover -s scripts/annotate -p 'test_queue_batch_trajectories.py' -v
python -m unittest discover -s scripts/annotate -p 'test_resume_distillation_v8.py' -v
python -m unittest discover -s scripts/annotate -p 'test_export_distillation_v8.py' -v
python -m unittest discover -s scripts/annotate -p 'test_download_distillation_v8.py' -v
```
