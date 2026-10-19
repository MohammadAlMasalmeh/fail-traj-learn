---
language:
- en
size_categories:
- n<1K
tags:
- robotics
- multimodal
- distillation
- synthetic
- failure-analysis
pretty_name: Sol V8 Distillation 500
---

# Sol V8 Distillation 500

An export of 500 robot trajectory episodes with paired agent/wrist camera images,
task text, timestamps, sampled robot telemetry, and teacher-generated V8 failure
annotations. The teacher model is `gpt-6.1-sol`.

This is sample-approved teacher supervision, not human ground truth. Export was
approved after review of 20 randomly selected episodes; that review was batch-level
acceptance, not independent adjudication of all 500 episodes.

## Download

Download `sol_v8_distillation_500_export_v1.zip` from the Files tab or use:

```python
from huggingface_hub import hf_hub_download

archive = hf_hub_download(
    repo_id="andaloo23/sol-v8-distillation-500",
    filename="sol_v8_distillation_500_export_v1.zip",
    repo_type="dataset",
)
```

Extract the archive to obtain `sol_v8_distillation_500_export_v1/`. Image paths
in `train.jsonl` are relative to that directory. This release preserves the
approved archive and custom training format; use a training adapter that honors
the supervision masks described below.

## Contents

| Item | Count |
| --- | ---: |
| Training episodes | 500 |
| JPEG images | 44,652 |
| Annotated segments | 4,093 |
| Unflagged segments eligible for teacher supervision | 1,410 |
| Masked segments | 2,683 |
| Masked identities | 260 |
| Episodes without unflagged segment targets | 121 |

The archive contains:

- `train.jsonl`: one multimodal training record per episode, with `id`, `split`,
  `messages`, `annotation`, `supervision`, and `metadata` fields.
- `images/`: exact JPEGs supplied to the teacher.
- `annotations/` and `responses/`: original parsed answers and response text.
- `inputs/`: frozen episode metadata.
- `prompt.txt`, `assets.json`, `manifest.json`, and `provenance.json`: prompt,
  asset metadata, hashes, and annotation provenance.
- `review_approval.json`, `splits.json`, `verification.json`, and `README.txt`:
  review scope, split reservations, integrity checks, and the export contract.

The categories are collision, grasp, manipulation, other, reaching,
sequencing_semantic, sudden_episode_termination, task_not_attempted, and
visible_hardware_issue.

## Required training masks

A model-specific adapter must mask user, image, and padding tokens, and every
assistant token overlapping an `assistant_excluded_char_ranges` interval.
Intervals use Unicode character offsets relative to the assistant content string
before adding model-specific chat-template delimiters. Mask uncertain identity
objects and whole `review_required` segments.

Unflagged segment masks identify eligible teacher supervision; they do not certify
human-verified positives or negatives. Retain the 121 records without unflagged
segment targets for provenance and review, rather than treating flagged segments
as negatives. A generic trainer that ignores these custom masks does not satisfy
the export's training contract. Training raw teacher outputs with flags unmasked
would be a separate, explicit ablation.

## Evaluation scope and quality

`splits.json` reserves 50 development and 50 test episode identities. Their
examples and labels are not included in this training export. These are
episode-disjoint reservations within the same collecting policy, not task-held-out
or cross-policy evaluation splits.

Export verification reports valid annotations and uncertainty masks, unique
episode IDs, disjoint split identities, and resolved images matching recorded
hashes. These integrity checks do not establish annotation correctness. No
annotation has been promoted to human ground truth.

## Archive integrity

- Bytes: `795088474`
- SHA-256: `844b68b5727d5a95412f82a815190aa308141f1f5d8b2cef59c10ac54280ba3e`
- ZIP CRC verification: passed.

The original checksum receipt is included as
`sol_v8_distillation_500_export_v1.zip.json`.
