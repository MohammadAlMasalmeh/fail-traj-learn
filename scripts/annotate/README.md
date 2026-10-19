# Current trajectory annotation tools

The working baseline is **V8 atomic actions with failure/non-failure labels**.
Sol and Astra receive the same frozen prompt, images, timestamps, and ordinary
telemetry. The latest saved batches cover R001-R012; generated evidence stays
local under ignored `outputs/` directories.

Start with the [V8 runbook and next steps](../../docs/annotation_runbook_v8.md).
It contains environment setup, exact prepare/run/review commands, handling failed
calls, combining batches, fixed scoring, and the R005 review priorities.

The published **500-episode Sol V8 distillation dataset** has a separate
[download, training, and reproduction guide](../../docs/sol_v8_distillation_500.md).
It covers `prepare_distillation_v8.py`, `batch_trajectories.py`,
`queue_batch_trajectories.py`, `resume_distillation_v8.py`,
`review_distillation_sample.py`, `export_distillation_v8.py`,
`download_distillation_v8.py`, and `publish_distillation_v8.py`.

## Supported entry points

| Tool | Purpose |
|---|---|
| `annotate_trajectories.py` | Prepare, verify, execute, and report frozen atomic annotation runs |
| `prompts/atomic_actions/annotation.txt` | Current shared V8 rules and JSON output contract |
| `prompts/atomic_actions/comparison.json` | Sol/Astra comparison settings and independence rules |
| `review_trajectory_annotations.py` | Build a browser review from validated saved outputs and original images |
| `compare_trajectory_annotations.py` | Compute offline binary/category and temporal agreement |
| `compare_action_groups.py` | Compare temporally aligned atomic operations and category assignments |
| `combine_review_batches.py` | Combine complete disjoint same-protocol reviews with per-episode provenance |
| `score_review_text.py` | Score raw descriptions with the pinned STS baseline and frame-weighted means |

## Protocol

- Each segment has an action, physical result, failure boolean, categories,
  evidence, and uncertainty. Progress and neutral behavior are non-failure.
- Failure timing follows observable failing behavior rather than causal
  responsibility or guessed hidden onset.
- Supported wrong-object engagement can fail semantically before acquisition.
  Physical success and task correctness are separate judgments.
- Collision includes supported unwanted contact that disturbs a destination.
  Direct visibility of the exact contacting part is not always required.
- Use canonical action verbs, stable instruction-aligned names for supported
  matches, and concise operation/result/failure descriptions. Preserve genuine
  identity differences, negation, and uncertainty.
- Separation or destination entry does not establish final support or settling.
  Stop claims at the last supported observation and flag unverified outcomes.

The prompt contains general rules rather than episode-specific required labels.
Disagreement diagnoses uncertain evidence, model mistakes, or protocol gaps; it
is not an objective to optimize to 100% agreement. Schema validation alone does
not establish visual accuracy.

## Scoring and artifacts

The current text scorer uses raw bidirectional STS predictions, without an
identical-text override or calibration. It compares only `what_happened`; the
scores are similarity indices rather than percentages of semantic agreement.
Model weights and the tested dependency stack are pinned. Separate binary,
category, and action metrics retain differences that text similarity can miss.

Every paid run keeps its original manifest, prompt, input bytes, request attempts,
raw answers, annotations, and cost estimates. Existing reservations are never
resent automatically. Same-protocol supplements can fill missing outputs, but
cannot replace validated ones. Combined viewers are derived review artifacts,
not additional paid runs. Keep original source directories when sharing a viewer.

Older state-based and description-judge workflows remain historical comparators.
They are not the current human-label schema or independent evaluation protocol.
See the runbook before collecting human labels or exporting training supervision.
