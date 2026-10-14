# Frozen V8 annotation workflow and next steps

Updated: 2026-10-14. Run commands from the repository root.

## Current state

The working protocol is [the shared atomic-action prompt](../scripts/annotate/prompts/atomic_actions/annotation.txt).
Its frozen V8 SHA-256 is
`e7ed5aae4443ce1ee721bacf6746c9abef4b1c2d11f18039414d5a00811ce20e`.
The saved prompt bytes, inputs, settings, raw answers, validation records, and costs
belong to each source run. Do not edit those artifacts to correct a prediction.

| Local artifact | Contents |
|---|---|
| `outputs/atomic_observed_end_states_v8` | R001-R004: eight validated Sol/Astra annotations |
| `outputs/atomic_observed_end_states_v8_r005_r012` | R005-R012: sixteen validated annotations under exactly the same prompt |
| `outputs/atomic_observed_end_states_v8_all_r/current_review` | Combined R001-R012 viewer, provenance, and agreement reports |

All 24 annotations passed schema/coverage validation. This does not establish
visual correctness or compliance with every semantic rule. The combined viewer
covers 3,071 source frames. Its saved mean raw text similarity is about 0.880;
R005-R012 alone average about 0.866. Neither number is accuracy or a calibrated
percentage of acceptable annotations.

R001-R004 were repeatedly inspected during prompt development. R005-R012 were
annotated after freezing V8; earlier project exposure has not been audited, so do
not describe them as a proven unseen test set. The prompt remains unchanged after
examining R005. No fresh API calls are needed to reopen the saved viewer.

Generated outputs, the review pack, credentials, virtual environments, and cached
weights are excluded from Git. A fresh clone needs the review pack from
[REVIEW.md](../REVIEW.md) or an equivalent observable source pack. Pushing the
scripts does not upload the saved annotations or images; preserve those separately.

## Rules to keep fixed

- One continuous operation on the same participants within an attempt defines an
  atomic action. Changes of operation, participants, or distinct attempt start a
  new action. Local failure/category transitions split segments while retaining
  the continuing action's `action_id`.
- Segments partition `[0, n_frames)` using half-open intervals. Each segment has
  one `failure` boolean. Progress, neutral behavior, and ordinary pauses are all
  non-failure; non-failure does not imply task completion.
- Localize failure when supplied observations support it. Do not backdate it to
  the responsible decision, command, or hidden physical onset. Keep uncertainty
  brackets between sampled observations explicit.
- Separate operation, physical result, task correctness, and uncertainty. An
  achieved physical result can coexist with semantic failure. Supported
  wrong-object engagement does not require successful acquisition.
- Collision means supported unwanted contact, including contact that displaces,
  lifts, tilts, drags, or obstructs the destination. A supported disturbance in a
  stable view can establish contact without revealing the exact contacting part.
  Ambiguous motion alone is insufficient. Unresolved claims remain review flags.
- Use instruction terms for supported matching referents; actual supported names
  for wrong objects; generic IDs for unresolved identities. Keep IDs stable and
  match referents across annotations through evidence, not shared names alone.
- Write `what_happened` as operation and participants, observed physical result,
  then supported failure effect or material outcome limit. Use canonical verbs
  and result phrases. Do not standardize away negation, uncertainty, or differences.
- Separation, destination entry, surface contact, and retained support/settling
  require separate evidence. Do not infer an unobserved final state. Confirmed
  separation can remain affirmative while final settling is unverified.
- Use general rules rather than episode-specific examples or required outputs.
  Sol/Astra disagreement is a diagnostic, not an objective to drive to 100%.

## Environment and credentials

The following commands use Windows PowerShell and Python 3.12. On Linux/macOS,
use the corresponding environment's `bin/python` instead of `Scripts/python.exe`.
Reuse existing environments when available.

```powershell
py -3.12 -m venv .venv-review
.venv-review/Scripts/python.exe -m pip install -r scripts/annotate/requirements-current-annotation.txt

py -3.12 -m venv .venv-eval
.venv-eval/Scripts/python.exe -m pip install -r scripts/annotate/text_similarity_requirements.txt
```

Set `OPENAI_API_KEY` in the process environment or as a named entry in the ignored
`.env` file. Never put credentials in a prompt, command example, or tracked file.
The annotation runner uses the standard-library HTTP client; no OpenAI SDK is
required. The local text scorer needs Torch and Sentence Transformers, not API
credits. CPU inference is supported and used by this workflow.

## Annotate a fresh batch

Each source episode must contain `observable.json` and `cameras.npz`. Metadata
includes `id`, `task`, `n_frames`, `fps`, and frame-indexed telemetry; camera arrays
are synchronized RGB `agent` and `wrist` images. Model telemetry is limited to
`eef_xyz_m`, `finger_qpos_m`, and `gripper_command`. Candidate labels, human labels,
and privileged simulator contacts are not model inputs.

Use a **new** output directory. The example below regenerates R005-R012 and incurs
new API charges; the already completed batch above does not need to be rerun.

```powershell
.venv-review/Scripts/python.exe scripts/annotate/annotate_trajectories.py prepare --scheme atomic --source gpt6_v3_12_episode_review_pack/episodes --episodes R005 R006 R007 R008 R009 R010 R011 R012 --models sol astra --reasoning low --out outputs/my_new_atomic_batch
.venv-review/Scripts/python.exe scripts/annotate/annotate_trajectories.py verify --out outputs/my_new_atomic_batch
```

Preparation is offline. Inspect `manifest.json`, `prompt.txt`, episode inputs,
planned call count, and reservations before running. V8 uses Sol `gpt-6.1-sol`,
Astra `gpt-6-astra`, low reasoning, high image detail, every fifth source frame
plus the last frame, two workers, and no automatic retries. The source pack uses
20 fps; the runner honors each episode's actual metadata. Both models receive
identical frozen evidence and the same shared prompt. `--models sol` supports a
Sol-only diagnostic; do not silently compare it with a different-prompt baseline.

`run` spends API credits. Pricing/reservations in the runner are dated estimates,
not a live account balance. Check current prices before a new paid batch. A
maximum reservation is a bound, not a charge or an expected cost.

```powershell
.venv-review/Scripts/python.exe scripts/annotate/annotate_trajectories.py run --out outputs/my_new_atomic_batch
.venv-review/Scripts/python.exe scripts/annotate/annotate_trajectories.py report --out outputs/my_new_atomic_batch
```

Check that `summary.json` has `validated == planned` and inspect failures. Each
attempt has its own reservation, raw response, answer text, validated annotation,
and token/cost record. Calling `run` again does not resend existing reservations.

`verify` checks the frozen runner code as well as prompt/settings/input bytes.
It can reject an old run after a checkout or code change, including line-ending
changes. Do not rewrite its hashes to force acceptance. Use the originating code
and bytes to execute a frozen run, or prepare a new directory. The viewer can
inspect older saved outputs after independently validating their frozen inputs.

## Failed or interrupted calls

1. Read the corresponding `attempts/<episode>_<model>.json`. HTTP failures retain
   sanitized status, error code/type, request ID, and available routing headers.
   An interrupted request may have incurred an unknown charge; do not erase it.
2. Resolve the actual transport, billing, or validation issue. Do not alter a
   successful annotation to hide a failed neighboring request.
3. If a fresh attempt is needed, prepare a separate single-model, episode-specific
   run with identical prompt, inputs, and settings. Compare bytes/settings with
   the parent before sending the request. If runner code has changed, use the
   parent's code version rather than modifying its frozen manifest.
4. A validated same-protocol retry can fill a missing viewer cell:

```powershell
.venv-review/Scripts/python.exe scripts/annotate/review_trajectory_annotations.py --run outputs/my_new_atomic_batch --source gpt6_v3_12_episode_review_pack/episodes --supplement-run outputs/my_single_episode_retry --prepare-only
```

Supplements cannot replace already validated outputs. Original failures remain
saved. A semantic disagreement is not a transport failure and does not justify
sampling until a preferred answer appears.

## Build, score, and open a viewer

For a saved annotation run, these operations are local:

```powershell
.venv-review/Scripts/python.exe scripts/annotate/review_trajectory_annotations.py --run outputs/my_new_atomic_batch --source gpt6_v3_12_episode_review_pack/episodes --prepare-only
.venv-eval/Scripts/python.exe scripts/annotate/score_review_text.py outputs/my_new_atomic_batch/current_review
.venv-review/Scripts/python.exe -m http.server 8774 --bind 127.0.0.1 --directory outputs/my_new_atomic_batch/current_review
```

Open `http://127.0.0.1:8774/`. The first scoring call can download the pinned model
weights. Add `--local-files-only` after caching them. Use `--summarize-only` to
reaggregate saved pair scores without loading a model. Stop the server with Ctrl+C.

The viewer compares failure/non-failure and categories, provides structured
action/category agreement, and displays raw description similarity and frame-
weighted means. `what_happened` appears above the failure label. Use model-supplied
frames to check what each model could observe; full playback supplies additional
review evidence. Download browser review notes to retain them durably.

To build a single viewer from complete disjoint batches under the same protocol:

```powershell
.venv-review/Scripts/python.exe scripts/annotate/combine_review_batches.py --reviews outputs/atomic_observed_end_states_v8/current_review outputs/atomic_observed_end_states_v8_r005_r012/current_review --out outputs/my_combined_v8/current_review --label "V8 shared prompt / R001-R012"
.venv-eval/Scripts/python.exe scripts/annotate/score_review_text.py outputs/my_combined_v8/current_review --local-files-only
.venv-review/Scripts/python.exe -m http.server 8774 --bind 127.0.0.1 --directory outputs/my_combined_v8/current_review
```

The target must be new. The combiner checks source manifests, prompt/settings,
saved annotations, raw answers, and duplicate episodes. It copies review assets,
retains per-episode provenance, and recomputes action/category and binary reports.
`review_manifest.json` describes this derived review; it is not a new paid run.
Text scores must be computed after combining because the comparison fingerprint
and episode set change. Rebuilding a source viewer can replace its generated
labels/variants; keep originals and rebuild with the required supplement options.

To reopen the already completed combined viewer without rebuilding or API calls:

```powershell
.venv-review/Scripts/python.exe -m http.server 8773 --bind 127.0.0.1 --directory outputs/atomic_observed_end_states_v8_all_r/current_review
```

Open `http://127.0.0.1:8773/?episode=R005&frame=0`. If that port already serves this
viewer, simply open it; otherwise choose a free port. Server processes are local
and are not published or kept alive by Git.

## Fixed scorer and interpretation

- Raw STS model: `cross-encoder/stsb-distilroberta-base`, revision
  `6b71347df6e2b34246b53e06d6bce70ef67de368`. The scorer pins the revision used in
  the completed V8 batch. Dependency versions and model revision are recorded in
  `text_similarity.json`; use `text_similarity_requirements.txt` for the tested stack.
- Score only `what_happened`, average both text input orders, then weight each
  active pair by covered source frames. This text scorer does not directly read
  video, task instructions, categories, `observed_result`, or uncertainty fields.
- Show raw predictions, including approximately 0.99 for identical text. The
  hardcoded identical-text override was rejected and removed. There is no score
  calibration or forced perfect match. Missing scores are not zero; an all-frame
  mean requires complete score coverage.
- Binary/category agreement and ordered action/category Dice measure different
  properties. Structured Dice compares temporally aligned action operations and
  category assignments, not full prose, referent identity, physical results, or
  video correctness. Harmless segmentation differences should be inspected through
  grouped actions rather than judged by segment count alone.
- Keep the same scorer, fields, weights, and aggregation for human-human and
  human-VLM comparisons. The score is a development index; a fixed scorer can
  still have systematic style or contradiction errors. Do not call it accuracy
  or use it alone to establish a human ambiguity floor.
- Independent Gemini judges, NLI, PAWS, embeddings, and larger STS models were
  explored. None of those experiments replaced this baseline. More score bins or
  repeated LLM sampling alone do not establish calibration or validity. The old
  Astra-based description judge is a historical experiment, not the current
  independent evaluator; an unscored LLM panel is expected in the V8 viewer.

## What to do next

1. **Audit disagreements in R005-R012 before changing the prompt.** Log model,
   action/segment, supplied frame(s), camera, supported facts, disputed category,
   and relevant rule. Distinguish uncertain evidence, a model mistake, and an
   actual protocol gap. Preserve defensible alternatives and uncertain boundaries.
2. **Start with R005.** Astra marks collision at frames 140-160 and failed outside-
   basket placement at 160-165. Sol omits those labels despite describing failed
   placement, reasoning that confirmation arrives when release ends. Sol does
   mark later grasp failures at 220-225 and 279-280; Astra marks them at 215-220
   and 275-280. Review whether an overly rigid action boundary suppresses an
   observable failure. Also separate physical release from destination correctness
   and evaluate stable-view evidence of unwanted contact. This is an open review
   issue, not an independently adjudicated reference or a reason to force agreement.
3. **Keep V8 as a recorded baseline.** If review justifies a general rule change,
   create a separately versioned prompt/run and record the reason. Do not insert
   required R005 labels, rewrite old outputs, or tune against aggregate agreement.
4. **Prepare independent human labels.** Use the V8 JSON contract, or first build
   and validate an adapter for the existing human editor's legacy exclusive
   progress/failure/recovery/neutral schema. It is not a drop-in binary V8 editor.
   Have at least two people annotate independently without seeded VLM predictions.
   Use the same task/evidence conditions, document whether full or sampled video
   is shown, and keep privileged-contact-assisted review separate from vision-only
   labels. Preserve uncertainty and incomplete coverage instead of manufacturing
   verified negatives.
5. **Compare human-human and human-VLM agreement.** Validate IDs/coverage and match
   referents before scoring. Apply the same fixed metrics and report frame,
   action/category, and text results separately. Adjudicate a small set of genuine
   semantic differences to check that the scorer treats human and VLM wording
   comparably. Use fresh episodes for claims beyond these development examples.
6. **Measure variability only if needed.** Predeclare sample count and generation
   settings; save every sample. Separate annotation variability from judge
   variability. Report score spread and qualitative errors without selecting the
   best-matching samples. Stable scores can still be systematically wrong.
7. **Export training supervision after review.** Keep episode-disjoint splits,
   original model predictions, reviewer provenance, and explicit uncertainty or
   exclusion masks. Audit the training manifest before using these labels as
   accepted supervision. Unreviewed Sol/Astra agreement is not ground truth.

## Offline verification

These checks use fixtures and mocked API responses; they do not spend API credits:

```powershell
.venv-review/Scripts/python.exe -m unittest discover -s scripts/annotate -p test_annotate_trajectories.py
.venv-review/Scripts/python.exe -m unittest discover -s scripts/annotate -p test_annotation_workflow.py
.venv-review/Scripts/python.exe -m unittest discover -s scripts/annotate -p test_trajectory_metrics.py
.venv-review/Scripts/python.exe -m unittest discover -s scripts/annotate -p test_compare_action_groups.py
.venv-review/Scripts/python.exe -m unittest discover -s scripts/annotate -p test_score_review_text.py
.venv-review/Scripts/python.exe -m unittest discover -s scripts/annotate -p test_combine_review_batches.py
```

For a generated viewer, additionally check each episode's first/middle/final
images, both model-supplied frame modes, raw answer links, displayed annotations
against source JSON, the exact prompt hash, and score coverage. The current local
combined viewer was checked against all 24 source outputs and sampled image bytes.
