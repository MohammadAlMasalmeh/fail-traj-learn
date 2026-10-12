# Current trajectory annotation tools

The retained workflow uses one shared prompt and identical observable inputs for
Sol and Astra. Choose atomic actions (the latest development protocol) or maximal
contiguous progress/failure/category states (the V15 protocol). The two formats
answer different questions: action boundaries versus task-label transitions.
Both prompts preserve independent progress and failure labels.

These entry points prepare evidence directly from recordings. They do not require
the historical V1–V15 experiment runners. Old frozen runs and research artifacts
remain local; new runs use new directories and do not rewrite that evidence.

## Setup and input

Use Python 3.12 and install `requirements-current-annotation.txt` in a virtual
environment. Each source episode directory contains `observable.json` with
`id`, `task`, `n_frames`, `fps`, and per-frame `telemetry`, plus `cameras.npz`
with synchronized RGB `agent` and `wrist` arrays. Telemetry inputs are limited to
`eef_xyz_m`, `finger_qpos_m`, and `gripper_command`. Sample every five source
frames and include the final frame.

Set `OPENAI_API_KEY` in the environment or a named entry in the ignored `.env`.
Only OpenAI credentials are needed for these current runners. API transport has
no automatic retries. Stored pricing is a dated development estimate; check it
before using the printed reservation for a new paid batch.

## Prepare and inspect a new run

```powershell
python scripts/annotate/annotate_trajectories.py prepare --scheme atomic --source gpt6_v3_12_episode_review_pack/episodes --episodes R001 R002 R003 R004 --out outputs/my_atomic_run
python scripts/annotate/annotate_trajectories.py verify --out outputs/my_atomic_run
```

Preparation is offline and prints the episode matrix, call count, manifest hash,
and maximum reservation. Use `--scheme states` and a different output directory
for state-based segmentation. Models receive the same images, telemetry, prompt,
and request settings. Scheme, code, prompt, image bytes, and settings are frozen.

When ready to spend API credits, run the prepared batch:

```powershell
python scripts/annotate/annotate_trajectories.py run --out outputs/my_atomic_run
python scripts/annotate/annotate_trajectories.py report --out outputs/my_atomic_run
```

Existing attempt reservations are never resent. Interrupted, invalid, or incomplete
responses remain visible in the ledger. A changed experiment requires a new run;
do not remove attempt files to retry uncertain billed calls.

## Review and compare saved outputs

```powershell
python scripts/annotate/review_trajectory_annotations.py --run outputs/my_atomic_run --source gpt6_v3_12_episode_review_pack/episodes
python scripts/annotate/compare_trajectory_annotations.py --run outputs/my_atomic_run --out outputs/my_atomic_run/agreement.json
```

The viewer checks source and frozen input hashes, displays exact sampled evidence
and full-frame playback, and shows annotations and missing/invalid outputs. Review
notes stay in browser storage until downloaded. It makes no API calls. Existing
`outputs/atomic_actions` can also be reviewed and compared with these commands.

The offline report separates progress/failure agreement, failure categories,
phase boundaries, binary-state boundaries, segment overlap, and review flags.
Unavailable outputs remain explicit. Undefined failure F1 is null when neither
side labels failures. Agreement is not annotation accuracy.

For the optional paid description judge on a new atomic-action run:

```powershell
python scripts/annotate/compare_annotation_descriptions.py prepare --annotations outputs/my_atomic_run --out outputs/my_description_comparison
python scripts/annotate/compare_annotation_descriptions.py run --annotations outputs/my_atomic_run --out outputs/my_description_comparison
```

It compares temporally intersecting descriptions in both input orders with frozen
paraphrase, negation, identity, attempt/result, and task-progress controls. Scores
measure textual agreement, not video correctness. Astra is also an annotator, so
order balancing does not eliminate judge bias. The comparison supports atomic
annotations; it is not a reference-based quality evaluator.

## Verification

```powershell
python -m unittest discover -s scripts/annotate -p test_annotate_trajectories.py
python -m unittest discover -s scripts/annotate -p test_annotation_workflow.py
python -m unittest discover -s scripts/annotate -p test_compare_annotation_descriptions.py
python -m unittest discover -s scripts/annotate -p test_trajectory_metrics.py
```

## Next experiment

Use atomic actions as the working candidate, retain states as a comparator, and
freeze the prompt before adding new evaluation episodes. The current four R
episodes have already been inspected repeatedly and are development data.

1. Independently annotate a small new evaluation set using the chosen format.
   Resolve object identity, brief failures, boundaries, and uncertain coverage.
2. Run the frozen annotation protocol on that set and evaluate against adjudicated
   human references. Implement the reference adapter for this exact schema first;
   older exclusive-label H references do not validate four-state labels.
3. Decide whether the protocol has enough usable positive and reviewed negative
   coverage to expand annotation. Compare reliability and cost alongside quality.
4. Export a reviewed, episode-disjoint training manifest with unknown masks, then
   compare the causal temporal learner with a chunk MLP on the same split.

High inter-model agreement and passing description controls alone do not justify
bulk annotation or a claim that the resulting supervision is correct.
