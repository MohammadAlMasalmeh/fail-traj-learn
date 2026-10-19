# Script entry points

Run from the repository root using the external LeRobot `.venv/bin/python`. See the
[main README](../README.md) for environment variables and the [documentation index](../docs/README.md)
for experiment status. This index lists entry points; imported helpers and benchmark methods remain
in their existing locations so historical runs and tests stay reproducible.

Current portable annotation commands and retained prompts are documented in
[the annotation guide](annotate/README.md).

| Task | Entry point | Notes |
|---|---|---|
| Download the Sol V8 distillation dataset | `annotate/download_distillation_v8.py` | Pinned public release with size/SHA-256 verification; [training and reproduction guide](../docs/sol_v8_distillation_500.md) |
| Audit failure-training data | `analysis/audit_failure_corpus.py` | Read-only corpus checks; only writes the requested report |
| Compare references against current oracle | `analysis/recompute_failure_references.py` | Recomputes from telemetry in memory; does not replace references |
| Verify simulator snapshots | `analysis/check_snapshot_restore.py` | Restore/replay spot checks, with exact/approximate modes |
| Review episode evidence | `analysis/review_failures.py`, `analysis/build_review_index.py` | Generate camera/telemetry reports |
| Check collection outcomes/inits | `analysis/summarize_datasets.py`, `analysis/predicate_artifacts.py`, `analysis/init_state_check.py` | Recorded outcomes, geometric artifacts and initial-state diagnostics |
| Train failure representations | `rl/train_failure_model.py` | `--audit-only` checks manifest coverage first |
| Test RL and failure modules | `rl/run_tests.sh` | CPU unittest discovery; failures propagate to the exit status |
| Cache observable features | `rl/encode_frames.py` | Re-renders snapshots; fixture replay limits still apply |
| Build RL transitions | `rl/build_dataset.py` | Explicitly choose `--obs-spec v2` for observable features |
| Train BC/IQL baselines | `rl/train.py` | Frame-based actors; not yet connected to the learned failure scorer |
| Evaluate baseline policies/critics | `rl/eval_env.py`, `rl/eval_critic.py` | Closed-loop outcomes and offline diagnostics |
| Run reward comparisons | `rl/reward_experiments.sh` | Retained independent reward experiments; see RL pipeline |
| Record new data | `record/record_config.sh`, `record/record_rollouts.py` | Choose a new dataset name and explicit task/init settings |
| Build rule-derived annotations | `annotate/bench/oracle_reference.py`, `annotate/bench/oracle_labels.py` | Versioned weak supervision; review before using as new manifest labels |
| Run annotation benchmarks | `annotate/bench/run_method.py` | Historical VLM and rule baselines |
| Build and verify GPT-6 pilot labels | `annotate/bench/gpt6_pilot.py` | Observable-only sparse labels, hash freeze, privileged fact checks, masked training candidates |
| Prepare GPT-6 labels for human review | `annotate/bench/gpt6_human_review_v1.py`, `annotate/bench/gpt6_human_review_fewshot_v1.py` | New disjoint target pack plus zero-shot and human-demonstration prompts, candidate and review freezes |
| Review trajectories or GPT candidates | `annotate/human_annotator.py` | Independent annotation by default; separate candidate-assisted output with `--seed-name` and `--annotation-name` |
| Local simulation utilities | `tools/list_tasks.py`, `tools/probe_libero.py`, `tools/view_libero.py` | Task inspection, rendering and viewer |

## Historical collection schedules

`pilot/*.sh` and `record/full_run*.sh` preserve the original collection settings and seeds. They are
not the default entry point for a new collection. In particular, `full_run_ext2.sh` is a historical
migration that moves existing shifted datasets; `full_run_ext*.sh` wait on earlier log markers.
Use `record_config.sh` with a fresh name for new data.

The one-time `full_run_resume_goals8.sh` crash recovery was removed after the verified completed
collection. The redundant `annotate/bench/kill_review.sh` was removed; the existing
`review_oracle_stop.sh` targets the oracle renderer instead of killing every ffmpeg process.
Both removed scripts remain available in Git history.

Baseline sweep defaults can target privileged `object_v1` and historical oracle exports. Inspect
or override `DATASET`, `LABELS`, seeds and run tags before launching a new comparison. The new
failure learner instead requires explicit dataset/manifest paths and rejects privileged observations
unless a diagnostic override is given.
