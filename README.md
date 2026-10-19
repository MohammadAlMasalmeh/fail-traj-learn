# fail-traj-learn

Failure-conditioned temporal learning from robot rollouts. The research question is whether
localized failure semantics can provide a useful training signal for improving a policy beyond
terminal-reward offline RL.

The current implementation learns chunk representations with a causal transformer, failure-category
queries, and supervised role/event heads. The next stage is a prospective action-conditioned scorer,
followed by frozen-score costs for RL. Neither the scorer nor that RL integration is implemented yet.
The earlier ordinal-advantage methods remain available as baselines.

## Start here

| Document | Purpose |
|---|---|
| [Sol V8 distillation dataset](docs/sol_v8_distillation_500.md) | Download the public 500-episode release, apply training masks, and reproduce the dataset pipeline |
| [Atomic annotation runbook (V8)](docs/annotation_runbook_v8.md) | Frozen Sol/Astra protocol, annotation and viewer commands, fixed scoring, and next steps |
| [Training guide](docs/failure_training.md) | Implemented model, annotation manifest, commands, checkpoints |
| [GPT-6 review guide](REVIEW.md) | Download the review pack and launch the local browser reviewer on Windows, macOS, or Linux |
| [Data verification](docs/failure_data_verification.md) | Current simulation/oracle inventory, checks and limitations |
| [Method design](docs/failure_attention_design.md) | Research direction, planned scorer/RL interface and experiments |
| [Oracle annotation contract](docs/oracle_segmentation.md) | Meaning and limitations of the existing r6 labels |
| [Offline RL](docs/rl_pipeline.md) | Existing BC/IQL, observation specs, reward and segment baselines |
| [Documentation index](docs/README.md) | Current guides and dated experiment history |
| [Script index](scripts/README.md) | Entry points for recording, annotation, verification and training |

## Current status

- **Model:** chunk encoder, causal transformer, category-query pooling, role and event-timing heads;
  training CLI and held-out validation. Thirteen focused tests and a GPU smoke benchmark passed.
  Real-data representation training still requires an audited annotation manifest.
- **Data:** 2,080 simulation episodes across 87 datasets. The built `object_v2` dataset contains
  1,300 episodes / 251,229 frames, all covered by the current r6 frame-label export.
- **Annotation:** rule-derived weak supervision, not semantic ground truth. The full reference
  directory mixes r3/r4/r6. Known r6 semantic disagreements, episode-level causes, placeholder
  visible-failure timestamps and missing verified-negative masks require explicit handling.
- **Results:** the [saved RL results](docs/rl_results.md) concern historical baseline experiments
  and predate documented fixes. They do not demonstrate the new attention method.
- **Transfer:** the current corpus uses one collecting checkpoint. Cross-policy and cross-embodiment
  transfer, including real-robot validation, remain future work.

## Run the first model

Use the existing LeRobot environment (Python 3.12 with NumPy, pandas, PyArrow and PyTorch).
Dataset/model downloads and simulation dependencies are managed in that separate environment.

```bash
export FTL_PROJ=/home/aliu/projects/fail-traj-learn
export FTL_RL="$FTL_PROJ/rl"
PY="$FTL_PROJ/lerobot/.venv/bin/python"

# Verify the new model and loader on CPU.
$PY -m unittest discover -s scripts/rl -p 'test_failure_model.py' -v

# Review coverage before training. Create this manifest using the training guide.
$PY scripts/rl/train_failure_model.py \
  --dataset "$FTL_RL/datasets/object_v2" \
  --manifest /path/to/reviewed_failure_manifest.json --audit-only

# The output directory must be new.
$PY scripts/rl/train_failure_model.py \
  --dataset "$FTL_RL/datasets/object_v2" \
  --manifest /path/to/reviewed_failure_manifest.json \
  --out "$FTL_RL/failure_runs/attention_s0"
```

Use `obs_v2` for the main experiment: proprioception, frozen features from two cameras, task-language
features and time. `obs_v1` contains privileged object/contact/fixture state and is a diagnostic
baseline. Oracle annotations supervise training; they are not observation inputs.

## Repository and artifact layout

| Location | Contents |
|---|---|
| `scripts/rl/` | Failure representation learning and existing BC/IQL baselines |
| `scripts/record/` | Recorder, configurable wrapper and historical collection schedules |
| `scripts/annotate/` | Annotation experiments, rule-derived references and exports |
| `scripts/analysis/` | Corpus audits, snapshot checks and review tools |
| `scripts/tools/`, `scripts/pilot/` | Simulation utilities and historical pilot experiments |
| `docs/` | Current guides, audits and dated research history |
| `outputs/` | Ignored generated reports, images, videos and review evidence |
| `$FTL_PROJ/data/` | External recordings; pilot datasets live in `archive_pre_v3/` |
| `$FTL_RL/` | External feature caches, built datasets, training runs and checkpoints |
| `$FTL_PROJ/bench/` | External oracle references, exports and annotation benchmark artifacts |
| `$FTL_PROJ/lerobot/` | LeRobot checkout and `.venv` |

Python tools generally use `FTL_PROJ`, `FTL_RL` and `FTL_WIN_OUT`. Most shell wrappers instead use
`PROJ` and `SCRIPTS`; set these explicitly when changing machines. A direct Python invocation avoids
wrapper-specific logging and default settings. Cached features and experiment artifacts are not
included in a fresh clone.

Recorded decisions are 10 frames at 20 Hz in the verified corpus. The final chunk may be shorter.
Sidecars store chunk snapshots and gripper state; newer recordings can also store fixed-fixture
poses. Existing fixture scenes can lack those poses, so exact replay must be checked per recording.
See the [data audit](docs/failure_data_verification.md) before using restored snapshots for repair studies.
