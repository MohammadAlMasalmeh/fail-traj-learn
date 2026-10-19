# Documentation index

## Current implementation and decisions

| Document | Scope |
|---|---|
| [Sol V8 distillation dataset](sol_v8_distillation_500.md) | Public release, verified download, required training masks, source inventory, and reproduction commands |
| [Atomic annotation runbook (V8)](annotation_runbook_v8.md) | Frozen binary atomic-action protocol, reproducible runs, combined review, scoring, and human-evaluation priorities |
| [Failure training](failure_training.md) | Implemented representation learner and manifest contract |
| [Failure attention design](failure_attention_design.md) | Research plan; separates implemented pretraining from future scoring/control |
| [Joint failure process model](joint_failure_process_model.md) | Proposed multi-task causal architecture for segment recognition, future failure risk and recovery |
| [Data verification](failure_data_verification.md) | 2026-09-13 inventory, structural checks, fresh reference comparison and replay limits |
| [Oracle segmentation](oracle_segmentation.md) | Current label semantics, uncertainty and conversion requirements |
| [RL pipeline](rl_pipeline.md) | Existing BC/IQL and historical segment/reward baselines |

## Evidence and experiment history

These documents explain earlier choices and failures. Their dates, reference versions and evaluation
sets matter; they are not current training instructions or independent confirmations of the new method.

| Document | What it preserves |
|---|---|
| [Frozen r6 audit](oracle_r6_audit.md) | Unresolved semantic disagreements; still relevant to current oracle code |
| [RL results](rl_results.md) | Historical runs, explicitly marked as predating implementation fixes |
| [RL verification](rl_verification.md) | Audit of those runs and the resulting fixes |
| [Segmentation benchmark](segmentation_benchmark.md) | VLM methods compared against versioned rule-derived references |
| [GPT-6 annotation experiments](gpt6_annotation_experiments.md) | Assistant-assisted annotation evidence and limitations |
| [GPT-6 observable-label pilot v1](gpt6_pilot_v1.md) | Fixed multi-head taxonomy, 30-episode blind pilot and privileged fact-check results |
| [GPT-6 conversation labels with human review](gpt6_human_review_v1.md) | Frozen zero-shot and human-demonstration in-context Codex protocols, with candidate-assisted review |
| [VLM pipeline](annotation_pipeline.md) | Experimental Qwen annotation workflow |
| [Observable-event experiment](event_experiment.md) | Local event-recognition experiment |
| [Segmentation iterations](segmentation_iteration_results.md) | Development on one reviewed episode, not held-out evidence |
| [VLM input experiments](vlm_input_experiments.md) | Historical input/prompt ablations |
| [Pilot results](pilot_results.md) | Original failure-induction and data-collection experiments |
| [Ordinal-advantage proposal](proposal_overview.md) | Superseded research proposal underlying retained RL baselines |

Generated reports linked under `../outputs/` are local artifacts and may be absent in a fresh clone.
Keep those artifacts when cleaning local storage: some contain visual reviews and raw model responses
that are evidence, not interchangeable caches.
