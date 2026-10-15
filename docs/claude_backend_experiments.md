# Claude-backend annotation experiments

Started: 2026-10-14. Branch: `claude-subscription-backend`.

This document records running the frozen V8 atomic-action protocol with Claude
models, the failure modes found, and every pipeline change made in response,
with its rationale. Generated outputs live under ignored `outputs/`.

## Backend and deviations from V8

| Item | V8 (OpenAI) | Claude backend | Rationale |
|---|---|---|---|
| Annotators | `gpt-6.1-sol`, `gpt-6-astra`, low reasoning | Claude Haiku and Claude Sonnet, run as Claude Code subagents on the team subscription | No API key or per-call billing. The standalone `claude` CLI on this machine is not logged in, so subagents with file tools are the transport. |
| Image input | ~114 separate high-detail images per 280-frame episode | Contact sheets (`annotate_claude.py prepare`): 8 sampled frames per sheet, agent and wrist side by side, frame/camera label in a strip above each tile, native 256 px | Claude request image limits and per-image overhead. Pixels are not resized or covered. |
| Sampling | every 5th frame + last | same | Unchanged. |
| Prompt | `prompts/atomic_actions/annotation.txt` (V8) | same bytes for the baseline | Later variants are separate named conditions. |
| Validation | `validate_atomic`, zero retries | same validator; agent may repair structure, at most 3 validation runs; every attempt logged in `answers/<EP>_<TAG>_validation.json` | First-pass validity remains measurable. |

## Evaluation without human labels

The review pack excludes human labels. Evidence used:

1. Schema validity, first pass and after repair.
2. Cross-model agreement (`claude_metrics.py`): frame-level failure agreement,
   failure IoU, Cohen's kappa, category Jaccard, action-sequence edit distance.
3. Consistency lint (`claude_lint.py`): claims checked against model-visible
   gripper telemetry and against the annotation's own identity/category fields.
4. Coarse comparison with the GPT-6 Astra few-shot v3 candidates in the pack.
5. Targeted visual spot checks of disagreements.

None of these is accuracy. Agreement is a diagnostic only.

## Log

Entries are appended below as experiments complete.

### Conditions

| ID | Run dir | Prompt | Sampling | Models | Episodes |
|---|---|---|---|---|---|
| E0 | `claude_v8_baseline` | V8 | stride 5 + last | Haiku, Sonnet | R001-R006 |
| E1 | `claude_add1` | V8 + addendum v1 | stride 5 | Haiku, Sonnet | R001-R006 |
| E2 | `claude_add1_digest` | V8 + v1 + gripper digest | stride 5 | Haiku | R001-R006 |
| E3 | `claude_add2` | V8 + v2 | stride 5 | Haiku, Sonnet | R001-R006 |
| E4 | `claude_add3`, `claude_add3_rep` | V8 + v3 | stride 5 | Haiku, Sonnet (x2) | R001-R006 |
| E5 | `claude_add4` | V8 + v4 (frozen) | stride 5 | Sonnet | R001-R006 |
| HO | `heldout_v8`, `heldout_v4` | V8 / V8 + v4 | stride 5 | Haiku, Sonnet | R007-R010 |
| EV | `v8_events` | V8 | stride 5 + event frames | Sonnet | R001, R004, R005, R009 |
| EV4 | `v4_events` | V8 + v4 | stride 5 + event frames | Sonnet | R001, R004, R005, R008, R009 |

R001-R006 are the development set used to write the addenda. R007-R010 are held
out: their references were labeled before any model output on them was read,
and v4 was frozen before they were annotated.

### References

`scripts/annotate/references/claude/reference_dev_v1.json` (R001, R002, R004, R005) and
`scripts/annotate/references/claude/reference_heldout_v2.json` (R007-R010) are one reviewer's visual
judgments (Claude Opus reading the camera arrays as overviews and frame strips
with telemetry). They are not human ground truth. Two held-out entries were
corrected after targeted strip checks (R008 collision, R009 landing outside the
basket); the corrections are recorded in the files' notes.

## Findings: where the pipeline goes wrong

- **F1 Review/uncertainty coupling.** 3/4 first-pass schema failures in E0 were
  `review_required=false` with nonempty `uncertainty`. Addendum rule 1 removed
  them (E1+: 11/12 or better; held-out v4 0/8 vs V8 1/8 first-pass failures).
- **F2 Review-flag inflation.** Haiku flagged 75-100% of segments. Addendum rule
  2 cut flags to 0-67% (dev) and 0-19% for Sonnet on held-out vs 25-63% under V8.
- **F3 Referent fragmentation.** Haiku split one carried object into two
  referents once its label became legible, inventing a wrong-object failure over
  55% of R005. Rule 3 reduced it; its first wording also suppressed true
  semantic failure (R001 E1 Haiku), fixed in v2.
- **F7 Attachment after opening.** In R001 (and briefly R004) the object stays
  attached to the opened gripper. V8 gives no positive attachment test, and every
  E0 annotation reported separation. Fixed for Sonnet by the attachment rule (v2+)
  and, independently, by event-aware sampling with the unchanged V8 prompt (EV).
- **F8 Telemetry-induced false slips.** Fingers read fully closed (0.002-0.006 m)
  while the object stays wedged under the gripper (agent view). Models labeled a
  slip at 135-145 in R001/R004. My own addendum v1 rule 4 and the gripper digest
  reinforced this; v3 requires visual separation, which removed it for Sonnet.
- **F8b Over-correction.** The first attachment rule (v2/v3) fired on an object
  the robot was servoing onto (kept centered in the wrist camera), producing a
  false "still attached" over R005's regrasp attempts. v4 requires agent-view
  support and constant wrist-image size while gripper height changes.
- **F9 Short end-of-action events are under-sampled.** Delayed separation (R004,
  15 frames) and the R009 landing outside the basket (last 10 frames) get 1-2
  looks at stride 5. Event sampling fixed R001 with V8 but not R004 or R009.
- **F6 Protocol limit.** One `match_status` per episode cannot record a correct
  object followed by a wrong one (R003).

## Results summary

Dev-set event checks (`claude_events.py`, 11 checks): V8 Haiku 6, V8 Sonnet 8,
v4 Sonnet 10, v3 Sonnet 8 and 7 (two runs), v2 Haiku 10, EV (V8) Sonnet 7/10.

Held-out (R007-R010), category recall / precision:
V8 Sonnet 0.83/1.00, v4 Sonnet 0.67/1.00, V8 Haiku 0.67/0.67, v4 Haiku 0.67/1.00.
Outcome and identity were correct for every condition.

Interpretation: addendum rules 1-2 (schema coupling and review calibration) are a
robust improvement. The event-detection rules (3-6) improved the development set
but did not improve held-out recall (v4 Sonnet missed the R008 collision that V8
found), so their dev gains are at least partly fit to R001-R006. Run-to-run
variance is large: identical prompt and model detected R004's delayed separation
in one of two runs. Single-run differences of one event are not evidence.

## Later experiments

- **EV (event-aware sampling, `--event-frames`).** Adds frames at -4..+10 (step 2)
  around every gripper-command change and every second frame of the last 12.
  With the unchanged V8 prompt it let Sonnet find R001's delayed separation;
  with v4 it found R008's collision. It did not remove false slips, and v4 + events
  revived R005's false attachment (dev events 6/10).
- **R009 reference revised (v3).** The bottle is beside the basket only in the
  final frame and may still be falling; V8 forbids inferring an unobserved end
  state, and all models flagged that segment for review. The placement outcome is
  review-only and unscored. Held-out under v3: V8 Sonnet recall/precision
  1.00/1.00, v4 Sonnet 0.83/1.00, V8 Haiku 0.83/0.67, v4 Haiku 0.83/1.00.
- **AU (second-pass Sonnet audit, `references/claude/auditor_brief.txt`).** A
  checklist audit of an existing annotation against the same frames. Held-out:
  fixed a false lift/drop in R007 (strip-verified), wording only in R010, no
  change in R008/R009. Dev: fixed R001's missed delayed separation (V8 Sonnet
  draft) and R005's referent split and missed drop (V8 Haiku draft); did not fix
  R004's false slip or its 15-frame delayed separation. No new false positives in
  seven audits.

## Recommendation

Keep addendum rules 1-2 (schema coupling, review calibration). Treat rules 3-6 as
dev-fit until they are re-evaluated on new episodes. The most promising
structural change is annotate then audit: V8 (+ rules 1-2) followed by a Sonnet
audit, with Haiku-draft + Sonnet-audit as a cheaper variant. Before any paper
claim: human review of R001-R010, repeated runs to estimate variance, and an
evaluation on the untouched R011-R012.
