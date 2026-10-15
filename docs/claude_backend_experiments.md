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
