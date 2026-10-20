"""Compact replacement for the ~33k-char generic annotation prompt (kept: schema, vocab, key rules)."""

SHORT_INSTRUCTIONS = """# Annotate atomic actions, physical results, and failure/non-failure
Label the whole robot-manipulation episode relative to its language task, using only the supplied synchronized camera images (agent cam, wrist cam) and robot telemetry. Judge from the episode's own evidence; do not assume success or failure. Telemetry lines list every sampled source frame; images are shown only for some of them.

Rules:
- Separate action (what the robot does/attempts), observed physical result (what changes/unchanged/unresolved), and failure (independently supported failure during the behavior).
- Actions: reach, grasp, transport, release, withdraw, idle (or a task-specific verb; unknown only if unresolvable). Action terms never assert success; use "attempts" where acquisition/separation is not established. Result phrases: "The gripper aligns around <obj>", "establishes control of <obj>", "<obj> remains under gripper control", "<obj> separates from the gripper", "<obj> remains supported inside/on <dest>" (only if support is established).
- Claim only what the evidence shows; absence of evidence is not a verified negative. If uncertain, give the best-supported judgment as provisional, set review_required=true, and explain in uncertainty.
- Failure is observable unsuccessful attempt, task-inconsistent action, harmful side effect, or sustained unproductive behavior; start a failure at the first supplied observation that supports it (never backdate). Wrong-object engagement is semantic failure even if acquisition is uncertain. Physically successful actions can be task-inconsistent.
- Object identity: separate the requested object from the object actually manipulated. Use the instruction's terms as canonical names when supported, lowercase_with_underscore ids with numeric suffix (object_1, basket_1); generic object_N/destination_N when unresolved.
- Segments: atomic actions with ids a1, a2, ... in order; half-open frame intervals [start_frame, end_frame) partitioning [0, n_frames) with no gaps/overlaps; split an action when failure or its category set changes (same action_id). Each segment needs an evidence-based boundary_reason (episode_start for the first). Cite only supplied frames/views.
- Failure categories: reaching, grasp, manipulation, sequencing_semantic, collision, task_not_attempted, sudden_episode_termination, visible_hardware_issue, other. Severity: low|medium|catastrophic. Resetability: immediate|minor_reset|hard_reset|unresettable.
- what_happened: actor + canonical operation + object/destination + observed result + failure effect.

Output: only an unfenced JSON object with exactly episode_id, identity_check, segments.
identity_check = {requested_object, manipulated_object, match_status (supported_match|supported_mismatch|unresolved), evidence:[{frame,camera(agent|wrist),observation}], uncertainty}.
Each segment = {start_frame, end_frame, action_id, action, object_id|null, destination_id|null, observed_result, boundary_reason, failure(bool), what_happened, categories([] unless failure), recovery|null, severity|null, resetability|null, evidence:[{frame,camera,observation}], review_required(bool), uncertainty("" unless review_required)}.
"""
