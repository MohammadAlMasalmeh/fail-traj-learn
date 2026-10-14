"""Check action linking, temporal coverage, evidence, and safe request submission."""
from copy import deepcopy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import annotate_trajectories as run


EP = dict(id="test", n_frames=20, sampled_frames=[0, 5, 10, 15, 19])


def segment(start, end, action_id, action, failure=False):
    return dict(
        start_frame=start, end_frame=end, action_id=action_id, action=action,
        object_id="object_1", destination_id="basket_1", observed_result="Observed relocation.",
        boundary_reason="episode_start" if start == 0 else "Supported operation or failure transition.",
        failure=failure, what_happened="The robot transports object_1.",
        categories=["collision"] if failure else [],
        recovery="Clear the obstacle." if failure else None,
        severity="low" if failure else None, resetability="immediate" if failure else None,
        evidence=[dict(frame=0, camera="agent", observation="Visible object.")],
        review_required=False, uncertainty="",
    )


def annotation(segments):
    return dict(episode_id="test", identity_check=dict(
        requested_object="object", manipulated_object="object_1",
        match_status="supported_match", uncertainty="",
        evidence=[dict(frame=0, camera="agent", observation="Object identity evidence.")],
    ), segments=segments)


class AtomicActionsTests(unittest.TestCase):
    def test_binary_schema_rejects_progress_and_non_boolean_failure(self):
        value = annotation([segment(0, 20, 'a1', 'transport')])
        run.validate(value, EP)
        extra = deepcopy(value)
        extra['segments'][0]['progress'] = True
        with self.assertRaisesRegex(ValueError, 'fields'):
            run.validate(extra, EP)
        invalid = deepcopy(value)
        invalid['segments'][0]['failure'] = 0
        with self.assertRaisesRegex(ValueError, 'boolean failure'):
            run.validate(invalid, EP)

    def test_legacy_progress_annotations_remain_reviewable(self):
        value = annotation([segment(0, 10, 'a1', 'transport'),
                            segment(10, 20, 'a1', 'transport')])
        value['segments'][0]['progress'] = True
        value['segments'][1]['progress'] = False
        run.validate_atomic(value, EP, label_mode='progress_failure')
        del value['segments'][1]['progress']
        with self.assertRaisesRegex(ValueError, 'fields'):
            run.validate_atomic(value, EP, label_mode='progress_failure')

    def test_different_actions_with_identical_task_labels_are_valid(self):
        run.validate(annotation([segment(0, 10, "a1", "grasp"),
                                 segment(10, 20, "a2", "transport")]), EP)

    def test_brief_failure_keeps_link_to_continuing_action(self):
        run.validate(annotation([segment(0, 5, "a1", "transport"),
                                 segment(5, 10, "a1", "transport", True),
                                 segment(10, 20, "a1", "transport")]), EP)

    def test_continuing_reach_can_resolve_its_previously_unspecified_object(self):
        initial = segment(0, 10, "a1", "reach")
        initial.update(object_id=None, destination_id=None)
        directed = segment(10, 20, "a1", "reach", True)
        directed.update(destination_id=None, categories=["sequencing_semantic"])
        run.validate(annotation([initial, directed]), EP)

    def test_known_object_cannot_change_within_one_action(self):
        first = segment(0, 5, "a1", "transport")
        middle = segment(5, 10, "a1", "transport", True)
        middle["object_id"] = None
        last = segment(10, 20, "a1", "transport")
        last["object_id"] = "object_2"
        with self.assertRaises(ValueError):
            run.validate(annotation([first, middle, last]), EP)

    def test_same_verb_distinct_attempts_can_have_identical_labels(self):
        run.validate(annotation([segment(0, 10, "a1", "grasp"),
                                 segment(10, 20, "a2", "grasp")]), EP)

    def test_invalid_coverage_evidence_and_action_links_are_rejected(self):
        valid = annotation([segment(0, 10, "a1", "grasp"),
                            segment(10, 20, "a2", "transport")])
        mutations = [
            ("gap", lambda a: a["segments"][1].update(start_frame=11)),
            ("missing tail", lambda a: a["segments"][1].update(end_frame=19)),
            ("unsupplied frame", lambda a: a["segments"][1]["evidence"][0].update(frame=7)),
            ("wrong camera", lambda a: a["segments"][1]["evidence"][0].update(camera="external")),
            ("reused action with changed operation", lambda a: a["segments"][1].update(action_id="a1")),
            ("skipped action id", lambda a: a["segments"][1].update(action_id="a3")),
            ("unexplained uncertainty", lambda a: a["segments"][1].update(review_required=True)),
        ]
        for name, mutate in mutations:
            with self.subTest(name=name):
                invalid = deepcopy(valid)
                mutate(invalid)
                with self.assertRaises(ValueError):
                    run.validate(invalid, EP)

    def test_identical_pieces_of_one_action_must_merge(self):
        with self.assertRaisesRegex(ValueError, "must be merged"):
            run.validate(annotation([segment(0, 10, "a1", "grasp"),
                                     segment(10, 20, "a1", "grasp")]), EP)

    def test_models_receive_identical_content(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(run, "OUT", Path(tmp)):
            root = Path(tmp)
            (root / "prompt.txt").write_text("Shared prompt.", encoding="utf-8")
            (root / "test").mkdir()
            for camera in ("agent", "wrist"):
                (root / "test" / f"0000_{camera}.jpg").write_bytes(b"same evidence")
            ep = dict(EP, task="task", fps=20,
                      telemetry=[dict(frame=0, seconds=0, gripper_command="open")])
            sol, astra = run.payload("sol", ep), run.payload("astra", ep)
            self.assertEqual(sol["input"], astra["input"])
            self.assertEqual({k:v for k,v in sol.items() if k != "model"},
                             {k:v for k,v in astra.items() if k != "model"})

    def test_existing_reservation_is_never_resent(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(run, "OUT", Path(tmp)):
            (Path(tmp) / "attempts").mkdir()
            (Path(tmp) / "attempts/test_sol.json").write_text("{}", encoding="utf-8")
            with patch.object(run.transport, "post") as post:
                run.execute("sol", EP, {"openai": "test"})
                post.assert_not_called()

    def test_failed_preflight_never_submits_generation(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(run, "OUT", Path(tmp)):
            (Path(tmp) / "attempts").mkdir()
            body = dict(model="test", input=[], reasoning={}, text={}, truncation="disabled")
            with patch.object(run, "payload", return_value=body), patch.object(
                run.transport, "post", return_value={"input_tokens": run.MAX_INPUT + 1}
            ) as post:
                run.execute("sol", EP, {"openai": "test"})
                self.assertEqual(post.call_count, 1)
                record = run.common.load(Path(tmp) / "attempts/test_sol.json")
                self.assertFalse(record["generation_sent"])
                self.assertEqual(record["status"], "stopped")


if __name__ == "__main__":
    unittest.main()
