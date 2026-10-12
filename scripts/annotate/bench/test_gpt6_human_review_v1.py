import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("gpt6_human_review_v1", HERE / "gpt6_human_review_v1.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class GPT6HumanReviewTests(unittest.TestCase):
    def meta(self):
        return {"id": "R001", "task": "pick up the butter and place it in the basket",
                "fps": 20, "n_frames": 10}

    def candidate(self):
        return {
            "version": "gpt6_human_review_v1", "model_family": "GPT-6 Astra",
            "id": "R001", "task": self.meta()["task"], "fps": 20,
            "episode_outcome": "failure", "target_description": "red and yellow box",
            "summary": "The gripper misses the target.",
            "segments": [
                {"start": 0, "end": 4, "text": "approaches target", "classification": "progress",
                 "failure_type": "unknown", "confidence": 0.8, "note": "", "evidence_frames": [0, 3]},
                {"start": 4, "end": 10, "text": "closes beside target", "classification": "failure",
                 "failure_type": "missed_target", "confidence": 0.7, "note": "occluded",
                 "evidence_frames": [4, 9]},
            ],
            "events": [], "trajectory_note": "",
        }

    def test_valid_candidate(self):
        candidate = self.candidate()
        self.assertIs(MODULE.validate_gpt_annotation(candidate, self.meta()), candidate)

    def test_rejects_gap_and_unknown_failure_type(self):
        candidate = self.candidate()
        candidate["segments"][1]["start"] = 5
        with self.assertRaisesRegex(ValueError, "coverage"):
            MODULE.validate_gpt_annotation(candidate, self.meta())
        candidate = self.candidate()
        candidate["segments"][1]["failure_type"] = "unknown"
        with self.assertRaisesRegex(ValueError, "specific"):
            MODULE.validate_gpt_annotation(candidate, self.meta())

    def test_rejects_out_of_interval_evidence(self):
        candidate = self.candidate()
        candidate["segments"][0]["evidence_frames"] = [7]
        with self.assertRaisesRegex(ValueError, "inside"):
            MODULE.validate_gpt_annotation(candidate, self.meta())

    def test_prompts_enforce_isolation_and_three_distinct_outputs(self):
        first, second, third = (MODULE.prompt_one("R001"), MODULE.prompt_two("R001"),
                                MODULE.prompt_three("R001"))
        self.assertIn("Do not read private manifests", first)
        self.assertIn("inspect every listed page", first.lower())
        self.assertIn("gpt6_observation.json", first)
        self.assertIn("gpt6_audit.json", second)
        self.assertIn("gpt6_annotation.json", third)
        self.assertNotIn("private_manifest.json`", second)

    def test_human_review_requires_explicit_decision_and_note_for_correction(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate_path = Path(tmp) / "gpt6_annotation.json"
            candidate_path.write_text(json.dumps(self.candidate()))
            review = {
                "id": "R001", "episode_outcome": "failure",
                "review_of": {"sha256": MODULE.sha(candidate_path)},
                "candidate_decision": "unreviewed", "candidate_review_note": "",
                "segments": [{"start": 0, "end": 10, "text": "missed target",
                              "classification": "failure", "failure_type": "missed_target"}],
            }
            with self.assertRaisesRegex(ValueError, "accept or correct"):
                MODULE.validate_human_review(review, self.meta(), candidate_path)
            review["candidate_decision"] = "corrected"
            with self.assertRaisesRegex(ValueError, "reviewer note"):
                MODULE.validate_human_review(review, self.meta(), candidate_path)
            review["candidate_review_note"] = "Moved the failure boundary."
            MODULE.validate_human_review(review, self.meta(), candidate_path)
            review["episode_outcome"] = "unknown"
            review["trajectory_note"] = "The recording ends before final support is visible."
            MODULE.validate_human_review(review, self.meta(), candidate_path)
            review["trajectory_note"] = ""
            with self.assertRaisesRegex(ValueError, "terminal uncertainty"):
                MODULE.validate_human_review(review, self.meta(), candidate_path)

    def test_conversation_artifacts_require_every_manifest_page(self):
        manifest = {"id": "R001", "pages": ["pages/a.jpg", "pages/b.jpg"]}
        observation = {
            "id": "R001", "inspected_pages": manifest["pages"], "target_description": "butter box",
            "phase_ledger": [], "transition_candidates": [], "uncertainties": [],
        }
        audit = {"id": "R001", **{key: [] for key in (
            "checked_command_runs", "supported_transitions", "rejected_transitions",
            "possible_omissions", "semantic_questions", "remaining_uncertainties",
        )}}
        MODULE.validate_conversation_artifacts(observation, audit, manifest)
        audit["remaining_uncertainties"] = {"frame_10": "occluded"}
        MODULE.validate_conversation_artifacts(observation, audit, manifest)
        observation["inspected_pages"] = ["pages/a.jpg"]
        with self.assertRaisesRegex(ValueError, "exactly match"):
            MODULE.validate_conversation_artifacts(observation, audit, manifest)


if __name__ == "__main__":
    unittest.main()
