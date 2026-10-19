"""Check semantic preservation and uncertainty masks in distillation export."""
import json
import unittest

from export_distillation_v8 import messages, target_with_masks, validate_splits
from test_annotate_trajectories import annotation, segment


class ExportTests(unittest.TestCase):
    def test_mask_selects_whole_uncertain_value_without_changing_label(self):
        first = segment(0,10,"a1","transport")
        second = segment(10,20,"a2","release")
        second.update(review_required=True,uncertainty='Separation is unresolved.',
                      what_happened='The robot releases the café object? {"uncertain": true}')
        original = annotation([first,second])
        text,masks = target_with_masks(original)
        self.assertEqual(json.loads(text),original)
        self.assertEqual([m["teacher_loss_mask"] for m in masks["segments"]],[1,0])
        self.assertFalse(masks["human_verified"])
        self.assertEqual(len(masks["assistant_excluded_char_ranges"]),1)
        span = masks["assistant_excluded_char_ranges"][0]
        self.assertEqual(json.loads(text[span["start"]:span["end"]]),second)

    def test_unresolved_identity_and_all_flagged_episode_are_explicit(self):
        value = annotation([dict(segment(0,20,"a1","transport"),review_required=True,uncertainty="Occluded")])
        value["identity_check"].update(match_status="unresolved",uncertainty="Cannot distinguish objects")
        text,masks = target_with_masks(value)
        self.assertEqual(masks["identity_teacher_loss_mask"],0)
        self.assertFalse(masks["has_unflagged_segment_targets"])
        self.assertEqual(len(masks["assistant_excluded_char_ranges"]),2)
        a,b = masks["assistant_excluded_char_ranges"]
        self.assertLess(a["end"],b["start"])
        self.assertEqual(json.loads(text[a["start"]:a["end"]]),value["identity_check"])

    def test_multimodal_input_order_and_assistant_target(self):
        ep = dict(id="D1",task="Move the object",n_frames=6,fps=20,sampled_frames=[0,5],
                  telemetry=[dict(frame=0,seconds=0),dict(frame=5,seconds=.25)])
        result = messages("Frozen prompt",ep,"target")
        self.assertEqual([c["type"] for c in result[0]["content"]],["text","text","image","image","text","image","image"])
        self.assertEqual([c["image"] for c in result[0]["content"] if c["type"]=="image"],
                         ["images/D1/0000_agent.jpg","images/D1/0000_wrist.jpg","images/D1/0005_agent.jpg","images/D1/0005_wrist.jpg"])
        self.assertEqual(result[1],dict(role="assistant",content="target"))

    def test_source_episode_identity_cannot_cross_splits(self):
        row = dict(id="D1",dataset="scene",episode_index=1)
        with self.assertRaisesRegex(ValueError,"leakage"):
            validate_splits(dict(splits=dict(train=[row],test=[dict(row,id="T1")])),{"D1"})


if __name__ == "__main__":
    unittest.main()
