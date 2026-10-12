"""Focused checks for frame alignment and judge isolation/validation."""
import json
import unittest
from unittest.mock import patch

import compare_annotation_descriptions as judge


class DescriptionEvaluationTests(unittest.TestCase):
    def test_different_boundaries_pair_every_frame_once(self):
        def annotation(spans):
            return dict(segments=[dict(start_frame=a, end_frame=b, what_happened=str(a))
                                  for a, b in spans])
        pairs = judge.overlapping_pairs("R001", annotation([(0, 10), (10, 20)]),
                                        annotation([(0, 5), (5, 15), (15, 20)]))
        self.assertEqual([(p["start_frame"], p["end_frame"]) for p in pairs],
                         [(0, 5), (5, 10), (10, 15), (15, 20)])
        self.assertEqual([p["left_index"] for p in pairs], [0, 0, 1, 1])
        self.assertEqual([p["right_index"] for p in pairs], [0, 1, 1, 2])

    def test_reverse_order_swaps_text_and_excludes_expected_scores(self):
        batch = dict(task="Task", pairs=judge.controls())
        with patch.object(judge.Path, "read_text", return_value="Rubric"):
            a, b = [judge.payload(batch, order) for order in ("ab", "ba")]
        first, second = [json.loads(x["input"][1]["content"])["pairs"] for x in (a, b)]
        for x, y in zip(first, second):
            self.assertEqual(set(x), {"pair_id", "A", "B"})
            self.assertEqual((x["A"], x["B"]), (y["B"], y["A"]))
        self.assertEqual(a["input"][0], b["input"][0])

    def test_validator_rejects_duplicate_missing_and_conflicting_scores(self):
        batch = dict(pairs=[dict(pair_id="p001"), dict(pair_id="p002")])
        def row(pid, score=5):
            return dict(pair_id=pid, score=score, verdict=judge.VERDICTS[score],
                        explanation="Same event.", differences=[] if score == 5 else ["Different object."])
        judge.validate(dict(evaluations=[row("p002"), row("p001")]), batch)
        for invalid in ([row("p001"), row("p001")], [row("p001")],
                        [row("p001"), dict(row("p002"), score=True)],
                        [row("p001"), dict(row("p002"), verdict="contradiction")],
                        [row("p001"), dict(row("p002", 2), differences=[])]):
            with self.assertRaises(ValueError):
                judge.validate(dict(evaluations=invalid), batch)

    def test_existing_reservation_never_resends(self):
        with patch.object(judge.Path, "open", side_effect=FileExistsError), \
                patch.object(judge.transport, "post") as post:
            judge.execute(dict(id="controls"), "ab", {})
        post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
