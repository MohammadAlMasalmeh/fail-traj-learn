"""Verify split invariance, unmatched actions, unknowns, and category support."""
from copy import deepcopy
import unittest

from compare_action_groups import action_groups, compare


def segment(start, end, action_id='a1', action='transport', categories=()):
    return dict(start_frame=start, end_frame=end, action_id=action_id, action=action,
                categories=list(categories), what_happened='Move the object.',
                observed_result='Object moves.', review_required=False, uncertainty='')


def annotation(*segments):
    return dict(segments=list(segments))


class ActionGroupTests(unittest.TestCase):
    def test_equivalent_split_and_local_ids_do_not_change_scores(self):
        a = annotation(segment(0, 20, categories=['sequencing_semantic']))
        b = annotation(segment(0, 10, 'local_9', categories=['sequencing_semantic']),
                       segment(10, 20, 'local_9', categories=['sequencing_semantic']))
        self.assertEqual(compare(a, b)['scores'], dict(action_agreement=1, category_agreement=1, combined_agreement=1))
        self.assertEqual(action_groups(b)[0]['category_spans']['sequencing_semantic'], [[0, 20]])

    def test_extra_collision_counts_once_despite_repeated_segments(self):
        a = annotation(segment(0, 20, categories=['sequencing_semantic']))
        b = annotation(segment(0, 10, categories=['sequencing_semantic']),
                       segment(10, 15, categories=['sequencing_semantic', 'collision']),
                       segment(15, 20, categories=['sequencing_semantic', 'collision']))
        result = compare(a, b)
        self.assertEqual(result['counts']['matched_actions'], 1)
        self.assertEqual(result['scores']['category_agreement'], 2/3)
        self.assertEqual(result['scores']['combined_agreement'], 4/5)
        self.assertEqual(result['matched'][0]['right_only_categories'], ['collision'])
        self.assertEqual(result['matched'][0]['right']['category_spans']['collision'], [[10, 20]])

    def test_wrong_action_does_not_receive_category_match_credit(self):
        a = annotation(segment(0, 20, action='reach', categories=['grasp']))
        b = annotation(segment(0, 20, action='grasp', categories=['grasp']))
        self.assertEqual(compare(a, b)['scores']['combined_agreement'], 0)

    def test_missing_action_reduces_agreement(self):
        a = annotation(segment(0, 10, action='reach'), segment(10, 20, 'a2', 'grasp'))
        b = annotation(segment(0, 10, action='reach'))
        result = compare(a, b)
        self.assertEqual(result['scores']['action_agreement'], 2/3)
        self.assertEqual(len(result['unmatched_left']), 1)

    def test_no_categories_is_undefined_instead_of_perfect_category_score(self):
        a = annotation(segment(0, 20))
        self.assertIsNone(compare(a, a)['scores']['category_agreement'])
        self.assertEqual(compare(a, a)['scores']['combined_agreement'], 1)

    def test_unknown_action_does_not_count_as_known_agreement(self):
        a = annotation(segment(0, 20, action='unknown'))
        self.assertEqual(compare(a, a)['scores']['action_agreement'], 0)

    def test_order_and_overlap_determine_alignment_not_matching_categories(self):
        a = annotation(segment(0, 10, categories=['collision']), segment(10, 20, 'a2', categories=['grasp']))
        b = annotation(segment(0, 10, categories=['grasp']), segment(10, 20, 'a2', categories=['collision']))
        result = compare(a, b)
        self.assertEqual([(r['left_index'], r['right_index']) for r in result['matched']], [(0, 0), (1, 1)])
        self.assertEqual(result['scores']['category_agreement'], 0)
        self.assertEqual(result['scores']['action_agreement'], 1)


if __name__ == '__main__':
    unittest.main()
