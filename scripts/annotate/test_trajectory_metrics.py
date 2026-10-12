import unittest
import tempfile
from pathlib import Path
from trajectory_metrics import evaluate,boundaries,semantic_overlap


def seg(a,b,p=False,f=False,**kw):
    return dict(start=a,end=b,progress=p,failure=f,description='Robot moves the object.',failure_types='wrong_action' if f else 'none',**kw)

class MetricsTests(unittest.TestCase):
    def test_perfect_labels(self):
        rows=[seg(0,10,p=True),seg(10,20,f=True)]
        m=evaluate(rows,rows,20,20)
        self.assertEqual(m['joint_accuracy'],1)
        self.assertEqual(m['axes']['failure']['f1'],1)
        self.assertEqual(m['phase_boundaries'][0]['f1'],1)
    def test_splitting_same_state_keeps_state_score_but_penalizes_phase_boundaries(self):
        r=[seg(0,20,f=True)];p=[seg(0,10,f=True),seg(10,20,f=True)]
        m=evaluate(p,r,20,20)
        self.assertEqual(m['binary_state_segment_overlap'][1]['f1'],1)
        self.assertEqual(m['phase_boundaries'][1]['fp'],1)
        self.assertIsNone(m['binary_state_boundaries'][1]['f1'])
    def test_boundary_one_to_one_and_timing(self):
        m=boundaries([9,10,11],[10],20,.1)
        self.assertEqual(m['tp'],1);self.assertEqual(m['fp'],2)
        self.assertEqual(m['matched_mae_seconds'],0)
        self.assertEqual(boundaries([15],[10],20,.25)['tp'],1)
        self.assertEqual(boundaries([16],[10],20,.25)['tp'],0)
    def test_missed_failure_and_missing_prediction_not_negative(self):
        m=evaluate([seg(0,10)], [seg(0,20,f=True)],20,20)
        self.assertEqual(m['axes']['failure']['fn'],20)
        self.assertEqual(m['prediction_coverage'],.5)
        self.assertEqual(m['joint_accuracy'],0)
    def test_both_is_not_single_class(self):
        m=evaluate([seg(0,20,p=True,f=True)],[seg(0,20,p=True)],20,20)
        self.assertEqual(m['axes']['progress']['f1'],1)
        self.assertEqual(m['axes']['failure']['fp'],20)
    def test_no_failures_is_na_not_perfect_f1(self):
        m=evaluate([seg(0,20)],[seg(0,20)],20,20)
        self.assertIsNone(m['axes']['failure']['f1'])
        self.assertEqual(m['axes']['failure']['accuracy'],1)
    def test_overlap_weighting_handles_split_merge(self):
        p=[seg(0,10),seg(10,20)];r=[seg(0,20)]
        m=semantic_overlap(p,r,20,{'Robot moves the object.':[1,0]})
        self.assertEqual(m['time_weighted_cosine'],1)
        self.assertEqual(m['text_coverage_fraction'],1)
    def test_unresolved_reference_blocked_but_agreement_allowed(self):
        r=[seg(0,20,review_required=True)]
        with self.assertRaises(ValueError):evaluate(r,r,20,20)
        self.assertEqual(evaluate(r,r,20,20,require_verified=False)['joint_accuracy'],1)
    def test_matching_maximizes_count_before_timing(self):
        m=boundaries([0,4],[3,7],1,4)
        self.assertEqual(m['tp'],2)

if __name__=='__main__':unittest.main()
