"""Verify duration weighting, split invariance, and incomplete score coverage."""
import unittest
from score_review_text import summarize


def annotation(*pieces):
    return dict(segments=[dict(start_frame=a, end_frame=b, what_happened=t) for a,b,t in pieces])


def dataset(left, right):
    return dict(annotation_manifest_sha256='batch', runs=[dict(id='atomic_actions')],
                episodes=[dict(id='R001', n_frames=10, versions={
                    'atomic_actions': dict(openai=left, gemini=right)})])


def scores(*rows):
    return dict(model='test', annotation_manifest_sha256='batch',
                pairs=[dict(left=a,right=b,score=s) for a,b,s in rows])


class SimilaritySummaryTests(unittest.TestCase):
    def test_weighted_mean_and_segment_split_invariance(self):
        right=annotation((0,10,'reference'))
        cached=scores(('brief','reference',.2),('long','reference',.8))
        a=dataset(annotation((0,2,'brief'),(2,10,'long')),right)
        b=dataset(annotation((0,2,'brief'),(2,5,'long'),(5,10,'long')),right)
        first=summarize(a,cached)['runs'][0]['overall']
        self.assertAlmostEqual(first['mean_all_frames'],.68)
        self.assertEqual(first,summarize(b,cached)['runs'][0]['overall'])

    def test_missing_pairs_leave_all_frame_mean_undefined(self):
        d=dataset(annotation((0,2,'brief'),(2,10,'missing')),annotation((0,10,'reference')))
        row=summarize(d,scores(('brief','reference',.2)))['runs'][0]['overall']
        self.assertIsNone(row['mean_all_frames'])
        self.assertEqual(row['scored_frames'],2)
        self.assertAlmostEqual(row['mean_scored_frames'],.2)

    def test_overall_weights_episodes_by_frame_count(self):
        d=dataset(annotation((0,10,'a')),annotation((0,10,'b')))
        d['episodes'].append(dict(id='R002',n_frames=2,versions={'atomic_actions':dict(
            openai=annotation((0,2,'c')),gemini=annotation((0,2,'d')))}))
        row=summarize(d,scores(('a','b',.8),('c','d',.2)))['runs'][0]['overall']
        self.assertAlmostEqual(row['mean_all_frames'],.7)

    def test_zero_is_a_valid_score_and_uncertain_frames_are_unscored(self):
        left=annotation((0,10,'a'))
        left['uncertain_intervals']=[dict(start_frame=2,end_frame=5)]
        row=summarize(dataset(left,annotation((0,10,'b'))),scores(('a','b',0)))['runs'][0]['overall']
        self.assertEqual(row['scored_frames'],7)
        self.assertEqual(row['mean_scored_frames'],0)
        self.assertIsNone(row['mean_all_frames'])

    def test_stale_batch_and_overlapping_segments_are_rejected(self):
        d=dataset(annotation((0,10,'a')),annotation((0,10,'b')))
        cached=scores(('a','b',.8))
        cached['annotation_manifest_sha256']='old'
        with self.assertRaisesRegex(ValueError,'different annotation batch'):
            summarize(d,cached)
        d['episodes'][0]['versions']['atomic_actions']['openai']=annotation((0,8,'a'),(5,10,'a'))
        with self.assertRaisesRegex(ValueError,'Overlapping'):
            summarize(d,scores(('a','b',.8)))


if __name__ == '__main__':
    unittest.main()
