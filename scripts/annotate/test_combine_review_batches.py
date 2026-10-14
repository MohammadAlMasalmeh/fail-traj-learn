from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import annotation_runtime as io
from combine_review_batches import combine,verify_bundle
from test_annotate_trajectories import EP,annotation,segment


def bundle(base,eid,prompt='frozen shared prompt'):
    run=base/(eid+'_run');run.mkdir();(run/'attempts').mkdir()
    ep=dict(EP,id=eid,task='Move the object',fps=20,source_sha256={},telemetry=[])
    settings=dict(scheme='atomic',label_mode='failure_binary',models={'sol':'sol-model','astra':'astra-model'},
                  reasoning='low',image_detail='high',sample_stride=5,max_input_tokens=80000,max_output_tokens=12288)
    (run/'prompt.txt').write_text(prompt)
    io.write_json(run/'manifest.json',dict(settings=settings,episodes=[ep],files={'prompt.txt':io.digest(run/'prompt.txt')}))
    review=base/(eid+'_review');review.mkdir()
    for prefix in ('frames','sampled'):
        folder=review/prefix/eid;folder.mkdir(parents=True)
        (folder/'0000_agent.jpg').write_bytes(b'fixture-image')
    (review/'annotation_prompt.txt').write_text(prompt)
    values={};statuses={};evidence={};sources={}
    for col,key in (('openai','sol'),('gemini','astra')):
        value=annotation([segment(0,20,'a1','transport')]);value['episode_id']=eid
        stem=eid+'_'+key;answer=json.dumps(value)
        (run/(stem+'_answer.txt')).write_text(answer);(review/(stem+'_answer.txt')).write_text(answer)
        io.write_json(run/(stem+'_annotation.json'),value)
        io.write_json(run/'attempts'/(stem+'.json'),{'status':'validated'})
        values[col]=value;statuses[col]=dict(status='validated',raw_link=stem+'_answer.txt')
        evidence[col]=dict(prefix='sampled',frames=ep['sampled_frames'])
        sources[col]=dict(run=str(run),manifest_sha256=io.digest(run/'manifest.json'),prompt_sha256=io.digest(run/'prompt.txt'))
    choice=dict(id='atomic_actions',columns={'openai':'sol-model','gemini':'astra-model'},prompt_file='annotation_prompt.txt',heading='Sol vs Astra')
    io.write_json(review/'data.json',dict(default_run='atomic_actions',display_label_mode='failure_binary',runs=[choice],
        annotation_sources=sources,episodes=[dict(ep,versions={'atomic_actions':values},statuses={'atomic_actions':statuses},
            evidence_versions={'atomic_actions':evidence})]))
    io.write_json(review/'action_group_agreement.json',dict(left_model='sol-model',right_model='astra-model',definition='Test definition',limitations='Test limitations'))
    return review,run


class CombinedReviewTests(unittest.TestCase):
    def test_combines_disjoint_episodes_without_altering_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);a,ra=bundle(base,'R001');b,rb=bundle(base,'R005')
            before={p:io.digest(p/'data.json') for p in (a,b)}
            out=combine([a,b],base/'combined')
            data=io.load(out/'data.json')
            self.assertEqual([e['id'] for e in data['episodes']],['R001','R005'])
            self.assertEqual(io.load(out/'review_manifest.json')['paid_api_calls'],0)
            self.assertEqual(data['episode_sources']['R005']['openai']['run'],str(rb))
            self.assertEqual(len(io.load(out/'failure_binary_agreement.json')['comparisons']),2)
            for p,digest in before.items():self.assertEqual(io.digest(p/'data.json'),digest)
            for ep in data['episodes']:
                for col in ('openai','gemini'):
                    self.assertTrue((out/ep['statuses']['atomic_actions'][col]['raw_link']).is_file())

    def test_rejects_different_protocols_and_duplicate_episodes_before_copying(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);a,_=bundle(base,'R001');b,_=bundle(base,'R005',prompt='different prompt')
            with self.assertRaisesRegex(ValueError,'different prompts'):combine([a,b],base/'out')
            self.assertFalse((base/'out').exists())
            with self.assertRaisesRegex(ValueError,'Duplicate'):combine([a,a],base/'out')
            self.assertFalse((base/'out').exists())

    def test_rejects_edited_displayed_annotations_and_unsafe_answer_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);a,_=bundle(base,'R001');data=io.load(a/'data.json');original=deepcopy(data)
            data['episodes'][0]['versions']['atomic_actions']['openai']['segments'][0]['what_happened']='invented result'
            io.write_json(a/'data.json',data)
            with self.assertRaisesRegex(ValueError,'differs from saved output'):verify_bundle(a)
            original['episodes'][0]['statuses']['atomic_actions']['openai']['raw_link']='../outside.txt'
            (base/'outside.txt').write_text('outside')
            io.write_json(a/'data.json',original)
            with self.assertRaisesRegex(ValueError,'unsafe'):verify_bundle(a)


if __name__=='__main__':unittest.main()
