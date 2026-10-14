"""Exercise new-run freezing, state coverage, credentials, and saved-output scoring."""
from copy import deepcopy
from email.message import Message
from io import BytesIO
import json
import os
from pathlib import Path
import tempfile
import unittest
import urllib.error
from unittest.mock import patch

import numpy as np

import annotation_runtime as runtime
import annotation_states
import annotate_trajectories as runner
from compare_trajectory_annotations import compare
from test_annotate_trajectories import EP, annotation, segment


class WorkflowTests(unittest.TestCase):
    def test_http_error_preserves_billing_code_and_request_id_without_retry_or_secret(self):
        headers = Message()
        headers['x-request-id'] = 'request-for-support'
        body = json.dumps({'error': {'code': 'credit_balance_exhausted',
                                    'type': 'insufficient_quota',
                                    'message': 'Rejected test-secret-key'}}).encode()
        failure = urllib.error.HTTPError('https://api.openai.com/v1/responses', 429,
                                        'Too Many Requests', headers, BytesIO(body))
        with patch.object(runtime.urllib.request, 'urlopen', side_effect=failure) as send:
            with self.assertRaises(runtime.APIHTTPError) as caught:
                runtime.post('https://api.openai.com/v1/responses', 'openai', {},
                             {'openai': 'test-secret-key'})
            self.assertEqual(send.call_count, 1)
        details = caught.exception.details
        self.assertEqual(details['status'], 429)
        self.assertEqual(details['code'], 'credit_balance_exhausted')
        self.assertEqual(details['type'], 'insufficient_quota')
        self.assertEqual(details['request_id'], 'request-for-support')
        self.assertNotIn('test-secret-key', str(caught.exception))
        self.assertNotIn('test-secret-key', json.dumps(details))

    def test_submission_records_structured_http_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / 'attempts').mkdir()
            failure = runtime.APIHTTPError(dict(provider='openai', status=429,
                code='project_spend_limit_exceeded', type='insufficient_quota',
                request_id='request-for-support', message='Project limit reached'))
            payload = dict(model='gpt-6.1-sol', input=[], reasoning={'effort': 'low'},
                           text={}, truncation='disabled')
            with patch.object(runner, 'OUT', folder), patch.object(runner, 'payload', return_value=payload), \
                 patch.object(runtime, 'post', side_effect=failure):
                runner.execute('sol', EP, {'openai': 'test-key'})
            record = runtime.load(folder / 'attempts/test_sol.json')
            self.assertEqual(record['status'], 'stopped')
            self.assertEqual(record['error_details']['code'], 'project_spend_limit_exceeded')
            self.assertEqual(record['error_details']['request_id'], 'request-for-support')

    def test_saved_baseline_requires_matching_evidence_and_keeps_separate_prompts(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            folders = [base / name for name in ('new', 'baseline')]
            for folder, model, prompt in zip(folders, ('sol', 'astra'), ('new prompt', 'old prompt')):
                folder.mkdir()
                (folder / 'prompt.txt').write_text(prompt)
                (folder / 'image.jpg').write_bytes(b'same image')
                runtime.write_json(folder / 'manifest.json', dict(
                    settings=dict(models={model: model}), episodes=[EP],
                    files={name: runtime.digest(folder / name) for name in ('prompt.txt', 'image.jpg')}))
            _, saved = runtime.comparison_sources(folders[0], 'sol', 'astra', folders[1])
            self.assertNotEqual(saved[0]['files']['prompt.txt'], saved[1]['files']['prompt.txt'])
            (folders[1] / 'image.jpg').write_bytes(b'changed image')
            with self.assertRaisesRegex(ValueError, 'Frozen evidence'):
                runtime.comparison_sources(folders[0], 'sol', 'astra', folders[1])
            manifest = runtime.load(folders[1] / 'manifest.json')
            manifest['files']['image.jpg'] = runtime.digest(folders[1] / 'image.jpg')
            runtime.write_json(folders[1] / 'manifest.json', manifest)
            with self.assertRaisesRegex(ValueError, 'identical frozen'):
                runtime.comparison_sources(folders[0], 'sol', 'astra', folders[1])

    def test_sol_only_run_sends_no_astra_requests_and_freezes_reasoning(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            ep = base / 'source/E001'
            ep.mkdir(parents=True)
            np.savez(ep / 'cameras.npz', agent=np.zeros((6, 3, 3, 3), dtype=np.uint8),
                     wrist=np.zeros((6, 3, 3, 3), dtype=np.uint8))
            runtime.write_json(ep / 'observable.json', dict(
                id='E001', task='Move the object', n_frames=6, fps=20,
                telemetry=[dict(frame=i, eef_xyz_m=[0, 0, 0], finger_qpos_m=[0, 0],
                                gripper_command='open') for i in range(6)]))
            value = annotation([segment(0, 6, 'a1', 'reach')])
            value['episode_id'] = 'E001'
            raw = dict(status='completed', model='gpt-6.1-sol',
                       output=[dict(content=[dict(type='output_text', text=json.dumps(value))])],
                       usage=dict(input_tokens=100, output_tokens=20))
            with patch.object(runner, 'OUT'), patch.object(runner, 'SOURCE'), \
                 patch.object(runner, 'IDS'), patch.object(runner, 'SCHEME'), \
                 patch.object(runner, 'LABEL_MODE'), patch.object(runner, 'PROMPT_DIR'), \
                 patch.object(runner, 'MODELS'), patch.object(runner, 'REASONING'):
                runner.configure(base / 'run', base / 'source', ['E001'], models=['sol'], reasoning='high')
                runner.prepare()
                self.assertEqual(runner.verify()['settings']['maximum_reserved_usd'], .5)
                runner.configure(base / 'run')
                self.assertEqual(runner.MODELS, {'sol': 'gpt-6.1-sol'})
                self.assertEqual(runner.REASONING, 'high')
                with self.assertRaisesRegex(ValueError, 'Model selection'):
                    runner.configure(base / 'run', models=['astra'])
                with patch.object(runtime, 'credentials', return_value={'openai': 'test-key'}), \
                     patch.object(runtime, 'post', side_effect=[{'input_tokens': 100}, raw]) as post:
                    self.assertTrue(runner.run())
                    self.assertEqual(post.call_count, 2)
                    self.assertTrue(all(call.args[2]['model'] == 'gpt-6.1-sol' for call in post.call_args_list))
                    self.assertEqual(post.call_args_list[-1].args[2]['reasoning'], {'effort': 'high'})
                self.assertFalse((base / 'run/attempts/E001_astra.json').exists())

    def test_state_segments_merge_identical_labels_and_preserve_category_changes(self):
        value = annotation([segment(0, 10, 'a1', 'transport'),
                            segment(10, 20, 'a2', 'transport', True)])
        for s in value['segments']:
            s['progress'] = True
            for k in ('action_id', 'action', 'object_id', 'destination_id',
                      'observed_result', 'boundary_reason'):
                del s[k]
        annotation_states.validate(value, EP)
        invalid = deepcopy(value)
        invalid['segments'][1].update(progress=True, failure=False, categories=[],
                                      recovery=None, severity=None, resetability=None)
        with self.assertRaisesRegex(ValueError, 'must be merged'):
            annotation_states.validate(invalid, EP)
        invalid = deepcopy(value)
        invalid['episode_id'] = 'different'
        with self.assertRaises(ValueError):
            annotation_states.validate(invalid, EP)

    def test_prepare_freezes_only_observable_evidence_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = base / 'source'
            ep = source / 'E001'
            ep.mkdir(parents=True)
            np.savez(ep / 'cameras.npz', agent=np.zeros((6, 3, 3, 3), dtype=np.uint8),
                     wrist=np.zeros((6, 3, 3, 3), dtype=np.uint8))
            runtime.write_json(ep / 'observable.json', dict(
                id='E001', task='Move the object', n_frames=6, fps=20,
                episode_outcome='withheld', telemetry=[dict(
                    frame=i, eef_xyz_m=[0, 0, 0], finger_qpos_m=[0, 0],
                    gripper_command='open', privileged_contact=True) for i in range(6)]))
            out = base / 'run'
            with patch.object(runner, 'OUT', out), patch.object(runner, 'SOURCE', source), \
                 patch.object(runner, 'IDS', ('E001',)):
                runner.prepare()
                manifest = runner.verify()
                self.assertEqual(manifest['episodes'][0]['sampled_frames'], [0, 5])
                self.assertEqual(manifest['settings']['maximum_reserved_usd'], 3)
                self.assertEqual(manifest['settings']['label_mode'], 'failure_binary')
                self.assertNotIn('privileged_contact', json.dumps(manifest['episodes']))
                self.assertNotIn('episode_outcome', manifest['episodes'][0])
                with self.assertRaises(ValueError):
                    runner.prepare()
                (out / 'E001/0000_agent.jpg').write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'evidence changed'):
                    runner.verify()

    def test_openai_only_environment_does_not_require_gemini(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(runtime, 'ROOT', Path(tmp)), \
             patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}, clear=True):
            self.assertEqual(runtime.credentials(), {'openai': 'test-key'})

    def test_missing_output_is_reported_and_categories_are_scored(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / 'attempts').mkdir()
            ep = dict(EP, fps=20)
            value = annotation([segment(0, 20, 'a1', 'transport', True)])
            runtime.write_json(out / 'manifest.json', dict(
                settings=dict(scheme='atomic', label_mode='failure_binary', models=dict(sol='sol', astra='astra')),
                episodes=[ep]))
            runtime.write_json(out / 'test_sol_annotation.json', value)
            runtime.write_json(out / 'attempts/test_sol.json', dict(status='validated'))
            # configure changes module globals; retain the prior configuration.
            with patch.object(runner, 'OUT'), patch.object(runner, 'IDS'), \
                 patch.object(runner, 'SCHEME'), patch.object(runner, 'LABEL_MODE'), patch.object(runner, 'PROMPT_DIR'), \
                 patch.object(runner, 'MODELS'), patch.object(runner, 'REASONING'):
                result = compare(out, out / 'agreement.json')
                self.assertEqual(len(result['unavailable']), 1)
                self.assertEqual(result['comparisons'], [])
                runtime.write_json(out / 'test_astra_annotation.json', value)
                runtime.write_json(out / 'attempts/test_astra.json', dict(status='validated'))
                result = compare(out, out / 'agreement.json')
                self.assertEqual(result['comparisons'][0]['metrics']['failure_types']['collision']['f1'], 1)


if __name__ == '__main__':
    unittest.main()
