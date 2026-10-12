"""Exercise new-run freezing, state coverage, credentials, and saved-output scoring."""
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import annotation_runtime as runtime
import annotation_states
import annotate_trajectories as runner
from compare_trajectory_annotations import compare
from test_annotate_trajectories import EP, annotation, segment


class WorkflowTests(unittest.TestCase):
    def test_state_segments_merge_identical_labels_and_preserve_category_changes(self):
        value = annotation([segment(0, 10, 'a1', 'transport'),
                            segment(10, 20, 'a2', 'transport', True)])
        for s in value['segments']:
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
                settings=dict(scheme='atomic', models=dict(sol='sol', astra='astra')),
                episodes=[ep]))
            runtime.write_json(out / 'test_sol_annotation.json', value)
            runtime.write_json(out / 'attempts/test_sol.json', dict(status='validated'))
            # configure changes module globals; retain the prior configuration.
            with patch.object(runner, 'OUT'), patch.object(runner, 'IDS'), \
                 patch.object(runner, 'SCHEME'), patch.object(runner, 'PROMPT_DIR'):
                result = compare(out, out / 'agreement.json')
                self.assertEqual(len(result['unavailable']), 1)
                self.assertEqual(result['comparisons'], [])
                runtime.write_json(out / 'test_astra_annotation.json', value)
                runtime.write_json(out / 'attempts/test_astra.json', dict(status='validated'))
                result = compare(out, out / 'agreement.json')
                self.assertEqual(result['comparisons'][0]['metrics']['failure_types']['collision']['f1'], 1)


if __name__ == '__main__':
    unittest.main()
