"""Offline agreement of saved annotations; model agreement is not accuracy."""
import argparse
import itertools
from pathlib import Path

import annotation_runtime as common
import annotate_trajectories as runner
from trajectory_metrics import evaluate


def segments(annotation):
    return [dict(s, start=s['start_frame'], end=s['end_frame'],
                 description=s['what_happened']) for s in annotation['segments']]


def compare(run_dir, out):
    manifest = common.load(run_dir / 'manifest.json')
    scheme = manifest['settings'].get('scheme', 'atomic')
    runner.configure(run_dir, scheme=scheme)
    models = manifest['settings']['models']
    rows, unavailable = [], []
    for ep in manifest['episodes']:
        predictions = {}
        for model in models:
            stem = ep['id'] + '_' + model
            slot = run_dir / 'attempts' / (stem + '.json')
            answer = run_dir / (stem + '_annotation.json')
            record = common.load(slot) if slot.exists() else {}
            if record.get('status') != 'validated' or not answer.exists():
                unavailable.append(dict(episode=ep['id'], model=model,
                                        status=record.get('status', 'missing')))
                continue
            value = common.load(answer)
            runner.validate(value, ep)
            predictions[model] = value
        for left, right in itertools.combinations(predictions, 2):
            rows.append(dict(episode=ep['id'], left=left, right=right,
                             metrics=evaluate(segments(predictions[left]),
                                              segments(predictions[right]),
                                              ep['n_frames'], ep['fps'],
                                              require_verified=False)))
    result = dict(kind='model_agreement', limitations='Agreement does not establish video correctness.',
                  manifest_sha256=common.digest(run_dir / 'manifest.json'),
                  expected_outputs=len(manifest['episodes']) * len(models),
                  unavailable=unavailable, comparisons=rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    common.atomic_json(out, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.run.resolve(), args.out.resolve())
    print(f"Compared {len(result['comparisons'])} episode/model pairs; "
          f"{len(result['unavailable'])} outputs unavailable. Saved {args.out}")
