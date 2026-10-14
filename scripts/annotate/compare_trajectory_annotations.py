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


def compare(run_dir, out, right_run=None, left='sol', right='astra'):
    manifest = common.load(run_dir / 'manifest.json')
    scheme = manifest['settings'].get('scheme', 'atomic')
    runner.configure(run_dir, scheme=scheme)
    models = manifest['settings']['models']
    sources = {key: (run_dir, manifest) for key in models}
    if right_run:
        folders, saved = common.comparison_sources(run_dir, left, right, right_run)
        models = {left: saved[0]['settings']['models'][left], right: saved[1]['settings']['models'][right]}
        sources = dict(zip((left, right), zip(folders, saved)))
    rows, unavailable = [], []
    for ep in manifest['episodes']:
        predictions = {}
        for model in models:
            folder, saved = sources[model]
            stem = ep['id'] + '_' + model
            slot = folder / 'attempts' / (stem + '.json')
            answer = folder / (stem + '_annotation.json')
            record = common.load(slot) if slot.exists() else {}
            if record.get('status') != 'validated' or not answer.exists():
                unavailable.append(dict(episode=ep['id'], model=model,
                                        status=record.get('status', 'missing')))
                continue
            value = common.load(answer)
            if scheme == 'atomic':
                runner.validate_atomic(value, ep, label_mode=saved['settings'].get('label_mode', 'progress_failure'))
            else:
                runner.validate(value, ep)
            predictions[model] = value
        for left, right in itertools.combinations(predictions, 2):
            rows.append(dict(episode=ep['id'], left=left, right=right,
                             metrics=evaluate(segments(predictions[left]),
                                              segments(predictions[right]),
                                              ep['n_frames'], ep['fps'],
                                              require_verified=False, label_mode='failure_binary')))
    result = dict(kind='model_agreement', label_mode='failure_binary', limitations='Agreement does not establish video correctness.',
                  manifest_sha256=common.digest(run_dir / 'manifest.json'),
                  source_manifest_sha256={key: common.digest(folder / 'manifest.json')
                                          for key, (folder, _) in sources.items()},
                  expected_outputs=len(manifest['episodes']) * len(models),
                  unavailable=unavailable, comparisons=rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    common.atomic_json(out, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--right-run', type=Path)
    parser.add_argument('--left', default='sol')
    parser.add_argument('--right', default='astra')
    args = parser.parse_args()
    result = compare(args.run.resolve(), args.out.resolve(), args.right_run, args.left, args.right)
    print(f"Compared {len(result['comparisons'])} episode/model pairs; "
          f"{len(result['unavailable'])} outputs unavailable. Saved {args.out}")
