"""Review a saved run against original and supplied camera frames, without API calls."""
import argparse
import hashlib
import json
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import shutil
import webbrowser

import annotation_runtime as common
import annotate_trajectories as runner


def build(run_dir, source, left='sol', right='astra', right_run=None):
    import numpy as np
    from PIL import Image
    folders, manifests = common.comparison_sources(run_dir, left, right, right_run)
    manifest, right_manifest = manifests
    runner.configure(run_dir, scheme=manifest['settings'].get('scheme', 'atomic'))
    models = {left: manifest['settings']['models'][left], right: right_manifest['settings']['models'][right]}
    if left == right:
        raise ValueError('Choose two distinct model keys from the run manifest')
    # Verify the saved inputs even when inspecting an older runner version.
    for name, digest in manifest['files'].items():
        path = (run_dir / name).resolve()
        if not path.is_relative_to(run_dir) or common.digest(path) != digest:
            raise ValueError('Frozen evidence changed')
    target = run_dir / 'current_review'
    target.mkdir(exist_ok=True)
    shutil.copyfile(run_dir / 'prompt.txt', target / 'annotation_prompt.txt')
    if right_run:
        shutil.copyfile(folders[1] / 'prompt.txt', target / 'right_annotation_prompt.txt')
    episodes = []
    for ep in manifest['episodes']:
        eid = ep['id']
        if not eid.replace('_', '').replace('-', '').isalnum():
            raise ValueError('Invalid episode identifier')
        folder = target / 'frames' / eid
        folder.mkdir(parents=True, exist_ok=True)
        for name, digest in ep['source_sha256'].items():
            if common.digest(source / eid / name) != digest:
                raise ValueError('Original source evidence changed')
        with np.load(source / eid / 'cameras.npz', allow_pickle=False) as cameras:
            for camera in ('agent', 'wrist'):
                if len(cameras[camera]) != ep['n_frames']:
                    raise ValueError('Camera length differs')
                for i, frame in enumerate(cameras[camera]):
                    destination = folder / f'{i:04d}_{camera}.jpg'
                    Image.fromarray(frame).save(destination, quality=95)
        sampled = target / 'sampled' / eid
        sampled.mkdir(parents=True, exist_ok=True)
        for frame in ep['sampled_frames']:
            for camera in ('agent', 'wrist'):
                name = f'{frame:04d}_{camera}.jpg'
                shutil.copyfile(run_dir / eid / name, sampled / name)
        annotations, statuses, evidence = {}, {}, {}
        for (column, model), folder, saved in zip((('openai', left), ('gemini', right)), folders, manifests):
            stem = eid + '_' + model
            slot = folder / 'attempts' / (stem + '.json')
            record = common.load(slot) if slot.exists() else {}
            path = folder / (stem + '_annotation.json')
            value = common.load(path) if path.exists() and record.get('status') == 'validated' else None
            if value is not None:
                if runner.SCHEME == 'atomic':
                    runner.validate_atomic(value, ep, label_mode=saved['settings'].get('label_mode', 'progress_failure'))
                else:
                    runner.validate(value, ep)
            answer = folder / (stem + '_answer.txt')
            link = stem + '_answer.txt' if answer.exists() else None
            if link:
                shutil.copyfile(answer, target / link)
            annotations[column] = value
            statuses[column] = dict(status=record.get('status', 'pending'),
                                    error=record.get('error', ''), raw_link=link,
                                    display_note=f"Source: {folder.name}; reasoning: {saved['settings'].get('reasoning', 'low')}" if right_run else '')
            evidence[column] = dict(frames=ep['sampled_frames'], initial_frames=ep['sampled_frames'],
                                    prefix='sampled', calls=int(record.get('generation_sent', False)),
                                    cost=record.get('generation_estimate_usd'),
                                    allowance=record.get('preflight_allowance_usd', 0), requests=[])
        episodes.append(dict(ep, versions={'atomic_actions': annotations},
                             statuses={'atomic_actions': statuses},
                             evidence_versions={'atomic_actions': evidence}))
    choice = dict(id='atomic_actions', label=f'Failure/non-failure ({run_dir.name})',
                  heading=f'{models[left]} vs {models[right]}',
                  columns={'openai': models[left], 'gemini': models[right]},
                  prompt_file='annotation_prompt.txt')
    if right_run:
        same_prompt = common.digest(folders[0] / 'prompt.txt') == common.digest(folders[1] / 'prompt.txt')
        choice.update(heading=(f'{models[left]} vs {models[right]} (same frozen prompt)' if same_prompt
                               else f'{models[left]} (new) vs {models[right]} (saved baseline)'),
                      prompt_label=f'Read {left.title()} prompt', right_prompt_file='right_annotation_prompt.txt',
                      right_prompt_label=f'Read {right.title()} prompt' if same_prompt else f'Read saved {right.title()} prompt')
    if runner.SCHEME == 'atomic':
        from compare_action_groups import run as compare_actions
        agreement = compare_actions(run_dir, left, right, target / 'action_group_agreement.json', right_run)
        choice['action_agreement_file'] = 'action_group_agreement.html'
        scores = {row['id']: row for row in agreement['episodes']}
        for episode in episodes:
            row = scores.get(episode['id'])
            if row is not None:
                episode['action_agreement_versions'] = {'atomic_actions': dict(
                    scores=row['scores'], counts=row['counts'])}
    common.write_json(target / 'data.json', dict(episodes=episodes, runs=[choice], default_run='atomic_actions',
                                               display_label_mode='failure_binary',
                                               annotation_manifest_sha256=common.digest(run_dir / 'manifest.json'),
                                               annotation_sources={column: dict(run=str(folder),
                                                   manifest_sha256=common.digest(folder / 'manifest.json'),
                                                   prompt_sha256=common.digest(folder / 'prompt.txt'))
                                                   for column, folder in zip(('openai', 'gemini'), folders)}))
    shutil.copyfile(Path(__file__).with_suffix('.html'), target / 'index.html')
    return target


def add_supplement(target, run_dir):
    """Fill missing viewer outputs from verified same-prompt retry runs; originals remain frozen."""
    run_dir = run_dir.resolve()
    manifest = common.load(run_dir / 'manifest.json')
    data = common.load(target / 'data.json')
    active = next(r for r in data['runs'] if r['id'] == data['default_run'])
    if len(manifest['settings']['models']) != 1 or manifest['settings'].get('scheme') != 'atomic':
        raise ValueError('A supplement must be a single-model atomic run')
    model_key, model_name = next(iter(manifest['settings']['models'].items()))
    columns = [k for k, v in active['columns'].items() if v == model_name]
    if len(columns) != 1:
        raise ValueError('Supplement model does not match a viewer column')
    column = columns[0]
    parent = data['annotation_sources'][column]
    baseline = common.load(Path(parent['run']) / 'manifest.json')
    if common.digest(run_dir / 'prompt.txt') != parent['prompt_sha256']:
        raise ValueError('Supplement prompt differs from the source comparison')
    for field in ('scheme', 'label_mode', 'reasoning', 'image_detail', 'sample_stride',
                  'max_input_tokens', 'max_output_tokens'):
        if manifest['settings'].get(field) != baseline['settings'].get(field):
            raise ValueError('Supplement annotation settings differ')
    for name, digest in manifest['files'].items():
        path = (run_dir / name).resolve()
        if not path.is_relative_to(run_dir) or common.digest(path) != digest:
            raise ValueError('Frozen supplement evidence changed')
        if baseline['files'].get(name) != digest:
            raise ValueError('Supplement input bytes differ from original evidence')
    episodes = {e['id']: e for e in data['episodes']}
    pending = []
    for observed in manifest['episodes']:
        ep = episodes.get(observed['id'])
        if ep is None or any(ep.get(k) != v for k, v in observed.items()):
            raise ValueError('Supplement episode evidence differs')
        if ep['versions'][data['default_run']].get(column) is not None:
            raise ValueError('A supplement can only fill missing outputs')
        stem = observed['id'] + '_' + model_key
        record = common.load(run_dir / 'attempts' / (stem + '.json'))
        if record.get('status') != 'validated':
            raise ValueError('Supplement output is not validated')
        value = common.load(run_dir / (stem + '_annotation.json'))
        runner.validate_atomic(value, observed, label_mode=manifest['settings']['label_mode'])
        pending.append((ep, stem, record, value))
    resources = target / 'supplements' / run_dir.name
    resources.mkdir(parents=True, exist_ok=True)
    for ep, stem, record, value in pending:
        raw = stem + '_answer.txt'
        shutil.copyfile(run_dir / raw, resources / raw)
        version = data['default_run']
        ep['versions'][version][column] = value
        ep['statuses'][version][column] = dict(status='validated', error='',
            raw_link=(resources / raw).relative_to(target).as_posix(),
            display_note=f"Source: {run_dir.name}; reasoning: {manifest['settings']['reasoning']}; retry")
        ep['evidence_versions'][version][column].update(calls=int(record.get('generation_sent', False)),
            cost=record.get('generation_estimate_usd'), allowance=record.get('preflight_allowance_usd', 0))
        data.setdefault('episode_sources', {}).setdefault(ep['id'], {})[column] = dict(
            run=str(run_dir), manifest_sha256=common.digest(run_dir / 'manifest.json'),
            prompt_sha256=common.digest(run_dir / 'prompt.txt'),
            annotation_sha256=common.digest(run_dir / (stem + '_annotation.json')))
    data['annotation_manifest_sha256'] = hashlib.sha256(json.dumps(dict(
        sources=data['annotation_sources'], retries=data['episode_sources'],
        annotations=[e['versions'] for e in data['episodes']]), sort_keys=True).encode()).hexdigest()
    common.write_json(target / 'data.json', data)


def refresh_comparison_metrics(target):
    """Score the displayed validated outputs, including separately sourced retries."""
    from compare_action_groups import compare, render
    from compare_trajectory_annotations import segments
    from trajectory_metrics import evaluate
    data = common.load(target / 'data.json')
    version = data['default_run']
    report = common.load(target / 'action_group_agreement.json')
    report.update(episodes=[], unavailable=[], annotation_sha256={},
                  annotation_sources=data['annotation_sources'], episode_sources=data.get('episode_sources', {}))
    frame_rows = []
    for ep in data['episodes']:
        annotations = ep['versions'][version]
        for column in ('openai', 'gemini'):
            value = annotations.get(column)
            if value is None:
                report['unavailable'].append(dict(episode=ep['id'], column=column))
            else:
                report['annotation_sha256'][ep['id'] + '/' + column] = hashlib.sha256(
                    json.dumps(value, sort_keys=True).encode()).hexdigest()
        a, b = annotations.get('openai'), annotations.get('gemini')
        if a is None or b is None:
            ep.setdefault('action_agreement_versions', {}).pop(version, None)
            continue
        row = dict(id=ep['id'], task=ep['task'], fps=ep['fps'], n_frames=ep['n_frames'], **compare(a, b))
        report['episodes'].append(row)
        ep.setdefault('action_agreement_versions', {})[version] = dict(scores=row['scores'], counts=row['counts'])
        frame_rows.append(dict(episode=ep['id'], metrics=evaluate(segments(a), segments(b),
            ep['n_frames'], ep['fps'], require_verified=False, label_mode='failure_binary')))
    report['comparison_sha256'] = data['annotation_manifest_sha256']
    common.write_json(target / 'action_group_agreement.json', report)
    render(report, target / 'action_group_agreement.html')
    common.write_json(target / 'failure_binary_agreement.json', dict(kind='model_agreement',
        comparisons=frame_rows, unavailable=report['unavailable'], comparison_sha256=data['annotation_manifest_sha256'],
        annotation_sources=data['annotation_sources'], episode_sources=data.get('episode_sources', {}),
        limitations='Agreement does not establish video correctness. Retry outputs retain separate source runs.'))
    common.write_json(target / 'data.json', data)


def add_variant(target, variant, variant_id):
    """Add a verified review bundle as a selectable candidate without copying labels into a run."""
    data = common.load(target / 'data.json')
    other = common.load(variant / 'data.json')
    if any(r['id'] == variant_id for r in data['runs']):
        raise ValueError('Duplicate viewer variant ID')
    base_episodes = {e['id']: e for e in data['episodes']}
    if set(base_episodes) != {e['id'] for e in other['episodes']}:
        raise ValueError('Viewer variants need matching episodes')
    for episode in other['episodes']:
        base = base_episodes[episode['id']]
        for field in ('task', 'n_frames', 'fps', 'sampled_frames', 'source_sha256', 'telemetry'):
            if base[field] != episode[field]:
                raise ValueError('Viewer variant evidence differs')
    resources = target / 'variants' / variant_id
    resources.mkdir(parents=True, exist_ok=True)
    for path in variant.glob('*.txt'):
        shutil.copyfile(path, resources / path.name)
    for name in ('action_group_agreement.html', 'action_group_agreement.json'):
        if (variant / name).exists():
            shutil.copyfile(variant / name, resources / name)
    prefix = resources.relative_to(target).as_posix() + '/'
    choice = dict(other['runs'][0], id=variant_id, annotation_sources=other.get('annotation_sources'))
    for field in ('prompt_file', 'right_prompt_file'):
        if choice.get(field):
            choice[field] = prefix + choice[field]
    choice['action_agreement_file'] = (prefix + 'action_group_agreement.html'
                                      if (resources / 'action_group_agreement.html').exists() else None)
    data['runs'].append(choice)
    for episode in other['episodes']:
        base = base_episodes[episode['id']]
        for field in ('versions', 'statuses', 'evidence_versions', 'action_agreement_versions'):
            if other['default_run'] in episode.get(field, {}):
                base.setdefault(field, {})[variant_id] = episode[field][other['default_run']]
        for status in base['statuses'][variant_id].values():
            if status.get('raw_link'):
                status['raw_link'] = prefix + status['raw_link']
    data.setdefault('comparison_manifest_sha256', [data['annotation_manifest_sha256']]).append(
        other['annotation_manifest_sha256'])
    fingerprint = dict(sources=data['comparison_manifest_sha256'],
                       annotation_sources=[r.get('annotation_sources', data.get('annotation_sources')) for r in data['runs']])
    data['annotation_manifest_sha256'] = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
    common.write_json(target / 'data.json', data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=common.SOURCE)
    parser.add_argument('--left', default='sol')
    parser.add_argument('--right', default='astra')
    parser.add_argument('--right-run', type=Path, help='Reuse a saved comparison baseline without API calls')
    parser.add_argument('--alternate-run', type=Path, action='append', default=[],
                        help='Add another saved candidate as a selectable viewer version')
    parser.add_argument('--supplement-run', type=Path, action='append', default=[],
                        help='Fill missing outputs from a verified same-prompt single-model retry')
    parser.add_argument('--port', type=int, default=8767)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    folder = build(args.run.resolve(), args.source.resolve(), args.left, args.right, args.right_run)
    for index, alternate in enumerate(args.alternate_run, 1):
        variant = build(alternate.resolve(), args.source.resolve(), args.left, args.right, args.right_run)
        add_variant(folder, variant, f'variant_{index}')
    for supplement in args.supplement_run:
        add_supplement(folder, supplement)
    if args.supplement_run:
        refresh_comparison_metrics(folder)
    print(f'Prepared {folder}', flush=True)
    if not args.prepare_only:
        server = ThreadingHTTPServer(('127.0.0.1', args.port),
                    partial(SimpleHTTPRequestHandler, directory=str(folder)))
        url = f'http://127.0.0.1:{args.port}/'
        print(url, flush=True)
        if not args.no_browser:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
