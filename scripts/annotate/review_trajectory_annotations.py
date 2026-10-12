"""Review a saved run against original and supplied camera frames, without API calls."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import shutil
import webbrowser

import annotation_runtime as common
import annotate_trajectories as runner


def build(run_dir, source, left='sol', right='astra'):
    import numpy as np
    from PIL import Image
    manifest = common.load(run_dir / 'manifest.json')
    runner.configure(run_dir, scheme=manifest['settings'].get('scheme', 'atomic'))
    models = manifest['settings']['models']
    if left == right or any(k not in models for k in (left, right)):
        raise ValueError('Choose two distinct model keys from the run manifest')
    # Verify the saved inputs even when inspecting an older runner version.
    for name, digest in manifest['files'].items():
        path = (run_dir / name).resolve()
        if not path.is_relative_to(run_dir) or common.digest(path) != digest:
            raise ValueError('Frozen evidence changed')
    target = run_dir / 'current_review'
    target.mkdir(exist_ok=True)
    shutil.copyfile(run_dir / 'prompt.txt', target / 'annotation_prompt.txt')
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
        for column, model in (('openai', left), ('gemini', right)):
            stem = eid + '_' + model
            slot = run_dir / 'attempts' / (stem + '.json')
            record = common.load(slot) if slot.exists() else {}
            path = run_dir / (stem + '_annotation.json')
            value = common.load(path) if path.exists() and record.get('status') == 'validated' else None
            if value is not None:
                runner.validate(value, ep)
            answer = run_dir / (stem + '_answer.txt')
            link = stem + '_answer.txt' if answer.exists() else None
            if link:
                shutil.copyfile(answer, target / link)
            annotations[column] = value
            statuses[column] = dict(status=record.get('status', 'pending'),
                                    error=record.get('error', ''), raw_link=link, display_note='')
            evidence[column] = dict(frames=ep['sampled_frames'], initial_frames=ep['sampled_frames'],
                                    prefix='sampled', calls=int(record.get('generation_sent', False)),
                                    cost=record.get('generation_estimate_usd'),
                                    allowance=record.get('preflight_allowance_usd', 0), requests=[])
        episodes.append(dict(ep, versions={'atomic_actions': annotations},
                             statuses={'atomic_actions': statuses},
                             evidence_versions={'atomic_actions': evidence}))
    choice = dict(id='atomic_actions', label='Current annotations',
                  heading=f'{models[left]} vs {models[right]}',
                  columns={'openai': models[left], 'gemini': models[right]},
                  prompt_file='annotation_prompt.txt')
    common.write_json(target / 'data.json', dict(episodes=episodes, runs=[choice], default_run='atomic_actions'))
    shutil.copyfile(Path(__file__).with_suffix('.html'), target / 'index.html')
    return target


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=common.SOURCE)
    parser.add_argument('--left', default='sol')
    parser.add_argument('--right', default='astra')
    parser.add_argument('--port', type=int, default=8767)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    folder = build(args.run.resolve(), args.source.resolve(), args.left, args.right)
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
