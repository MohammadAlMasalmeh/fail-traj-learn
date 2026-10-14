"""Offline action/category agreement, independent of within-action segment count.

Scores compare model assertions, not video correctness. Action IDs are local;
ordered temporal matching aligns groups without using category agreement.
"""
import argparse
from collections import defaultdict
import html
from pathlib import Path

import annotation_runtime as io
from annotate_trajectories import validate_atomic


def action_groups(value):
    groups = []
    for number, segment in enumerate(value['segments'], 1):
        if not groups or groups[-1]['action_id'] != segment['action_id']:
            groups.append(dict(action_id=segment['action_id'], action=segment['action'],
                               start_frame=segment['start_frame'], end_frame=segment['end_frame'],
                               segments=[], category_spans=defaultdict(list), descriptions=[]))
        group = groups[-1]
        group['end_frame'] = segment['end_frame']
        group['segments'].append(number)
        group['descriptions'].append(dict(segment=number, start_frame=segment['start_frame'],
            end_frame=segment['end_frame'], text=segment['what_happened'],
            observed_result=segment['observed_result'], review_required=segment['review_required'],
            uncertainty=segment['uncertainty']))
        for category in segment['categories']:
            spans = group['category_spans'][category]
            if spans and spans[-1][1] == segment['start_frame']:
                spans[-1][1] = segment['end_frame']
            else:
                spans.append([segment['start_frame'], segment['end_frame']])
    for group in groups:
        group['category_spans'] = dict(group['category_spans'])
        group['categories'] = sorted(group['category_spans'])
    return groups


def temporal_iou(a, b):
    overlap = max(0, min(a['end_frame'], b['end_frame']) - max(a['start_frame'], b['start_frame']))
    union = max(a['end_frame'], b['end_frame']) - min(a['start_frame'], b['start_frame'])
    return overlap / union


def align(left, right):
    """Maximum total temporal IoU, order-preserving, one-to-one; no text/category input."""
    dp = [[(0., 0, []) for _ in range(len(right)+1)] for _ in range(len(left)+1)]
    for i, a in enumerate(left, 1):
        for j, b in enumerate(right, 1):
            choices = [dp[i-1][j], dp[i][j-1]]
            score = temporal_iou(a, b)
            if score > 0:
                total, count, pairs = dp[i-1][j-1]
                choices.append((total+score, count+1, pairs+[(i-1, j-1)]))
            dp[i][j] = max(choices, key=lambda x: (x[0], x[1]))
    return dp[-1][-1][2]


def dice(shared, left_count, right_count):
    return 2*shared/(left_count+right_count) if left_count+right_count else None


def compare(left_value, right_value):
    left, right = action_groups(left_value), action_groups(right_value)
    pairs = align(left, right)
    rows = []
    matched_actions = matched_categories = 0
    for i, j in pairs:
        a, b = left[i], right[j]
        same_action = a['action'] == b['action'] and a['action'] != 'unknown'
        ac, bc = set(a['categories']), set(b['categories'])
        shared = sorted(ac & bc) if same_action else []
        matched_actions += same_action
        matched_categories += len(shared)
        rows.append(dict(left_index=i, right_index=j, left=a, right=b,
            action_matches=same_action, temporal_iou=temporal_iou(a, b),
            shared_categories=shared, left_only_categories=sorted(ac-set(shared)),
            right_only_categories=sorted(bc-set(shared))))
    left_categories = sum(len(g['categories']) for g in left)
    right_categories = sum(len(g['categories']) for g in right)
    return dict(
        scores=dict(
            action_agreement=dice(matched_actions, len(left), len(right)),
            category_agreement=dice(matched_categories, left_categories, right_categories),
            combined_agreement=dice(matched_actions+matched_categories,
                                    len(left)+left_categories, len(right)+right_categories)),
        counts=dict(left_segments=len(left_value['segments']), right_segments=len(right_value['segments']),
            left_actions=len(left), right_actions=len(right), matched_actions=matched_actions,
            left_category_assignments=left_categories, right_category_assignments=right_categories,
            matched_category_assignments=matched_categories),
        matched=rows,
        unmatched_left=[dict(index=i, group=g) for i, g in enumerate(left) if i not in {a for a, _ in pairs}],
        unmatched_right=[dict(index=j, group=g) for j, g in enumerate(right) if j not in {b for _, b in pairs}])


def run(run_dir, left_key, right_key, destination, right_run=None):
    folders, manifests = io.comparison_sources(run_dir, left_key, right_key, right_run)
    manifest, right_manifest = manifests
    if left_key == right_key:
        raise ValueError('Choose two distinct model keys in the source manifest')
    episodes, unavailable, artifacts = [], [], {}
    for ep in manifest['episodes']:
        values = []
        for model, folder, saved in zip((left_key, right_key), folders, manifests):
            stem = ep['id'] + '_' + model
            slot = folder / 'attempts' / (stem+'.json')
            path = folder / (stem+'_annotation.json')
            status = io.load(slot).get('status') if slot.exists() else 'missing'
            if status != 'validated' or not path.exists():
                unavailable.append(dict(episode=ep['id'], model=model, status=status))
                continue
            value = io.load(path)
            validate_atomic(value, ep, label_mode=saved['settings'].get('label_mode', 'progress_failure'))
            values.append(value)
            artifacts[path.name] = io.digest(path)
        if len(values) == 2:
            episodes.append(dict(id=ep['id'], task=ep['task'], fps=ep['fps'],
                                 n_frames=ep['n_frames'], **compare(*values)))
    result = dict(
        method='ordered_action_category_dice_v1',
        left_model=manifest['settings']['models'][left_key], right_model=right_manifest['settings']['models'][right_key],
        manifest_sha256=io.digest(run_dir / 'manifest.json'), annotation_sha256=artifacts,
        source_manifest_sha256=[io.digest(folder / 'manifest.json') for folder in folders],
        code_sha256=io.digest(Path(__file__)), unavailable=unavailable, episodes=episodes,
        definition='An action occurrence and each unique category assigned within it each receive one unit of weight. Category matches require matching action operations in temporally aligned groups. Scores use symmetric Dice; neither annotation is ground truth.',
        limitations='Prototype structured agreement only. Exact operation names are compared. Object identity, destination, prose, physical result, progress/failure booleans, uncertainty strength, and within-action category timing are not scored. Category intervals and source descriptions remain available for inspection. Repeated occurrences of one category within an action count once. Equal fact weights are a convention, not human-calibrated severity weights.',
        paid_api_calls=0)
    destination.parent.mkdir(parents=True, exist_ok=True)
    io.atomic_json(destination, result)
    render(result, destination.with_suffix('.html'))
    for ep in episodes:
        print(ep['id'], {k: round(v*100, 2) if v is not None else None for k, v in ep['scores'].items()})
    return result


def render(result, destination):
    esc = html.escape
    percent = lambda v: f'{v*100:.1f}%' if v is not None else 'N/A'
    parts = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>Action-group agreement</title>',
        '<style>body{font:16px/1.6 system-ui;background:#101720;color:#e9eef5;margin:28px;max-width:1200px}table{border-collapse:collapse;width:100%}td,th{border:1px solid #42556d;padding:9px;text-align:left}th{background:#213249}.difference{color:#ffbf87}details{margin:12px 0;border:1px solid #42556d;padding:12px}.texts{display:grid;grid-template-columns:1fr 1fr;gap:20px}summary{cursor:pointer}a{color:#9eceff}@media(max-width:700px){.texts{grid-template-columns:1fr}}</style>',
        '<h1>Action-group agreement</h1><p>'+esc(result['left_model'])+' versus '+esc(result['right_model'])+'</p>',
        '<p>Scores measure agreement in action and category assignments. They do not measure video correctness or full semantic equivalence.</p>',
        '<p>'+esc(result['definition'])+'</p>']
    for ep in result['episodes']:
        c, s = ep['counts'], ep['scores']
        parts += [f"<h2>{esc(ep['id'])}</h2><p>{esc(ep['task'])}</p>",
            f"<p>Segments: {c['left_segments']} / {c['right_segments']}. Actions: {c['left_actions']} / {c['right_actions']}. "
            f"Action agreement: <b>{percent(s['action_agreement'])}</b>. Category agreement: <b>{percent(s['category_agreement'])}</b>. "
            f"Combined: <b>{percent(s['combined_agreement'])}</b>.</p>",
            '<table><tr><th>Left action / segments</th><th>Right action / segments</th><th>Action overlap</th><th>Shared categories</th><th>Category differences</th></tr>']
        for row in ep['matched']:
            a, b = row['left'], row['right']
            difference = ('Left only: '+', '.join(row['left_only_categories'])+'; right only: '+', '.join(row['right_only_categories']))
            parts.append(f"<tr><td>{esc(a['action'])} / {a['segments']}<br>[{a['start_frame']}, {a['end_frame']})</td>"
                f"<td>{esc(b['action'])} / {b['segments']}<br>[{b['start_frame']}, {b['end_frame']})</td>"
                f"<td>{percent(row['temporal_iou'])}</td><td>{esc(', '.join(row['shared_categories']) or 'none')}</td>"
                f"<td class='difference'>{esc(difference) if row['left_only_categories'] or row['right_only_categories'] else 'none'}</td></tr>")
        parts.append('</table>')
        for side in ('left', 'right'):
            for missed in ep['unmatched_'+side]:
                parts.append(f"<p class='difference'>Unmatched {side} action: {esc(missed['group']['action'])}.</p>")
        for row in ep['matched']:
            parts.append('<details><summary>Original segments: '+esc(row['left']['action'])+' / '+esc(row['right']['action'])+'</summary><div class="texts">')
            for side in ('left', 'right'):
                group = row[side]
                parts.append('<section><b>'+side.title()+'</b><p>Category intervals: '+esc(str(group['category_spans']))+'</p>')
                for d in group['descriptions']:
                    parts.append(f"<p><b>Segment {d['segment']} [{d['start_frame']}, {d['end_frame']})</b><br>{esc(d['text'])}</p>"
                                 '<p>Physical result: '+esc(d['observed_result'])+'</p>')
                    if d['uncertainty']:
                        parts.append('<p>Uncertainty: '+esc(d['uncertainty'])+'</p>')
                parts.append('</section>')
            parts.append('</div></details>')
    parts += ['<h2>Scope</h2><p>'+esc(result['limitations'])+'</p>',
              f"<p>Unavailable outputs: {len(result['unavailable'])}. No API calls.</p></html>"]
    destination.write_text('\n'.join(parts), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--left', default='sol')
    parser.add_argument('--right', default='astra')
    parser.add_argument('--right-run', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run(args.run.resolve(), args.left, args.right, args.out.resolve(), args.right_run)
