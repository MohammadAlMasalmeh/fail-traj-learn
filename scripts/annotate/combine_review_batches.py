"""Combine disjoint, complete same-protocol review batches without API calls."""
import argparse
from copy import deepcopy
from pathlib import Path
import shutil

import annotation_runtime as io
from annotate_trajectories import validate_atomic
from review_trajectory_annotations import refresh_comparison_metrics


def safe_resource(folder, relative):
    path=(folder/relative).resolve()
    if not path.is_relative_to(folder.resolve()) or not path.is_file():
        raise ValueError('Missing or unsafe review resource')
    return path


def verify_bundle(folder):
    data=io.load(folder/'data.json')
    if data.get('display_label_mode')!='failure_binary':raise ValueError('Binary atomic reviews required')
    choice=next(r for r in data['runs'] if r['id']==data['default_run'])
    sources={}; manifests={}; protocol={}
    for col in ('openai','gemini'):
        source=data['annotation_sources'][col]
        run=Path(source['run']).resolve(); manifest=io.load(run/'manifest.json')
        if io.digest(run/'manifest.json')!=source['manifest_sha256'] or io.digest(run/'prompt.txt')!=source['prompt_sha256']:
            raise ValueError('Source manifest or prompt changed')
        for name,digest in manifest['files'].items():
            if io.digest(safe_resource(run,name))!=digest:raise ValueError('Frozen source evidence changed')
        keys=[k for k,v in manifest['settings']['models'].items() if v==choice['columns'][col]]
        if len(keys)!=1:raise ValueError('Viewer model differs from source')
        model=keys[0]; observed={e['id']:e for e in manifest['episodes']}
        if set(observed)!={e['id'] for e in data['episodes']}:raise ValueError('Incomplete source episode set')
        settings=manifest['settings']
        if settings.get('scheme')!='atomic' or settings.get('label_mode')!='failure_binary':raise ValueError('Binary atomic sources required')
        protocol[col]=dict(model=choice['columns'][col],prompt_sha256=source['prompt_sha256'],
            **{k:settings[k] for k in ('reasoning','image_detail','sample_stride','max_input_tokens','max_output_tokens')})
        sources[col]=dict(source,model_key=model);manifests[col]=manifest
        for episode in data['episodes']:
            ep=observed[episode['id']]
            if any(episode.get(k)!=v for k,v in ep.items()):raise ValueError('Displayed evidence differs from source')
            if not (folder/'frames'/ep['id']).is_dir():raise ValueError('Missing replay evidence')
            for evidence in episode['evidence_versions'][data['default_run']].values():
                if evidence['prefix'] not in ('sampled','sampled_right') or not (folder/evidence['prefix']/ep['id']).is_dir():
                    raise ValueError('Missing or unexpected sampled evidence')
            value=episode['versions'][data['default_run']].get(col)
            stem=episode['id']+'_'+model
            if value is None or io.load(run/'attempts'/f'{stem}.json').get('status')!='validated':
                raise ValueError('All displayed outputs must be validated')
            if value!=io.load(run/f'{stem}_annotation.json'):raise ValueError('Displayed annotation differs from saved output')
            validate_atomic(value,ep,label_mode='failure_binary')
            status=episode['statuses'][data['default_run']][col]
            raw=safe_resource(folder,status['raw_link'])
            if status['status']!='validated' or raw.read_bytes()!=(run/f'{stem}_answer.txt').read_bytes():
                raise ValueError('Displayed answer differs from saved response')
            if io.load(raw)!=value:raise ValueError('Saved answer and annotation disagree')
    if protocol['openai']['prompt_sha256']!=protocol['gemini']['prompt_sha256']:
        raise ValueError('Both models must use the same prompt')
    inputs=[{k:v for k,v in m['files'].items() if k!='prompt.txt'} for m in manifests.values()]
    if inputs[0]!=inputs[1]:raise ValueError('Both models must have identical frozen input bytes')
    if io.digest(safe_resource(folder,choice['prompt_file']))!=protocol['openai']['prompt_sha256']:
        raise ValueError('Displayed prompt differs from source')
    return data,choice,sources,protocol


def combine(folders,out,label='V8 shared prompt / all R episodes'):
    folders=[Path(f).resolve() for f in folders];out=Path(out).resolve()
    if out.exists():raise ValueError('Fresh combined review directory required')
    if any(out.is_relative_to(f) or f.is_relative_to(out) for f in folders):raise ValueError('Output must be separate from source reviews')
    bundles=[verify_bundle(f) for f in folders]
    if not bundles:raise ValueError('At least one source review required')
    protocol=bundles[0][3];seen=set()
    for data,choice,sources,settings in bundles:
        if settings!=protocol:raise ValueError('Review batches use different prompts or annotation settings')
        ids={e['id'] for e in data['episodes']}
        if len(ids)!=len(data['episodes']) or seen&ids:raise ValueError('Duplicate episode IDs')
        seen|=ids
    # All checks precede copying. The combined folder is a review artifact, not a paid run.
    out.mkdir(parents=True)
    episodes=[];episode_sources={};references=[]
    choice=deepcopy(bundles[0][1]);choice.update(id='atomic_actions',label=label,
        prompt_file='annotation_prompt.txt',action_agreement_file='action_group_agreement.html')
    choice.pop('right_prompt_file',None)
    shutil.copyfile(safe_resource(folders[0],bundles[0][1]['prompt_file']),out/'annotation_prompt.txt')
    shutil.copyfile(Path(__file__).with_name('review_trajectory_annotations.html'),out/'index.html')
    for folder,(data,original,sources,settings) in zip(folders,bundles):
        references.append(dict(review=str(folder),data_sha256=io.digest(folder/'data.json'),sources=sources))
        version=data['default_run']
        for episode in data['episodes']:
            ep=deepcopy(episode);eid=ep['id']
            for field in ('versions','statuses','evidence_versions','action_agreement_versions'):
                ep[field]={'atomic_actions':ep.get(field,{}).get(version,{})}
            shutil.copytree(folder/'frames'/eid,out/'frames'/eid)
            prefixes={v['prefix'] for v in ep['evidence_versions']['atomic_actions'].values()}
            for prefix in prefixes:
                if prefix not in ('sampled','sampled_right'):raise ValueError('Unexpected sampled evidence prefix')
                shutil.copytree(folder/prefix/eid,out/prefix/eid)
            episode_sources[eid]={}
            for col in ('openai','gemini'):
                status=ep['statuses']['atomic_actions'][col]
                raw=safe_resource(folder,status['raw_link'])
                source=sources[col];name=f"{eid}_{source['model_key']}_answer.txt"
                shutil.copyfile(raw,out/name);status['raw_link']=name
                status['display_note']=f"Source: {Path(source['run']).name}; reasoning: {protocol[col]['reasoning']}"
                episode_sources[eid][col]=dict(source,answer_sha256=io.digest(raw),
                    annotation_sha256=io.digest(Path(source['run'])/f"{eid}_{source['model_key']}_annotation.json"))
            episodes.append(ep)
    episodes.sort(key=lambda e:e['id'])
    io.write_json(out/'review_manifest.json',dict(kind='combined_review',paid_api_calls=0,
        protocol=protocol,sources=references,episode_sources=episode_sources,
        code_sha256=io.digest(Path(__file__)),
        limitation='Disjoint saved batches under one protocol; annotations retain their original source runs. Agreement is not accuracy.'))
    fingerprint=io.digest(out/'review_manifest.json')
    annotation_sources={col:dict(runs=[s[col] for _,_,s,_ in bundles],prompt_sha256=protocol[col]['prompt_sha256']) for col in ('openai','gemini')}
    io.write_json(out/'data.json',dict(episodes=episodes,runs=[choice],default_run='atomic_actions',
        display_label_mode='failure_binary',annotation_manifest_sha256=fingerprint,
        annotation_sources=annotation_sources,episode_sources=episode_sources))
    action_report=io.load(folders[0]/'action_group_agreement.json')
    action_report.update(manifest_sha256=fingerprint,annotation_sha256={},episodes=[],unavailable=[],
        source_manifest_sha256=sorted({source['manifest_sha256'] for _,_,s,_ in bundles for source in s.values()}))
    io.write_json(out/'action_group_agreement.json',action_report)
    refresh_comparison_metrics(out)
    print(f'Combined {len(episodes)} episodes from {len(folders)} saved batches: {out}')
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reviews',nargs='+',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--label',default='V8 shared prompt / all R episodes')
    args=p.parse_args();combine(args.reviews,args.out,args.label)
