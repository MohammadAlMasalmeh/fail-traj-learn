"""Validation for maximal contiguous progress/failure/category states."""
CATEGORIES = {'reaching', 'sequencing_semantic', 'grasp', 'task_not_attempted', 'sudden_episode_termination', 'collision', 'manipulation', 'visible_hardware_issue', 'other'}

def validate_partition(value, ep):
    def require(test, message):
        if not test:
            raise ValueError(message)
    require(isinstance(value, dict) and set(value) == {'episode_id','segments'}
            and value['episode_id'] == ep['id'], 'Envelope')
    require(isinstance(value['segments'], list) and bool(value['segments']), 'Nonempty complete partition required')
    cursor = 0
    for seg in value['segments']:
        require(isinstance(seg, dict) and set(seg) == {'start_frame','end_frame',
                'progress','failure','what_happened','recovery','categories','severity','resetability'}, 'Segment fields')
        a,b = seg['start_frame'],seg['end_frame']
        require(type(a) is int and type(b) is int and a == cursor and a < b <= ep['n_frames'],
                'Gap, overlap, invalid bounds, or unordered segments')
        cursor = b
        require(type(seg['progress']) is bool and type(seg['failure']) is bool, 'Explicit independent booleans required')
        require(isinstance(seg['what_happened'],str) and bool(seg['what_happened'].strip()), 'Description')
        cats = seg['categories']
        require(isinstance(cats,list) and all(isinstance(c,str) for c in cats)
                and set(cats) <= CATEGORIES and len(cats)==len(set(cats)), 'Categories')
        if seg['failure']:
            require(bool(cats), 'Failure requires categories')
            require(seg['severity'] in ('low','medium','catastrophic'), 'Severity')
            require(seg['resetability'] in ('immediate','minor_reset','hard_reset','unresettable'), 'Resetability')
            require(isinstance(seg['recovery'],str) and bool(seg['recovery'].strip()), 'Recovery')
        else:
            require(cats==[] and all(seg[k] is None for k in ('recovery','severity','resetability')),
                    'Nonfailure must have empty categories and null failure-only fields')
    require(cursor == ep['n_frames'], 'Unlabeled episode tail')

def validate_evidence(value, ep):
    def require(test, message):
        if not test: raise ValueError(message)
    def evidence(items):
        require(isinstance(items,list) and bool(items), 'Evidence required')
        for e in items:
            require(isinstance(e,dict) and set(e)=={'frame','camera','observation'}, 'Evidence fields')
            require(type(e['frame']) is int and e['frame'] in ep['sampled_frames']
                    and e['camera'] in ('agent','wrist') and isinstance(e['observation'],str)
                    and bool(e['observation'].strip()), 'Supplied frame/camera evidence')
    require(isinstance(value,dict) and set(value)=={'episode_id','identity_check','segments'}, 'Envelope')
    c=value['identity_check']
    require(isinstance(c,dict) and set(c)=={'requested_object','manipulated_object','match_status','evidence','uncertainty'}, 'Identity fields')
    require(all(isinstance(c[k],str) and bool(c[k].strip()) for k in ('requested_object','manipulated_object')), 'Identity text')
    require(c['match_status'] in ('supported_match','supported_mismatch','unresolved')
            and isinstance(c['uncertainty'],str), 'Identity status')
    require(c['match_status']!='unresolved' or bool(c['uncertainty'].strip()), 'Explain unresolved identity')
    evidence(c['evidence'])
    require(isinstance(value['segments'],list), 'Segments')
    stripped=[]
    for seg in value['segments']:
        require(all(k in seg for k in ('evidence','review_required','uncertainty')), 'Segment evidence fields')
        evidence(seg['evidence'])
        require(type(seg['review_required']) is bool and isinstance(seg['uncertainty'],str), 'Review metadata')
        require(not seg['review_required'] or bool(seg['uncertainty'].strip()), 'Explain review flag')
        stripped.append({k:v for k,v in seg.items() if k not in ('evidence','review_required','uncertainty')})
    validate_partition({'episode_id':value['episode_id'],'segments':stripped},ep)

def validate(value, ep):
    validate_evidence(value, ep)
    for left, right in zip(value['segments'], value['segments'][1:]):
        state = lambda s: (s['progress'], s['failure'], frozenset(s['categories']))
        if state(left) == state(right):
            raise ValueError('Adjacent identical states must be merged')
