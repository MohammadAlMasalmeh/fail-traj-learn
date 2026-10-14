"""Offline metrics for complete, independent progress/failure annotations."""
import math
import re
from statistics import mean

TYPES = {'missed_target','lost_control','wrong_action','collision','timeout','other','unknown'}


def ratio(a,b):
    return a/b if b else None


def prf(tp,fp,fn):
    return {'tp':tp,'fp':fp,'fn':fn,'precision':ratio(tp,tp+fp),
            'recall':ratio(tp,tp+fn),'f1':ratio(2*tp,2*tp+fp+fn)}


def state(s):
    return int(s['progress'])*2+int(s['failure']) if 'progress' in s else int(s['failure'])


def failure_types(s):
    if 'categories' in s:
        return set(s['categories'])
    return set(re.findall(r'\b(?:missed_target|lost_control|wrong_action|collision|timeout|other|unknown)\b',s.get('failure_types','').lower()))


def timeline(segments,n):
    result=[None]*n
    for s in segments:
        if not 0<=s['start']<s['end']<=n:raise ValueError('Invalid interval')
        for f in range(s['start'],s['end']):
            if result[f] is not None:raise ValueError('Overlapping segments')
            result[f]=s
    return result


def collapse(segments):
    out=[]
    for s in segments:
        if out and out[-1]['end']==s['start'] and state(out[-1])==state(s):out[-1]['end']=s['end']
        else:out.append(dict(s))
    return out


def match_ordered(left,right,score):
    """One-to-one ordered matching: maximize count, then total score; no reuse."""
    dp=[[(0,0.0,[]) for _ in range(len(right)+1)] for _ in range(len(left)+1)]
    for i,a in enumerate(left,1):
        for j,b in enumerate(right,1):
            options=[dp[i-1][j],dp[i][j-1]]
            value=score(a,b)
            if value is not None:
                count,total,pairs=dp[i-1][j-1]
                options.append((count+1,total+value,pairs+[(i-1,j-1)]))
            dp[i][j]=max(options,key=lambda v:(v[0],v[1]))
    return dp[-1][-1][2]


def boundaries(pred,ref,fps,tolerance):
    pairs=match_ordered(pred,ref,lambda a,b:-abs(a-b) if abs(a-b)/fps<=tolerance+1e-9 else None)
    errors=[abs(pred[i]-ref[j])/fps for i,j in pairs]
    result=prf(len(pairs),len(pred)-len(pairs),len(ref)-len(pairs))
    result.update(tolerance_seconds=tolerance,matched_mae_seconds=mean(errors) if errors else None,
                  unmatched_prediction=len(pred)-len(pairs),unmatched_reference=len(ref)-len(pairs),
                  matches=[{'prediction_frame':pred[i],'reference_frame':ref[j],
                            'signed_error_seconds':(pred[i]-ref[j])/fps} for i,j in pairs],
                  unmatched_prediction_frames=[a for i,a in enumerate(pred) if i not in {x for x,_ in pairs}],
                  unmatched_reference_frames=[a for j,a in enumerate(ref) if j not in {y for _,y in pairs}])
    return result


def segment_overlap(pred,ref,threshold):
    def score(a,b):
        overlap=max(0,min(a['end'],b['end'])-max(a['start'],b['start']))
        union=max(a['end'],b['end'])-min(a['start'],b['start'])
        iou=overlap/union
        return iou if state(a)==state(b) and iou>=threshold else None
    pairs=match_ordered(pred,ref,score)
    return {**prf(len(pairs),len(pred)-len(pairs),len(ref)-len(pairs)),'iou_threshold':threshold,
            'matched_mean_iou':mean(score(pred[i],ref[j]) for i,j in pairs) if pairs else None}


def text_for(s):
    # Keep explanations; omit confidence, generic instructions and evidence frame lists.
    return ' '.join(s.get(k,'').strip() for k in ('phase','description','failure_reason') if s.get(k)).strip()


def cosine(a,b):
    denom=math.sqrt(sum(x*x for x in a)*sum(y*y for y in b))
    return sum(x*y for x,y in zip(a,b))/denom if denom else None


def semantic_overlap(pred,ref,n,vectors):
    total=0.0;covered=0;matching_total=0.0;matching_frames=0;pairs=[]
    for i,a in enumerate(pred):
        for j,b in enumerate(ref):
            overlap=max(0,min(a['end'],b['end'])-max(a['start'],b['start']))
            ta,tb=text_for(a),text_for(b)
            if not overlap or not ta or not tb:continue
            similarity=cosine(vectors[ta],vectors[tb])
            if similarity is None:continue
            total+=overlap*similarity;covered+=overlap
            same=state(a)==state(b)
            if same:matching_total+=overlap*similarity;matching_frames+=overlap
            pairs.append({'pred_segment':i,'ref_segment':j,'overlap_frames':overlap,'cosine':similarity,'same_binary_state':same})
    return {'time_weighted_cosine':ratio(total,covered),'text_coverage_fraction':covered/n,
            'same_state_cosine':ratio(matching_total,matching_frames),
            'same_state_text_coverage_fraction':matching_frames/n,'pairs':pairs}


def evaluate(pred,ref,n,fps,vectors=None, require_verified=True, label_mode='progress_failure'):
    if label_mode not in ('progress_failure', 'failure_binary'):
        raise ValueError('Unknown label mode')
    binary = label_mode == 'failure_binary'
    if binary:
        pred=[{k:v for k,v in s.items() if k!='progress'} for s in pred]
        ref=[{k:v for k,v in s.items() if k!='progress'} for s in ref]
    p,r=timeline(pred,n),timeline(ref,n)
    if any(s is None for s in r):raise ValueError('Reference is not complete')
    for s in ref:
        confidences=(s.get('failure_confidence'),) if binary else (s.get('progress_confidence'),s.get('failure_confidence'))
        if require_verified and (s.get('review_required') or 'low' in confidences):
            raise ValueError('Reference still has unresolved/provisional labels')
    count=2 if binary else 4
    confusion=[[0]*(count+1) for _ in range(count)] # final prediction column = missing
    for a,b in zip(p,r):confusion[state(b)][state(a) if a is not None else count]+=1
    axes={}
    for axis in (('failure',) if binary else ('progress','failure')):
        tp=fp=fn=tn=0
        for a,b in zip(p,r):
            truth=b[axis];guess=a[axis] if a else None
            tp+=int(truth and guess is True);fp+=int(not truth and guess is True)
            fn+=int(truth and guess is not True);tn+=int(not truth and guess is False)
        axes[axis]={**prf(tp,fp,fn),'tn':tn,'accuracy':(tp+tn)/n,
                    'iou':ratio(tp,tp+fp+fn),'positive_reference_frames':tp+fn}
    labels={}
    for label in sorted((TYPES | set().union(*(failure_types(s) for s in pred + ref))) - {'unknown'}):
        tp=fp=fn=0
        for a,b in zip(p,r):
            truth=label in failure_types(b);guess=a is not None and label in failure_types(a)
            tp+=int(truth and guess);fp+=int(not truth and guess);fn+=int(truth and not guess)
        labels[label]=prf(tp,fp,fn)
    ps,rs=collapse(pred),collapse(ref)
    rawp=[s['start'] for s in pred if 0<s['start']<n];rawr=[s['start'] for s in ref if 0<s['start']<n]
    statep=[s['start'] for s in ps if 0<s['start']<n];stater=[s['start'] for s in rs if 0<s['start']<n]
    result={'label_mode':label_mode, 'joint_accuracy':sum(confusion[i][i] for i in range(count))/n,
            'joint_confusion_reference_rows_prediction_columns':confusion,
            'state_order':['non_failure','failure','missing_prediction'] if binary else ['neither','failure_only','progress_only','both','missing_prediction'],
            'failure_category_accuracy':sum(a is not None and a['failure']==b['failure'] and failure_types(a)==failure_types(b) for a,b in zip(p,r))/n,
            'prediction_coverage':sum(s is not None for s in p)/n,'axes':axes,'failure_types':labels,
            'phase_boundaries':[boundaries(rawp,rawr,fps,t) for t in (.10,.25,.50)],
            'binary_state_boundaries':[boundaries(statep,stater,fps,t) for t in (.10,.25,.50)],
            'binary_state_segment_overlap':[segment_overlap(ps,rs,t) for t in (.25,.50,.75)],
            'phase_segment_count':{'prediction':len(pred),'reference':len(ref)},
            'reference_review_required_fraction':sum(s.get('review_required',False) for s in r)/n,
            'prediction_review_required_fraction':sum(s is not None and s.get('review_required',False) for s in p)/n}
    if vectors is not None:result['semantic']=semantic_overlap(pred,ref,n,vectors)
    return result
