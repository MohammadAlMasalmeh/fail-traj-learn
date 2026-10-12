#!/usr/bin/env python3
"""Small local browser tool for human trajectory annotation.

Input folders contain ``cameras.npz`` with ``agent`` and ``wrist`` arrays and an
``observable.json`` file with ``task`` and optional ``telemetry``. The tool renders
the frames once, starts a local server, and writes ``human_annotation.json``.

Example:
  python scripts/annotate/human_annotator.py \
      outputs/preprocessing_ablation_v3/holdout_source/H001
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import re
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
from PIL import Image


PAGE = r'''<!doctype html>
<meta charset="utf-8"><title>Trajectory annotator</title>
<style>
body{font:15px system-ui;margin:0;background:#17202b;color:#edf2f7}main{max-width:1280px;margin:auto;padding:18px}
.toolbar{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:8px 0 12px}button,select,input,textarea{font:inherit}
button{background:#2d4058;color:white;border:1px solid #60758e;border-radius:5px;padding:7px 10px;cursor:pointer}
button.active{background:#18794e;border-color:#5ee0a0}.danger{background:#713c46}.muted{color:#aebaca}
.episodeNav{background:#263548;padding:10px;border-radius:6px}.episodeNav select{min-width:160px}
#views{display:grid;grid-template-columns:1fr 1fr;gap:12px}figure{margin:0;background:#0d141d;padding:8px;border-radius:6px}img{width:100%;display:block}
#timeline{position:relative;height:48px;background:#263548;border-radius:5px;margin:12px 0;cursor:pointer}
#cursor{position:absolute;z-index:3;top:0;bottom:0;width:2px;background:#ffcf5c}.segmentBar{position:absolute;z-index:1;top:7px;height:34px;min-width:2px;opacity:.68;border-radius:3px}.segmentBar.active{z-index:2;opacity:1;outline:2px solid #fff}.segmentBar.progress{background:#2fb171}.segmentBar.failure{background:#e05d55}.segmentBar.recovery{background:#5b8def}.segmentBar.neutral{background:#8996a5}.mark{position:absolute;z-index:4;top:0;width:5px;height:100%}
.event{background:#ea796b}.state{background:#63b3ed}.grid{display:grid;grid-template-columns:repeat(5,1fr);gap:8px}
label{display:block;margin:4px 0}.panel{background:#202d3c;padding:12px;border-radius:6px;margin-top:12px}
textarea{width:100%;min-height:74px;background:#111a24;color:#edf2f7;border:1px solid #52667d;border-radius:4px;padding:7px}
.segmentText{min-height:58px}.reviewPanel{border-left:5px solid #8996a5}.reviewPanel.progress{border-color:#2fb171}.reviewPanel.failure{border-color:#e05d55}.reviewPanel.recovery{border-color:#5b8def}.reviewPanel.neutral{border-color:#8996a5}.reviewHeadline{font-size:1.15em;margin:5px 0}.segmentRow{padding:6px;border-radius:4px;cursor:pointer}.segmentRow.active{background:#354b62;outline:1px solid #8299b3}.segmentRow.editing{outline:2px solid #ffcf5c}
.privilegedPanel{border:2px solid #d5a83f;background:#302a1d}.privilegedBadge{display:inline-block;background:#d5a83f;color:#17120a;font-weight:700;padding:3px 7px;border-radius:4px}.privilegedStatus{font-size:1.12em;font-weight:700;margin:9px 0}.privilegedStatus.correct{color:#62dda0}.privilegedStatus.wrong{color:#ff857d}.privilegedStatus.contact{color:#ffcf5c}.contactGrid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}.contactValue{display:block;margin-top:3px;color:#fff}
@media(max-width:800px){#views{grid-template-columns:1fr}.grid{grid-template-columns:repeat(2,1fr)}}
</style><main>
<h1 id="title">Trajectory annotation</h1><div id="meta" class="muted"></div>
<div class="panel" style="font-size:1.25em"><b>Task:</b> <span id="task"></span></div>
<div class="toolbar"><button id="back">◀</button><button id="play">Play</button><button id="forward">▶</button>
<input id="frame" type="range" min="0" value="0"><span id="frameNo"></span><label>Speed <select id="speed"><option>.25</option><option>.5</option><option selected>1</option><option>2</option><option>4</option></select>x</label>
<button id="replayAnnotations">Replay annotations from start</button><button id="save">Save annotation</button><span id="saved" class="muted"></span></div>
<div id="views"><figure><figcaption>External camera</figcaption><img id="agent"></figure><figure><figcaption>Wrist camera</figcaption><img id="wrist"></figure></div>
<div id="timeline" title="Click to seek"><div id="cursor"></div></div>
<div id="reviewPanel" class="panel reviewPanel"><b>Annotation replay</b><div id="reviewHeadline" class="reviewHeadline"></div><div id="reviewDetails" class="muted"></div></div>
<div class="toolbar episodeNav"><button id="prevEpisode">◀ Previous episode</button><label>Episode <select id="episodeSelect"></select></label><span id="episodeProgress" class="muted"></span><button id="nextEpisode">Save &amp; next episode ▶</button></div>
<div id="candidatePanel" class="panel" style="display:none;border-left:5px solid #b388ff"><b>GPT candidate review</b><div class="muted">The editor was seeded from an immutable model candidate. Verify every range and claim against the cameras, edit unsupported content, and save the human review separately. This is candidate-assisted review, not an independent human annotation.</div><div id="candidateSource"></div><label>Review decision <select id="candidateDecision"><option value="unreviewed">review not complete</option><option value="accepted">accept candidate as-is</option><option value="corrected">candidate corrected</option></select></label><label>Reviewer note<textarea id="candidateReviewNote" placeholder="Required for a corrected candidate; summarize the important changes"></textarea></label></div>
<div id="privilegedPanel" class="panel privilegedPanel" style="display:none"><span class="privilegedBadge">PRIVILEGED SIMULATOR STATE</span> <span class="muted">Use as annotation ground truth; this is not visual/model input.</span><div><b>Task target:</b> <span id="privilegedTarget"></span></div><div id="privilegedStatus" class="privilegedStatus"></div><div class="contactGrid"><div><b>Any gripper contact</b><span id="privilegedContact" class="contactValue"></span></div><div><b>Bilateral grasp</b><span id="privilegedGrasp" class="contactValue"></span></div><div><b>External support</b><span id="privilegedSupport" class="contactValue"></span></div></div></div>
<div class="panel"><b>Semantic segments</b><p class="muted">The first segment starts at frame 0. Scrub to the last frame in the segment, describe observable facts, then finish it. The classification describes the segment's task-level effect, not every fact in the text. For example, a failure segment may say that the gripper still holds the object while the placement misses. The next segment starts automatically at the following frame. Saved ranges use an exclusive end frame.</p>
<div class="toolbar"><span>Current segment: <strong id="segmentRange"></strong></span><label id="segmentEndEditWrap" style="display:none">End frame (inclusive) <input id="segmentEndEdit" type="number" min="0"></label><select id="segmentClass"><option value="progress">progress</option><option value="failure">failure</option><option value="recovery">recovery</option><option value="neutral">neutral</option></select><select id="failureType" style="display:none"><option value="unknown">failure type: choose</option><option value="missed_target">missed target</option><option value="lost_control">lost control</option><option value="wrong_action">wrong action</option><option value="collision">collision or obstruction</option><option value="timeout">timeout or stalled</option><option value="other">other</option></select></div>
<textarea id="segmentText" class="segmentText" placeholder="Describe observable facts, e.g. Gripper remains closed around the object, but the placement misses."></textarea><textarea id="segmentNote" placeholder="Optional note — use this for visual evidence or uncertainty"></textarea><div class="toolbar"><button id="finishSegment">Finish segment at current frame</button><button id="cancelSegmentEdit" style="display:none">Cancel edit</button></div>
<div id="segmentList" class="muted"></div></div>
<div class="panel"><b>Episode outcome</b><select id="episodeOutcome"><option value="unknown">unknown</option><option value="success">success</option><option value="failure">failure</option></select><p class="muted">Task-level result for the whole trajectory. If unknown, explain the terminal uncertainty in the trajectory note.</p></div>
<details class="panel"><summary><b>Detailed mode (optional)</b> — event, per-frame state, and notes</summary><div class="panel"><b>Event at current frame</b><div class="toolbar" id="events"></div><span id="eventHint" class="muted">Select an event type, then move to its frame and click again to remove it.</span></div>
<div class="panel"><b>State at current frame</b><div class="grid" id="states"></div><label>Frame note<textarea id="note" placeholder="What is visible, and what is uncertain?"></textarea></label></div>
<div class="panel"><b>Trajectory note</b><textarea id="trajectoryNote" placeholder="Overall target identity, task outcome, or unresolved ambiguity"></textarea></div></details>
</main><script>
const DATA=__DATA__, EPISODES=__EPISODES__, CURRENT_EPISODE=__EPISODE_INDEX__; let ann=__ANNOT__; let playing=false,timer=null,selectedEvent='grasp',editingSegment=null;
const $=id=>document.getElementById(id), n=DATA.n_frames;
document.title=`${DATA.id} — trajectory annotation`;$('meta').textContent=`Trajectory ${DATA.id} · ${n} frames · ${DATA.fps} Hz`;$('task').textContent=DATA.task||'Task not supplied';
if(DATA.review_of){$('candidatePanel').style.display='block';$('candidateSource').textContent=`Candidate: ${DATA.review_of.file} · SHA-256 ${DATA.review_of.sha256}`;$('candidateDecision').value=ann.candidate_decision||'unreviewed';$('candidateReviewNote').value=ann.candidate_review_note||'';$('candidateDecision').onchange=()=>ann.candidate_decision=$('candidateDecision').value;$('candidateReviewNote').oninput=()=>ann.candidate_review_note=$('candidateReviewNote').value}
ann.segments=(Array.isArray(ann.segments)?ann.segments:[]).map(s=>({...s,text:s.text||s.note||s.label||'',label:s.label||'',tags:Array.isArray(s.tags)?s.tags:[],classification:s.classification||(s.valence==='negative'?'failure':s.valence==='positive'?'progress':s.valence==='neutral'?'neutral':'progress'),failure_type:s.failure_type||'unknown'}));let segmentStart=ann.segments.length?ann.segments[ann.segments.length-1].end:0;
if(DATA.privileged){ann.annotation_aids=Array.isArray(ann.annotation_aids)?ann.annotation_aids:[];if(!ann.annotation_aids.includes('privileged_simulator_contacts'))ann.annotation_aids.push('privileged_simulator_contacts')}
const eventTypes=['grasp','bilateral_loss','release','loss_closed','loss_unknown','settled'];
const stateTypes=['held','any_gripper_contact','airborne_control','external_contact','settled'];
function frame(){return Number($('frame').value)}
function stateKey(t){return `${t}:${frame()}`}
function renderPrivileged(t){let p=DATA.privileged;if(!p)return;$('privilegedPanel').style.display='block';$('privilegedTarget').textContent=p.target;let f=p.frames[t]||{contact:[],grasped:[],supported:[]},targetGrasped=f.grasped.includes(p.target),targetContact=f.contact.includes(p.target),targetSupported=f.supported.includes(p.target),wrongGrasped=f.grasped.filter(x=>x!==p.target),wrongContact=f.contact.filter(x=>x!==p.target);let status=$('privilegedStatus');status.className='privilegedStatus';if(wrongGrasped.length){status.classList.add('wrong');status.textContent=`WRONG OBJECT GRASPED: ${wrongGrasped.join(', ')}`}else if(targetGrasped){status.classList.add('correct');status.textContent=targetSupported?'CORRECT TARGET GRASPED (still externally supported)':'CORRECT TARGET GRASPED (unsupported)'}else if(targetContact){status.classList.add('contact');status.textContent='Correct target contact, but no bilateral grasp'}else if(wrongContact.length){status.classList.add('wrong');status.textContent=`Non-target contact: ${wrongContact.join(', ')}`}else{status.textContent='No object contact'}$('privilegedContact').textContent=f.contact.length?f.contact.join(', '):'—';$('privilegedGrasp').textContent=f.grasped.length?f.grasped.join(', '):'—';$('privilegedSupport').textContent=f.supported.length?f.supported.join(', '):'—'}
function segmentAt(t){return ann.segments.findIndex(s=>Number(s.start)<=t&&t<Number(s.end))}
function segmentClass(s){return ['progress','failure','recovery','neutral'].includes(s?.classification)?s.classification:'neutral'}
function render(){let t=frame(),activeSegment=segmentAt(t),editing=editingSegment===null?null:ann.segments[editingSegment];$('frameNo').textContent=`frame ${t} / ${n-1} (${(t/(DATA.fps||20)).toFixed(2)}s)`;$('agent').src=`${DATA.base_url}/frames/agent_${String(t).padStart(6,'0')}.jpg`;$('wrist').src=`${DATA.base_url}/frames/wrist_${String(t).padStart(6,'0')}.jpg`;$('cursor').style.left=`${n<2?0:t/(n-1)*100}%`;$('segmentRange').textContent=editing?`${editing.start}–${editing.end-1} (editing saved segment)`:segmentStart>=n?'complete':`${segmentStart}–${Math.max(segmentStart,t)} (stop on ${t})`;$('finishSegment').disabled=editing?false:segmentStart>=n||t<segmentStart;renderPrivileged(t);
 $('segmentList').innerHTML=ann.segments.map((s,i)=>`<div class="segmentRow${i===activeSegment?' active':''}${i===editingSegment?' editing':''}" onclick="seekSegment(${i})"><button onclick="event.stopPropagation();editSegment(${i})">Edit</button> <button class="danger" onclick="event.stopPropagation();removeSegment(${i})">×</button> <b>${escapeHtml(s.text||s.label||'(no description)')}</b>${s.label?` <span class="muted">[${escapeHtml(s.label)}]</span>`:''}: frames ${s.start}–${s.end-1} · ${escapeHtml(s.classification||'progress')}${s.classification==='failure'&&s.failure_type&&s.failure_type!=='unknown'?` (${escapeHtml(s.failure_type)})`:''}${Array.isArray(s.tags)&&s.tags.length?` · #${escapeHtml(s.tags.join(' #'))}`:''}${s.note?` — ${escapeHtml(s.note)}`:''}</div>`).join('');
 let review=$('reviewPanel');review.className='panel reviewPanel';if(activeSegment>=0){let s=ann.segments[activeSegment],kind=segmentClass(s);review.classList.add(kind);$('reviewHeadline').innerHTML=`Segment ${activeSegment+1} / ${ann.segments.length} · <b>${escapeHtml(kind)}</b> · frames ${s.start}–${s.end-1}`;$('reviewDetails').innerHTML=`<b>${escapeHtml(s.text||s.label||'(no description)')}</b>${kind==='failure'&&s.failure_type&&s.failure_type!=='unknown'?`<br>Failure type: ${escapeHtml(s.failure_type)}`:''}${s.note?`<br>Note: ${escapeHtml(s.note)}`:''}`}else{$('reviewHeadline').textContent=ann.segments.length?'No segment covers this frame':'No completed annotations yet';$('reviewDetails').textContent=ann.segments.length?'Use the segment list to inspect a labeled range.':'Complete or load segments to replay their labels.'}
 $('note').value=ann.frame_notes[t]||''; stateTypes.forEach(x=>{let v=ann.states[x]?.[t]??null; document.querySelectorAll(`[data-state="${x}"]`).forEach(b=>b.classList.toggle('active',b.dataset.value===(v===null?'unknown':String(v))))});
 document.querySelectorAll('.mark,.segmentBar').forEach(x=>x.remove());ann.segments.forEach((s,i)=>{let m=document.createElement('i');m.className=`segmentBar ${segmentClass(s)}${i===activeSegment?' active':''}`;m.title=`${s.classification||'segment'}: frames ${s.start}–${s.end-1}`;m.style.left=`${Math.max(0,s.start)/n*100}%`;m.style.width=`${Math.max(1,s.end-s.start)/n*100}%`;m.onclick=e=>{e.stopPropagation();seekSegment(i)};$('timeline').append(m)});ann.events.forEach((e,i)=>{let m=document.createElement('i');m.className='mark event';m.title=`${e.type} @ ${e.frame}`;m.style.left=`${e.frame/(n-1)*100}%`;m.onclick=()=>{ann.events.splice(i,1);render()};$('timeline').append(m)});
} function escapeHtml(s){return String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))} function seekSegment(i){$('frame').value=Math.max(0,Math.min(n-1,Number(ann.segments[i].start)||0));render()}
function resetSegmentEditor(){editingSegment=null;$('segmentText').value='';$('segmentClass').value='progress';$('failureType').value='unknown';$('failureType').style.display='none';$('segmentNote').value='';$('segmentEndEditWrap').style.display='none';$('finishSegment').textContent='Finish segment at current frame';$('cancelSegmentEdit').style.display='none'}
function editSegment(i){let s=ann.segments[i];if(!s)return;editingSegment=i;$('frame').value=Math.max(0,Math.min(n-1,Number(s.start)||0));$('segmentText').value=s.text||s.label||'';$('segmentClass').value=segmentClass(s);$('failureType').value=s.failure_type||'unknown';$('failureType').style.display=s.classification==='failure'?'inline-block':'none';$('segmentNote').value=s.note||'';$('segmentEndEdit').value=Number(s.end)-1;$('segmentEndEdit').max=n-1;$('segmentEndEditWrap').style.display='block';$('finishSegment').textContent='Update segment';$('cancelSegmentEdit').style.display='inline-block';render()}
function cancelSegmentEdit(){let i=editingSegment;resetSegmentEditor();if(i!==null)seekSegment(i);else render()}
function removeSegment(i){resetSegmentEditor();ann.segments.splice(i);segmentStart=ann.segments.length?ann.segments[ann.segments.length-1].end:0;render()}
function finishSegment(){let t=frame(),text=$('segmentText').value.trim(),classification=$('segmentClass').value,failureType=classification==='failure'?$('failureType').value:'unknown',values={text,classification,failure_type:failureType,valence:classification==='failure'?'negative':classification==='progress'?'positive':classification==='neutral'?'neutral':'unknown',note:$('segmentNote').value.trim(),confidence:1};if(!text){alert('Describe what happens in this segment first.');return}if(editingSegment!==null){let i=editingSegment,s=ann.segments[i],endFrame=Number($('segmentEndEdit').value),end=endFrame+1,next=ann.segments[i+1];if(!Number.isInteger(endFrame)||endFrame<Number(s.start)||endFrame>=n){alert(`End frame must be an integer from ${s.start} to ${n-1}.`);return}if(next&&end!==Number(next.start)){alert(`End frame must remain ${Number(next.start)-1} so this segment stays contiguous with the next segment.`);return}ann.segments[i]={...s,...values,start:s.start,end};segmentStart=ann.segments.length?Number(ann.segments[ann.segments.length-1].end):0;delete ann.draft;resetSegmentEditor();$('saved').textContent='Unsaved changes';$('frame').value=Math.max(0,Math.min(n-1,Number(s.start)||0));render();return}if(t<segmentStart){alert('Move to a frame at or after the segment start.');return}ann.segments.push({start:segmentStart,end:t+1,...values});delete ann.draft;segmentStart=t+1;resetSegmentEditor();$('saved').textContent='Unsaved changes';if(segmentStart<n)$('frame').value=segmentStart;render()}
$('segmentClass').onchange=()=>{$('failureType').style.display=$('segmentClass').value==='failure'?'inline-block':'none'};
function setState(type,value){ann.states[type][frame()]=value==='unknown'?null:value==='true'}
function addEvent(){let t=frame(),i=ann.events.findIndex(e=>e.frame===t&&e.type===selectedEvent);if(i>=0)ann.events.splice(i,1);else ann.events.push({frame:t,type:selectedEvent,confidence:1,reason:'Human visual annotation',evidence_frames:[t]});render()}
eventTypes.forEach((x,i)=>{let b=document.createElement('button');b.textContent=x;b.onclick=()=>{selectedEvent=x;document.querySelectorAll('#events button').forEach(z=>z.classList.remove('active'));b.classList.add('active');addEvent()};if(i===0)b.classList.add('active');$('events').append(b)});
$('finishSegment').onclick=finishSegment;
$('cancelSegmentEdit').onclick=cancelSegmentEdit;
stateTypes.forEach(x=>{let box=document.createElement('div');box.innerHTML=`<label>${x}</label>`;['true','false','unknown'].forEach(v=>{let b=document.createElement('button');b.textContent=v;b.dataset.state=x;b.dataset.value=v;b.onclick=()=>{setState(x,v);render()};box.append(b)});$('states').append(box)});
$('frame').max=n-1;$('frame').oninput=render;$('back').onclick=()=>{$('frame').value=Math.max(0,frame()-1);render()};$('forward').onclick=()=>{$('frame').value=Math.min(n-1,frame()+1);render()};
$('timeline').onclick=e=>{let r=e.currentTarget.getBoundingClientRect();$('frame').value=Math.round((e.clientX-r.left)/r.width*(n-1));render()};$('note').oninput=()=>{ann.frame_notes[frame()]=$('note').value};$('trajectoryNote').value=ann.trajectory_note||'';$('trajectoryNote').oninput=()=>ann.trajectory_note=$('trajectoryNote').value;
function tick(){if(!playing)return;let t=frame()+1;if(t>=n){playing=false;$('play').textContent='Play';return} $('frame').value=t;render();timer=setTimeout(tick,1000/(DATA.fps*Number($('speed').value)))}$('play').onclick=()=>{playing=!playing;$('play').textContent=playing?'Pause':'Play';if(playing)tick()};$('replayAnnotations').onclick=()=>{clearTimeout(timer);$('frame').value=0;render();playing=true;$('play').textContent='Pause';timer=setTimeout(tick,1000/(DATA.fps*Number($('speed').value)))};$('episodeOutcome').value=ann.episode_outcome||'unknown';$('episodeOutcome').onchange=()=>ann.episode_outcome=$('episodeOutcome').value;
function captureDraft(){if(editingSegment!==null){delete ann.draft;return}let text=$('segmentText').value,note=$('segmentNote').value;if(segmentStart<n&&(text.trim()||note.trim()))ann.draft={start:segmentStart,frame:frame(),text,classification:$('segmentClass').value,failure_type:$('failureType').value,note};else delete ann.draft}
function annotationComplete(){let expected=0;if(!['success','failure','unknown'].includes(ann.episode_outcome)||!ann.segments.length)return false;if(ann.episode_outcome==='unknown'&&!String(ann.trajectory_note||'').trim())return false;if(ann.review_of&&!['accepted','corrected'].includes(ann.candidate_decision))return false;if(ann.candidate_decision==='corrected'&&!String(ann.candidate_review_note||'').trim())return false;for(let s of ann.segments){if(Number(s.start)!==expected||Number(s.end)<=expected||!['progress','failure','recovery','neutral'].includes(s.classification)||!String(s.text||'').trim())return false;if(s.classification==='failure'&&(!s.failure_type||s.failure_type==='unknown'))return false;expected=Number(s.end)}return expected===n}
async function saveAnnotation(){captureDraft();ann.updated_at=new Date().toISOString();let r=await fetch(`${DATA.base_url}/save`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(ann)});$('saved').textContent=r.ok?'Saved '+new Date().toLocaleTimeString():'Save failed';if(r.ok){EPISODES[CURRENT_EPISODE].annotated=annotationComplete();fillEpisodes()}return r.ok}
function fillEpisodes(){let selected=$('episodeSelect').value,last=CURRENT_EPISODE===EPISODES.length-1;$('episodeSelect').innerHTML='';EPISODES.forEach((e,i)=>{let o=document.createElement('option');o.value=i;o.textContent=`${e.id}${e.annotated?' ✓':''}`;$('episodeSelect').append(o)});$('episodeSelect').value=selected||String(CURRENT_EPISODE);$('episodeProgress').textContent=`${CURRENT_EPISODE+1} / ${EPISODES.length}`;$('prevEpisode').disabled=CURRENT_EPISODE===0;$('nextEpisode').textContent=last?'Save annotation':'Save & next episode ▶'}
async function goEpisode(i){i=Number(i);if(i<0||i>=EPISODES.length||i===CURRENT_EPISODE)return;if(await saveAnnotation())window.location=`/episode/${i}/`}
async function saveAndNext(){if(CURRENT_EPISODE===EPISODES.length-1){await saveAnnotation();return}await goEpisode(CURRENT_EPISODE+1)}
$('save').onclick=saveAnnotation;$('prevEpisode').onclick=()=>goEpisode(CURRENT_EPISODE-1);$('nextEpisode').onclick=saveAndNext;$('episodeSelect').onchange=()=>goEpisode($('episodeSelect').value);fillEpisodes();
if(ann.draft&&ann.draft.start===segmentStart){$('frame').value=Math.min(n-1,Math.max(segmentStart,Number(ann.draft.frame)||segmentStart));$('segmentText').value=ann.draft.text||'';$('segmentNote').value=ann.draft.note||'';$('segmentClass').value=ann.draft.classification||'progress';$('failureType').value=ann.draft.failure_type||'unknown';$('failureType').style.display=$('segmentClass').value==='failure'?'inline-block':'none'}
render();
</script>'''


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def render_frames(source: Path, frame_dir: Path) -> dict:
    with np.load(source / "cameras.npz", mmap_mode="r") as arrays:
        if "agent" not in arrays or "wrist" not in arrays:
            raise ValueError("cameras.npz must contain agent and wrist arrays")
        agent, wrist = arrays["agent"], arrays["wrist"]
        if agent.shape != wrist.shape or len(agent.shape) != 4:
            raise ValueError("agent and wrist must have matching [frames,height,width,channels] arrays")
        frame_dir.mkdir(parents=True, exist_ok=True)
        for i in range(len(agent)):
            for name, stream in (("agent", agent), ("wrist", wrist)):
                target = frame_dir / f"{name}_{i:06d}.jpg"
                if not target.exists():
                    Image.fromarray(np.asarray(stream[i]).astype(np.uint8)).save(target, quality=92)
        return {"n_frames": len(agent), "height": agent.shape[1], "width": agent.shape[2], "fps": 20}


STATE_TYPES = ("held", "any_gripper_contact", "airborne_control", "external_contact", "settled")


def natural_key(path: Path) -> tuple:
    return tuple(int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name))


def annotation_complete(source: Path, annotation_name: str = "human_annotation.json") -> bool:
    """Whether the semantic annotation covers the episode and all required choices."""
    path = source / annotation_name
    try:
        annotation = json.loads(path.read_text(encoding="utf-8"))
        with np.load(source / "cameras.npz", mmap_mode="r") as arrays:
            n_frames = len(arrays["agent"])
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False
    outcome = annotation.get("episode_outcome")
    if outcome not in ("success", "failure", "unknown"):
        return False
    if outcome == "unknown" and not str(annotation.get("trajectory_note", "")).strip():
        return False
    if annotation.get("review_of") and annotation.get("candidate_decision") not in ("accepted", "corrected"):
        return False
    if annotation.get("candidate_decision") == "corrected" and not str(annotation.get("candidate_review_note", "")).strip():
        return False
    segments = annotation.get("segments")
    if not isinstance(segments, list) or not segments:
        return False
    expected = 0
    for segment in segments:
        try:
            start, end = int(segment["start"]), int(segment["end"])
        except (KeyError, TypeError, ValueError):
            return False
        classification = segment.get("classification")
        if start != expected or end <= start or classification not in ("progress", "failure", "recovery", "neutral"):
            return False
        if not str(segment.get("text", "")).strip():
            return False
        if classification == "failure" and segment.get("failure_type") in (None, "", "unknown"):
            return False
        expected = end
    return expected == n_frames


def discover_sources(source: Path, single: bool = False,
                     annotation_name: str = "human_annotation.json") -> tuple[list[Path], int]:
    """Return annotation folders and the initially requested/unfinished index."""
    source = source.resolve()
    if (source / "cameras.npz").is_file():
        if single:
            return [source], 0
        sources = sorted(
            (p.resolve() for p in source.parent.iterdir() if p.is_dir() and (p / "cameras.npz").is_file()),
            key=natural_key,
        )
        return sources, sources.index(source)
    sources = sorted(
        (p.resolve() for p in source.iterdir() if p.is_dir() and (p / "cameras.npz").is_file()),
        key=natural_key,
    ) if source.is_dir() else []
    if not sources:
        raise ValueError(f"no episode folders containing cameras.npz under {source}")
    if single:
        raise ValueError("--single requires an episode folder, not a collection folder")
    first_unfinished = next((i for i, p in enumerate(sources) if not annotation_complete(p, annotation_name)), 0)
    return sources, first_unfinished


def episode_meta(source: Path) -> dict:
    observable = source / "observable.json"
    return json.loads(observable.read_text(encoding="utf-8")) if observable.exists() else {}


def episode_records(sources: list[Path], annotation_name: str = "human_annotation.json") -> list[dict]:
    records = []
    for source in sources:
        meta = episode_meta(source)
        records.append({
            "id": meta.get("id", source.name),
            "annotated": annotation_complete(source, annotation_name),
        })
    return records


def privileged_contact_payload(
    object_slots: list[str], target: str, contact: np.ndarray, grasped: np.ndarray, supported: np.ndarray
) -> dict:
    """Convert dense simulator contact arrays into compact, named per-frame state."""
    arrays = [np.asarray(value) > 0 for value in (contact, grasped, supported)]
    if any(value.ndim != 2 for value in arrays):
        raise ValueError("privileged contact columns must be [frames, objects] arrays")
    if len({value.shape[0] for value in arrays}) != 1:
        raise ValueError("privileged contact columns must have matching frame counts")
    if target not in object_slots:
        raise ValueError(f"task target {target!r} is not in object_slots")
    width = len(object_slots)
    if any(value.shape[1] < width for value in arrays):
        raise ValueError("privileged contact columns have fewer entries than object_slots")
    frames = []
    for t in range(arrays[0].shape[0]):
        frames.append({
            key: [name for i, name in enumerate(object_slots) if values[t, i]]
            for key, values in zip(("contact", "grasped", "supported"), arrays)
        })
    return {"target": target, "object_slots": object_slots, "frames": frames}


def infer_data_root(item: dict) -> Path | None:
    """Infer the recorded-data root from the absolute provenance paths in a private manifest."""
    dataset = item.get("dataset")
    for raw_path in item.get("facts", {}).get("raw_sha256", {}):
        path = Path(raw_path)
        try:
            index = path.parts.index(dataset)
        except (ValueError, TypeError):
            continue
        return Path(*path.parts[:index])
    return None


def load_privileged_contacts(sources: list[Path], manifest_path: Path, data_root: Path | None = None) -> dict[Path, dict]:
    """Load per-frame private simulator contact state for each selected episode."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("--privileged-contacts requires pyarrow in the active Python environment") from exc
    manifest_path = manifest_path.resolve()
    if not manifest_path.is_file():
        raise ValueError(f"privileged manifest does not exist: {manifest_path}")
    values = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(values, list):
        raise ValueError("privileged manifest must contain a list of episode records")
    by_id = {str(value.get("id")): value for value in values if isinstance(value, dict)}
    result = {}
    for source in sources:
        source_id = str(episode_meta(source).get("id", source.name))
        item = by_id.get(source_id)
        if item is None:
            raise ValueError(f"no privileged manifest record for {source_id}")
        root = data_root.resolve() if data_root else infer_data_root(item)
        if root is None:
            raise ValueError(f"cannot infer data root for {source_id}; pass --data-root")
        dataset = str(item["dataset"])
        episode_index = int(item["episode_index"])
        dataset_root = root / dataset
        sidecar_path = dataset_root / "sidecar" / f"episode_{episode_index:06d}.json"
        if not sidecar_path.is_file():
            raise ValueError(f"missing source sidecar for {source_id}: {sidecar_path}")
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        object_slots = list(sidecar.get("object_slots", []))
        target = item.get("facts", {}).get("target_name")
        if not target:
            goals = sidecar.get("goal_state", [])
            target = next((goal[1] for goal in goals if len(goal) > 1 and goal[1] in object_slots), None)
        paths = sorted((dataset_root / "data").glob("chunk-*/*.parquet"))
        if not paths:
            raise ValueError(f"no parquet source files for {source_id} under {dataset_root}")
        columns = ["frame_index", "priv.obj_gripper_contact", "priv.obj_grasped",
                   "priv.obj_support_contact", "priv.obj_obj_contact"]
        table = pq.read_table(paths, columns=columns, filters=[("episode_index", "=", episode_index)])
        table = table.sort_by([("frame_index", "ascending")])
        values = table.to_pydict()
        contact = np.asarray(values["priv.obj_gripper_contact"])
        grasped = np.asarray(values["priv.obj_grasped"])
        supported = (np.asarray(values["priv.obj_support_contact"]) > 0) | (np.asarray(values["priv.obj_obj_contact"]) > 0)
        payload = privileged_contact_payload(object_slots, str(target), contact, grasped, supported)
        with np.load(source / "cameras.npz", mmap_mode="r") as cameras:
            n_frames = len(cameras["agent"])
        if len(payload["frames"]) != n_frames:
            raise ValueError(f"privileged/source frame mismatch for {source_id}: {len(payload['frames'])} != {n_frames}")
        result[source.resolve()] = payload
    return result


def default_annotation(data: dict) -> dict:
    return {
        "id": data["id"], "task": data["task"], "fps": data["fps"], "episode_outcome": "unknown",
        "events": [], "states": {x: [None] * data["n_frames"] for x in STATE_TYPES},
        "frame_notes": [""] * data["n_frames"], "trajectory_note": "", "updated_at": None,
    }


def normalize_annotation(annotation: dict, data: dict) -> dict:
    if not isinstance(annotation, dict):
        raise ValueError("human_annotation.json must contain a JSON object")
    n = data["n_frames"]
    annotation.setdefault("events", [])
    annotation.setdefault("segments", [])
    annotation.setdefault("states", {})
    annotation.setdefault("episode_outcome", "unknown")
    annotation.setdefault("trajectory_note", "")
    notes = list(annotation.get("frame_notes", []))[:n]
    annotation["frame_notes"] = notes + [""] * (n - len(notes))
    for name in STATE_TYPES:
        values = list(annotation["states"].get(name, []))[:n]
        annotation["states"][name] = values + [None] * (n - len(values))
    return annotation


def prepare_episode(sources: list[Path], index: int, privileged: dict[Path, dict] | None = None,
                    annotation_name: str = "human_annotation.json",
                    seed_name: str | None = None) -> tuple[bytes, Path]:
    source = sources[index]
    meta = episode_meta(source)
    generated = source / ".human_annotator"
    info = render_frames(source, generated / "frames")
    out = source / annotation_name
    seed = source / seed_name if seed_name else None
    review_of = None
    if seed is not None:
        if not seed.is_file():
            raise ValueError(f"candidate seed does not exist: {seed}")
        review_of = {"file": seed.name, "sha256": hashlib.sha256(seed.read_bytes()).hexdigest()}
    data = {
        **info, "task": meta.get("task", ""), "id": meta.get("id", source.name),
        "telemetry": meta.get("telemetry", []), "base_url": f"/episode/{index}",
        "privileged": (privileged or {}).get(source.resolve()),
        "review_of": review_of,
    }
    if out.exists():
        annotation = json.loads(out.read_text(encoding="utf-8"))
    elif seed is not None:
        annotation = json.loads(seed.read_text(encoding="utf-8"))
        annotation["review_of"] = review_of
        annotation["candidate_decision"] = "unreviewed"
        annotation["candidate_review_note"] = ""
        annotation["annotation_aids"] = list(dict.fromkeys([
            *annotation.get("annotation_aids", []), "gpt6_candidate",
        ]))
    else:
        annotation = default_annotation(data)
    annotation = normalize_annotation(annotation, data)
    page = (PAGE.replace("__DATA__", json.dumps(data, default=json_default))
                .replace("__EPISODES__", json.dumps(episode_records(sources, annotation_name), default=json_default))
                .replace("__EPISODE_INDEX__", str(index))
                .replace("__ANNOT__", json.dumps(annotation, default=json_default)))
    # The review UI contains Unicode controls (for example, ◀/▶).  Use UTF-8
    # explicitly so the local server works under Windows' cp1252 locale.
    (generated / "index.html").write_text(page, encoding="utf-8")
    return page.encode("utf-8"), out


def handler_for(sources: list[Path], start_index: int, privileged: dict[Path, dict] | None = None,
                annotation_name: str = "human_annotation.json", seed_name: str | None = None):
    class Handler(BaseHTTPRequestHandler):
        def send_body(self, body: bytes, content_type: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def episode_match(self):
            return re.fullmatch(r"/episode/(\d+)(?:/(.*))?", urlparse(self.path).path)

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                self.send_response(302)
                self.send_header("Location", f"/episode/{start_index}/")
                self.end_headers()
                return
            match = self.episode_match()
            if not match:
                self.send_error(404)
                return
            index, tail = int(match.group(1)), match.group(2) or ""
            if index < 0 or index >= len(sources):
                self.send_error(404)
                return
            if tail in ("", "index.html"):
                try:
                    body, _ = prepare_episode(sources, index, privileged, annotation_name, seed_name)
                except ValueError as exc:
                    self.send_error(400, str(exc))
                    return
                self.send_body(body, "text/html; charset=utf-8")
                return
            if not tail.startswith("frames/"):
                self.send_error(404)
                return
            generated = sources[index] / ".human_annotator"
            file = generated / tail
            if not file.is_file() or not file.resolve().is_relative_to(generated.resolve()):
                self.send_error(404)
                return
            self.send_body(file.read_bytes(), mimetypes.guess_type(str(file))[0] or "application/octet-stream")

        def do_POST(self):
            match = re.fullmatch(r"/episode/(\d+)/save", urlparse(self.path).path)
            if not match:
                self.send_error(404)
                return
            index = int(match.group(1))
            if index < 0 or index >= len(sources):
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                value = json.loads(self.rfile.read(length))
            except (ValueError, json.JSONDecodeError):
                self.send_error(400, "invalid JSON")
                return
            if not isinstance(value, dict):
                self.send_error(400, "annotation must be a JSON object")
                return
            source = sources[index]
            value["source"] = str(source)
            out = source / annotation_name
            out.write_text(json.dumps(value, indent=2, default=json_default) + "\n", encoding="utf-8")
            self.send_body(b"saved", "text/plain; charset=utf-8")

        def log_message(self, *_):
            pass

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Episode folder, or a parent folder containing episodes")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--single", action="store_true", help="Do not discover sibling episode folders")
    parser.add_argument("--privileged-contacts", action="store_true",
                        help="Show private simulator contact/grasp state while annotating")
    parser.add_argument("--privileged-manifest", type=Path,
                        help="Private manifest mapping displayed IDs to source dataset episodes")
    parser.add_argument("--data-root", type=Path,
                        help="Recorded dataset root; normally inferred from the private manifest")
    parser.add_argument("--annotation-name", default="human_annotation.json",
                        help="Output filename inside each episode folder")
    parser.add_argument("--seed-name",
                        help="Immutable candidate JSON used only when the output file does not exist")
    args = parser.parse_args()
    for option, name in (("--annotation-name", args.annotation_name), ("--seed-name", args.seed_name)):
        if name is not None and (Path(name).name != name or not name.endswith(".json")):
            parser.error(f"{option} must be a JSON filename without directories")
    try:
        sources, start_index = discover_sources(args.source, args.single, args.annotation_name)
    except ValueError as exc:
        parser.error(str(exc))
    privileged = None
    if args.privileged_contacts:
        collection = args.source.resolve().parent if (args.source / "cameras.npz").is_file() else args.source.resolve()
        manifest = args.privileged_manifest or collection.parent / "holdout_private_manifest.json"
        try:
            privileged = load_privileged_contacts(sources, manifest, args.data_root)
        except (ValueError, RuntimeError, KeyError) as exc:
            parser.error(str(exc))
    if args.seed_name:
        missing = [source / args.seed_name for source in sources if not (source / args.seed_name).is_file()]
        if missing:
            parser.error(f"candidate seed is missing for {len(missing)} episode(s), first: {missing[0]}")
    server = ThreadingHTTPServer(("127.0.0.1", args.port),
                                 handler_for(sources, start_index, privileged,
                                             args.annotation_name, args.seed_name))
    port = server.server_address[1]
    url = f"http://127.0.0.1:{port}/episode/{start_index}/"
    records = episode_records(sources, args.annotation_name)
    print(f"Annotating {len(sources)} episode(s), starting at {records[start_index]['id']}: {url}")
    print(f"Each episode saves {args.annotation_name} in its own source folder.")
    if args.seed_name:
        print(f"Candidate-assisted review enabled from immutable {args.seed_name} files.")
    if privileged is not None:
        print("Privileged simulator contact overlay enabled (annotation aid only; not visual/model input).")
    if not args.no_browser: threading.Timer(.25, lambda: webbrowser.open(url)).start()
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()


if __name__ == "__main__": main()
