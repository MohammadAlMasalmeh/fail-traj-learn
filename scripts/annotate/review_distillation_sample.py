"""Build an offline, single-model random review sample from completed annotations.

This prepares review assets only. It does not export a training dataset or mark
the model-generated labels as approved. The seed and exact selected outputs are
saved so the draw can be reproduced without replacing inconvenient examples.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import random
import secrets
import shutil

import annotation_runtime as io
from annotate_trajectories import validate_atomic


def single_model_html(template):
    marker = "providers=['openai','gemini']"
    if template.count(marker) != 1:
        raise ValueError("Viewer provider declaration changed; inspect the template")
    template = template.replace(marker, "providers=['openai']")
    lines = [line for line in template.splitlines()
             if not line.startswith("fetch('llm_semantic_similarity.json')")
             and not line.startswith("fetch('text_similarity.json')")]
    template = "\n".join(lines)
    supplement = """
<style>
.review-workspace,.compare{grid-template-columns:minmax(0,1fr)}
.compare>section:nth-child(2),#scoreSidebar{display:none}
.review-main .cameras{max-width:640px}
</style>
<script>
document.title='Sol V8 - random 20 episode review';
$('mode').querySelector('option[value="right_sampled"]').remove();
$('mode').querySelector('option[value="sampled"]').textContent='Frames supplied to Sol';
$('jumps').closest('details').hidden=true;
$('scoreSidebar').hidden=true;
document.querySelector('h1 + p').textContent='Random sample of 20 from 500 Sol V8 annotations. Red = failure; gray = non-failure. Review against both cameras and the task. Training export is waiting for your review.';
showAgreement=()=>{};showTextScore=()=>{};showLLMScore=()=>{};
const seekSingleModel=seek;
seek=function(f){seekSingleModel(f);$('current').textContent=columnName('openai')+': '+label('openai',frame)};
</script>
"""
    return template.replace("</html>", supplement+"\n</html>")


def build(root, out, count=20, seed=None):
    import numpy as np
    from PIL import Image
    completion_path = root/"completion.json"
    completion = io.load(completion_path)
    population = sorted(completion["episodes"],key=lambda r:r["episode"])
    if len(population) != 500 or len({r["episode"] for r in population}) != 500:
        raise ValueError("Expected the completed 500 distinct episodes")
    if out.exists():
        raise ValueError("Fresh review directory required; never redraw an existing sample")
    seed = secrets.randbits(64) if seed is None else seed
    selected = random.Random(seed).sample(population,count)
    sample = dict(kind="random_human_review_sample",population_size=len(population),
                  sample_size=count,seed=str(seed),sampling="random.Random(seed).sample(sorted episode IDs, count), without replacement",
                  completion_sha256=io.digest(completion_path),selected=selected,
                  review_status="awaiting_user_review",training_export_status="not_exported")
    # Validate selected annotations and their provenance before preparing images.
    manifests = {}
    checked = []
    for row in selected:
        run = Path(row["source_run"])
        if run not in manifests:
            manifests[run] = io.load(run/"manifest.json")
        manifest = manifests[run]
        ep = next(e for e in manifest["episodes"] if e["id"] == row["episode"])
        path = Path(row["annotation"])
        if path.parent != run or io.digest(path) != row["sha256"]:
            raise ValueError("Completion index or selected annotation changed")
        value = io.load(path)
        validate_atomic(value,ep,label_mode="failure_binary")
        record = io.load(run/"batch_results"/(ep["id"]+"_sol_record.json"))
        if record["status"] != "validated":
            raise ValueError("Selected output is not validated")
        source = root/"source"/ep["id"]
        for name,sha in ep["source_sha256"].items():
            if io.digest(source/name) != sha:
                raise ValueError("Original camera/telemetry source changed")
        for name,sha in manifest["files"].items():
            if name == "prompt.txt" or name.startswith(ep["id"]+"/"):
                if io.digest(run/name) != sha:
                    raise ValueError("Frozen supplied evidence changed")
        checked.append((row,run,manifest,ep,value,record))
    prompt_hashes = {io.digest(run/"prompt.txt") for _,run,_,_,_,_ in checked}
    if len(prompt_hashes) != 1:
        raise ValueError("Selected outputs have different prompts")
    out.mkdir(parents=True)
    io.write_json(out/"review_sample.json",sample)
    shutil.copyfile(checked[0][1]/"prompt.txt",out/"annotation_prompt.txt")
    episodes = []
    sources = {}
    run_id = "sol_v8_random20_"+io.digest(out/"review_sample.json")[:12]
    for row,run,manifest,ep,value,record in checked:
        eid = ep["id"]
        frames = out/"frames"/eid
        frames.mkdir(parents=True)
        with np.load(root/"source"/eid/"cameras.npz",allow_pickle=False) as cameras:
            for camera in ("agent","wrist"):
                values = cameras[camera]
                if len(values) != ep["n_frames"]:
                    raise ValueError("Camera length differs from source metadata")
                for number,frame in enumerate(values):
                    Image.fromarray(frame).save(frames/f"{number:04d}_{camera}.jpg",quality=95)
        sampled = out/"sampled"/eid
        sampled.mkdir(parents=True)
        for frame in ep["sampled_frames"]:
            for camera in ("agent","wrist"):
                name = f"{frame:04d}_{camera}.jpg"
                shutil.copyfile(run/eid/name,sampled/name)
        raw = eid+"_sol_answer.txt"
        shutil.copyfile(run/raw,out/raw)
        sources[eid] = dict(run=str(run),manifest_sha256=io.digest(run/"manifest.json"),
                            annotation_sha256=row["sha256"],prompt_sha256=io.digest(run/"prompt.txt"))
        episode = deepcopy(ep)
        episode.update(versions={run_id:dict(openai=value,gemini=None)},
                       statuses={run_id:dict(openai=dict(status="validated",error="",raw_link=raw,display_note=""),
                                            gemini=dict(status="not_applicable",error="",raw_link=None))},
                       evidence_versions={run_id:dict(openai=dict(frames=ep["sampled_frames"],
                           initial_frames=ep["sampled_frames"],prefix="sampled",calls=1,
                           cost=record.get("generation_estimate_usd"),allowance=0,requests=[]))})
        episodes.append(episode)
        print(f"Prepared {eid}: {ep['n_frames']} frames",flush=True)
    data = dict(episodes=episodes,runs=[dict(id=run_id,label="Sol V8 - 20 random episodes",
                heading="Sol V8 - random sample of 20 / 500",columns=dict(openai="gpt-6.1-sol"),
                prompt_file="annotation_prompt.txt")],default_run=run_id,display_label_mode="failure_binary",
                annotation_manifest_sha256=io.digest(out/"review_sample.json"),episode_sources=sources,
                single_model=True)
    io.write_json(out/"data.json",data)
    template = Path(__file__).with_name("review_trajectory_annotations.html").read_text(encoding="utf-8")
    (out/"index.html").write_text(single_model_html(template),encoding="utf-8")
    print(json.dumps(dict(review=str(out),sample_size=count,seed=str(seed),
                          episodes=[r["episode"] for r in selected],training_export="not_exported")),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--count",type=int,default=20)
    parser.add_argument("--seed",type=int)
    args = parser.parse_args()
    build(args.root.resolve(),args.out.resolve(),args.count,args.seed)
