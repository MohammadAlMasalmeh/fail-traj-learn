"""Export the approved Sol/V8 pilot as a portable, provenance-preserving dataset."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import zipfile

import annotation_runtime as io
from annotate_trajectories import CATEGORIES, validate_atomic
from batch_trajectories import V8_SHA256


def compact(value):
    return json.dumps(value,ensure_ascii=False,separators=(",",":"))


def target_with_masks(annotation):
    """Keep the full teacher answer; mark uncertain value spans for zero loss."""
    text = '{"episode_id":'+compact(annotation["episode_id"])+',"identity_check":'
    ranges = []
    identity = annotation["identity_check"]
    identity_mask = identity["match_status"] != "unresolved" and not identity["uncertainty"].strip()
    start = len(text)
    text += compact(identity)
    if not identity_mask:
        ranges.append(dict(start=start,end=len(text),reason="uncertain_identity"))
    text += ',"segments":['
    masks = []
    for index,segment in enumerate(annotation["segments"]):
        if index:
            text += ','
        start = len(text)
        text += compact(segment)
        eligible = not segment["review_required"]
        masks.append(dict(segment_index=index,start_frame=segment["start_frame"],
                          end_frame=segment["end_frame"],teacher_loss_mask=int(eligible),
                          review_required=segment["review_required"]))
        if not eligible:
            ranges.append(dict(start=start,end=len(text),reason="review_required_segment",segment_index=index))
    text += ']}'
    if json.loads(text) != annotation:
        raise ValueError("Target serialization changed the teacher annotation")
    return text,dict(identity_teacher_loss_mask=int(identity_mask),segments=masks,
                     assistant_excluded_char_ranges=ranges,
                     offset_unit="Unicode characters in the assistant text; start inclusive, end exclusive",
                     has_unflagged_segment_targets=any(m["teacher_loss_mask"] for m in masks),
                     human_verified=False)


def messages(prompt,ep,target):
    metadata = dict(episode_id=ep["id"],**{k:ep[k] for k in ("task","n_frames","fps","sampled_frames")})
    content = [dict(type="text",text=prompt+"\n"+json.dumps(metadata))]
    for telemetry in ep["telemetry"]:
        frame = telemetry["frame"]
        content.append(dict(type="text",text=json.dumps(telemetry)+"; agent camera then wrist camera"))
        for camera in ("agent","wrist"):
            content.append(dict(type="image",image=f"images/{ep['id']}/{frame:04d}_{camera}.jpg"))
    return [dict(role="user",content=content),dict(role="assistant",content=target)]


def link_or_copy(source,target):
    target.parent.mkdir(parents=True,exist_ok=True)
    try:
        os.link(source,target)
    except OSError:
        shutil.copyfile(source,target)


def validate_splits(selection,ids):
    seen = set()
    for name,rows in selection["splits"].items():
        keys = {(row["dataset"],row["episode_index"]) for row in rows}
        if len(keys) != len(rows) or seen & keys:
            raise ValueError("Duplicate episode or train/development/test leakage")
        seen |= keys
        if name == "train" and {r["id"] for r in rows} != ids:
            raise ValueError("Training export differs from the frozen training selection")


def export(root,out):
    if out.exists():
        raise ValueError("Fresh export directory required")
    completion = io.load(root/"completion.json")
    sample = io.load(root/"random20_review/review_sample.json")
    selection = io.load(root/"selection.json")
    population = sorted(completion["episodes"],key=lambda e:e["episode"])
    ids = {r["episode"] for r in population}
    if len(population) != 500 or len(ids) != 500 or completion["missing"]:
        raise ValueError("Require exactly 500 completed, distinct episodes")
    if sample["completion_sha256"] != io.digest(root/"completion.json") or sample["sample_size"] != 20:
        raise ValueError("Reviewed sample differs from the completed dataset")
    by_id = {r["episode"]:r for r in population}
    for row in sample["selected"]:
        if row != by_id.get(row["episode"]):
            raise ValueError("Reviewed annotation differs from the export source")
    validate_splits(selection,ids)
    training_origins = {r["id"]:r for r in selection["splits"]["train"]}
    out.mkdir(parents=True)
    approval = dict(scope="Batch-level approval to export teacher annotations after inspecting a random sample of 20",
                    user_statement="looks good enough.",recorded_utc=datetime.now(timezone.utc).isoformat(),
                    sample_size=20,sample_seed=sample["seed"],sample_episode_ids=[r["episode"] for r in sample["selected"]],
                    sample_sha256=io.digest(root/"random20_review/review_sample.json"),
                    sample_approval_is_per_episode_adjudication=False,
                    data_quality="sample-approved teacher-generated supervision; not human ground truth")
    io.write_json(out/"review_approval.json",approval)
    splits = {name:[{k:r[k] for k in ("id","dataset","episode_index")} for r in rows]
              for name,rows in selection["splits"].items()}
    io.write_json(out/"splits.json",dict(splits=splits,
                  development_and_test_status="reserved, not annotated or included in train.jsonl"))
    manifests = {}
    provenance = []
    assets = {}
    counters = Counter()
    prompt = None
    with (out/"train.jsonl").open("x",encoding="utf-8",newline="\n") as train:
        for index,row in enumerate(population,1):
            eid = row["episode"]
            run = Path(row["source_run"])
            if not run.is_relative_to(root):
                raise ValueError("Annotation source is outside the completed run")
            if run not in manifests:
                manifests[run] = io.load(run/"manifest.json")
            manifest = manifests[run]
            ep = next(e for e in manifest["episodes"] if e["id"] == eid)
            path = Path(row["annotation"])
            if path.parent != run or io.digest(path) != row["sha256"]:
                raise ValueError("Source annotation hash changed")
            value = io.load(path)
            validate_atomic(value,ep,label_mode="failure_binary")
            record_path = run/"batch_results"/(eid+"_sol_record.json")
            record = io.load(record_path)
            if record["status"] != "validated":
                raise ValueError("Source record is not validated")
            if io.digest(run/"prompt.txt") != V8_SHA256:
                raise ValueError("Source prompt is not frozen V8")
            if prompt is None:
                shutil.copyfile(run/"prompt.txt",out/"prompt.txt")
                prompt = (out/"prompt.txt").read_text(encoding="utf-8")
            annotation_dest = out/"annotations"/(eid+".json")
            link_or_copy(path,annotation_dest)
            answer = run/(eid+"_sol_answer.txt")
            if json.loads(answer.read_text(encoding="utf-8")) != value:
                raise ValueError("Original response differs from parsed annotation")
            link_or_copy(answer,out/"responses"/(eid+".txt"))
            input_path = run/eid/"input.json"
            if io.digest(input_path) != manifest["files"][eid+"/input.json"] or io.load(input_path) != ep:
                raise ValueError("Frozen model input metadata changed")
            link_or_copy(input_path,out/"inputs"/(eid+".json"))
            for frame in ep["sampled_frames"]:
                for camera in ("agent","wrist"):
                    name = f"{frame:04d}_{camera}.jpg"
                    relative = eid+"/"+name
                    sha = io.digest(run/relative)
                    if sha != manifest["files"][relative]:
                        raise ValueError("Supplied model image changed")
                    dest = "images/"+relative
                    link_or_copy(run/relative,out/dest)
                    assets[dest] = sha
            target,supervision = target_with_masks(value)
            origin = training_origins[eid]
            train.write(compact(dict(id=eid,split="train",messages=messages(prompt,ep,target),
                annotation="annotations/"+eid+".json",supervision=supervision,
                metadata=dict(dataset=origin["dataset"],episode_index=origin["episode_index"],
                    n_frames=ep["n_frames"],fps=ep["fps"],teacher_model="gpt-6.1-sol",
                    annotation_source=run.name,quality="sample-approved teacher proposal")))+"\n")
            provenance.append(dict(episode=eid,source_run=run.relative_to(root).as_posix(),
                source_manifest_sha256=io.digest(run/"manifest.json"),annotation_sha256=row["sha256"],
                response_sha256=io.digest(answer),record_sha256=io.digest(record_path),
                original_source_sha256=ep["source_sha256"]))
            counters["episodes"] += 1
            counters["segments"] += len(value["segments"])
            counters["unflagged_segments"] += sum(not s["review_required"] for s in value["segments"])
            counters["masked_segments"] += sum(s["review_required"] for s in value["segments"])
            counters["masked_identities"] += not supervision["identity_teacher_loss_mask"]
            counters["episodes_without_unflagged_segments"] += not supervision["has_unflagged_segment_targets"]
            if index % 100 == 0:
                print(f"Exported {index}/500",flush=True)
    io.write_json(out/"assets.json",assets)
    io.write_json(out/"provenance.json",dict(completion_sha256=io.digest(root/"completion.json"),
        selection_sha256=io.digest(root/"selection.json"),prompt_sha256=V8_SHA256,episodes=provenance))
    (out/"README.txt").write_text(
        "Sol V8 distillation export v1\n\n"
        "500 training episodes. Each train.jsonl line has a user multimodal message, an assistant JSON answer, provenance metadata and explicit supervision masks. Image paths are relative to this directory. Task text, timestamps, paired agent/wrist images and sampled robot telemetry reproduce the teacher inputs. The assistant JSON preserves the original annotation values. Original answer text and parsed annotations are retained separately.\n\n"
        "Review: the user approved export after a random sample of 20. This is batch-level acceptance of teacher data, not independent human adjudication of every episode. No annotation has been promoted to human ground truth. Review flags and uncertainty are unchanged.\n\n"
        "LOSS CONTRACT: a training adapter MUST mask user/image/padding tokens and every assistant token overlapping an assistant_excluded_char_ranges interval. Intervals use Unicode character offsets relative to the assistant content string, before adding model-specific chat-template delimiters. Mask uncertain identity objects and whole review_required segments. Unflagged segment masks identify eligible teacher supervision; they do not certify human-verified positives or negatives. A generic trainer that ignores these custom masks is not a compliant training configuration.\n\n"
        "All 500 records are retained. Records without an unflagged segment are explicitly identified; keep them for provenance and review rather than treating their flagged segments as negatives. Raw teacher-output training with flags unmasked would be a separate, explicit ablation.\n\n"
        "The reserved development/test identities in splits.json have no exported training examples or labels. These are episode-disjoint reservations within the same collecting policy, not a task-held-out or cross-policy evaluation. No model training is launched by this export.\n\n"
        "Files: train.jsonl; annotations/ (unchanged parsed answers); responses/ (unchanged response text); inputs/ (frozen episode metadata); images/ (exact supplied JPEGs); prompt.txt; assets.json; provenance.json; review_approval.json; splits.json; manifest.json.\n",
        encoding="utf-8")
    verification = verify_export(out)
    io.write_json(out/"verification.json",verification)
    files = {p.relative_to(out).as_posix():dict(sha256=io.digest(p),bytes=p.stat().st_size)
             for p in sorted(out.rglob("*")) if p.is_file()}
    manifest = dict(format="sol_v8_multimodal_distillation_v1",model_specific_adapter_required=True,
        teacher_model="gpt-6.1-sol",prompt_sha256=V8_SHA256,counts=dict(counters),
        images=len(assets),development_reserved=50,test_reserved=50,
        review_approval="review_approval.json",categories=sorted(CATEGORIES),
        files=files,exporter_sha256=io.digest(Path(__file__)),
        created_utc=datetime.now(timezone.utc).isoformat())
    io.write_json(out/"manifest.json",manifest)
    print(json.dumps(dict(export=str(out),counts=dict(counters),images=len(assets),verified=True)),flush=True)
    return manifest


def verify_export(out):
    assets = io.load(out/"assets.json")
    ids = set()
    expected_images = set()
    rows = 0
    with (out/"train.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            eid = row["id"]
            if eid in ids or row["split"] != "train":
                raise ValueError("Duplicate example or unexpected split")
            ids.add(eid)
            annotation = io.load(out/row["annotation"])
            text,supervision = target_with_masks(annotation)
            if row["messages"][-1] != dict(role="assistant",content=text) or row["supervision"] != supervision:
                raise ValueError("Assistant target or uncertainty mask differs from annotation")
            ep = io.load(out/"inputs"/(eid+".json"))
            validate_atomic(annotation,ep,label_mode="failure_binary")
            for item in row["messages"][0]["content"]:
                if item["type"] == "image":
                    name = item["image"]
                    path = (out/name).resolve()
                    if not path.is_relative_to(out.resolve()) or io.digest(path) != assets.get(name):
                        raise ValueError("Missing or changed exported image")
                    expected_images.add(name)
            rows += 1
    if rows != 500 or set(assets) != expected_images:
        raise ValueError("Incomplete training export or unused/missing image assets")
    splits = io.load(out/"splits.json")["splits"]
    validate_splits(dict(splits=splits),ids)
    return dict(episodes=rows,unique_episode_ids=len(ids),images=len(expected_images),
                annotations_valid=True,uncertainty_masks_valid=True,split_disjointness=True,
                all_images_resolve_and_match_hashes=True)


def archive(out,path):
    if path.exists():
        raise ValueError("Archive already exists")
    with zipfile.ZipFile(path,"x",compression=zipfile.ZIP_STORED,allowZip64=True) as z:
        for source in sorted(out.rglob("*")):
            if source.is_file():
                z.write(source,path.stem+"/"+source.relative_to(out).as_posix())
    with zipfile.ZipFile(path) as z:
        bad = z.testzip()
        if bad is not None:
            raise ValueError("Archive CRC check failed: "+bad)
    result = dict(archive=str(path),bytes=path.stat().st_size,sha256=io.digest(path),crc_verified=True)
    io.write_json(path.with_suffix(".zip.json"),result)
    print(json.dumps(result),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--archive",type=Path)
    args = parser.parse_args()
    export(args.root.resolve(),args.out.resolve())
    if args.archive:
        archive(args.out.resolve(),args.archive.resolve())
