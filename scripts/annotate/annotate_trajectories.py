"""Prepare frozen observable inputs and annotate atomic actions or task states."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import time

import annotation_runtime as common
import annotation_runtime as transport

ROOT = common.ROOT
OUT = ROOT / "outputs/trajectory_annotations_atomic"
SCHEME = "atomic"
SOURCE = common.SOURCE
PROMPT_DIR = ROOT / "scripts/annotate/prompts/atomic_actions"
IDS = ("R001", "R002", "R003", "R004")
MODELS = {"sol": "gpt-6.1-sol", "astra": "gpt-6-astra"}
MAX_INPUT, MAX_OUTPUT = 80000, 12288
RESERVES = {"sol": .50, "astra": 2.50}
RATES = {
    "sol": dict(input=2, cached=.1, cache_write=2.5, output=10),
    "astra": dict(input=10, cached=1, cache_write=12.5, output=50),
}
CATEGORIES = {
    "reaching", "grasp", "manipulation", "sequencing_semantic", "collision",
    "task_not_attempted", "sudden_episode_termination", "visible_hardware_issue", "other",
}
FIELDS = {
    "start_frame", "end_frame", "action_id", "action", "object_id", "destination_id",
    "observed_result", "boundary_reason", "progress", "failure", "what_happened",
    "categories", "recovery", "severity", "resetability", "evidence",
    "review_required", "uncertainty",
}


def settings():
    return dict(
        scheme=SCHEME, models=MODELS, reasoning="low", image_detail="high", sample_stride=5,
        max_input_tokens=MAX_INPUT, max_output_tokens=MAX_OUTPUT, workers=2,
        retries=0, reservations=RESERVES, maximum_reserved_usd=len(IDS) * sum(RESERVES.values()),
        rates_per_million=RATES, price_checked="2026-10-10",
        pricing_sources=[
            "https://developers.openai.com/api/docs/models/gpt-6-astra",
            "https://developers.openai.com/api/docs/models/gpt-6.1-sol",
        ],
    )


def code_hashes():
    return {p.relative_to(ROOT).as_posix(): common.digest(p) for p in (
        Path(__file__), Path(common.__file__), ROOT / "scripts/annotate/annotation_states.py", PROMPT_DIR / "annotation.txt",
    )}


def prepare():
    import numpy as np
    from PIL import Image
    if OUT.exists():
        raise ValueError("Fresh output directory required; never overwrite a run")
    if OUT.is_relative_to(SOURCE):
        raise ValueError("Output directory must be separate from source evidence")
    # Read only raw camera/observable files, never candidate or human labels.
    records = []
    for eid in IDS:
        source = SOURCE / eid
        observed = common.load(source / "observable.json")
        n, fps = observed["n_frames"], observed["fps"]
        if observed["id"] != eid or type(n) is not int or n <= 0 or not isinstance(fps, (int, float)) or fps <= 0:
            raise ValueError("Invalid observable metadata")
        frames = sorted(set(range(0, n, 5)) | {n - 1})
        rows = {r["frame"]: r for r in observed["telemetry"]}
        telemetry = [
            dict(frame=f, seconds=f/fps, **{
                k: rows[f][k] for k in ("eef_xyz_m", "finger_qpos_m", "gripper_command")
            }) for f in frames
        ]
        records.append(dict(
            id=eid, task=observed["task"], n_frames=n, fps=fps,
            sampled_frames=frames, telemetry=telemetry,
            source_sha256={name: common.digest(source/name)
                           for name in ("cameras.npz", "observable.json")},
        ))
    OUT.mkdir(parents=True)
    shutil.copyfile(PROMPT_DIR / "annotation.txt", OUT / "prompt.txt")
    for ep in records:
        folder = OUT / ep["id"]
        folder.mkdir()
        with np.load(SOURCE / ep["id"] / "cameras.npz", allow_pickle=False) as arrays:
            for camera in ("agent", "wrist"):
                values = arrays[camera]
                if len(values) != ep["n_frames"]:
                    raise ValueError("Camera frame count differs from metadata")
                for frame in ep["sampled_frames"]:
                    Image.fromarray(values[frame]).save(
                        folder / f"{frame:04d}_{camera}.jpg", quality=90)
        common.write_json(folder / "input.json", ep)
    manifest = dict(
        name="trajectory_annotations_" + SCHEME, settings=settings(), code_hashes=code_hashes(),

        prepared_utc=datetime.now(timezone.utc).isoformat(), episodes=records,
        limitations="Model agreement is not accuracy. Record episode split and human review separately.",
        files={p.relative_to(OUT).as_posix(): common.digest(p)
               for p in sorted(OUT.rglob("*")) if p.is_file()},
    )
    common.write_json(OUT / "manifest.json", manifest)
    report()
    print(json.dumps(dict(
        episodes=[dict(id=e["id"], camera_pairs=len(e["sampled_frames"])) for e in records],
        models=MODELS, calls=len(IDS)*len(MODELS), maximum_reserved_usd=len(IDS) * sum(RESERVES.values()),
        manifest_sha256=common.digest(OUT / "manifest.json"),
    )), flush=True)


def verify():
    manifest = common.load(OUT / "manifest.json")
    expected_code = manifest["code_hashes"]
    if manifest["settings"] != settings() or expected_code != code_hashes():
        raise ValueError("Frozen settings, prompt, or runner changed")
    if [ep["id"] for ep in manifest["episodes"]] != list(IDS):
        raise ValueError("Unexpected episode matrix")
    for name, digest in manifest["files"].items():
        path = (OUT / name).resolve()
        if not path.is_relative_to(OUT.resolve()) or common.digest(path) != digest:
            raise ValueError("Frozen evidence changed")
    return manifest


def payload(model_key, ep):
    metadata = dict(episode_id=ep["id"], **{
        k: ep[k] for k in ("task", "n_frames", "fps", "sampled_frames")
    })
    parts = [dict(type="input_text", text=(
        (OUT / "prompt.txt").read_text(encoding="utf-8") + "\n" + json.dumps(metadata)))]
    for row in ep["telemetry"]:
        parts.append(dict(type="input_text", text=json.dumps(row) + "; agent camera then wrist camera"))
        for camera in ("agent", "wrist"):
            data = base64.b64encode(
                (OUT / ep["id"] / f'{row["frame"]:04d}_{camera}.jpg').read_bytes()
            ).decode()
            parts.append(dict(type="input_image", detail="high",
                              image_url="data:image/jpeg;base64," + data))
    return dict(
        model=MODELS[model_key], input=[dict(role="user", content=parts)],
        reasoning=dict(effort="low"), max_output_tokens=MAX_OUTPUT,
        text=dict(format=dict(type="json_object")), store=False,
        service_tier="default", truncation="disabled",
    )


def validate_atomic(value, ep):
    def require(condition, message):
        if not condition:
            raise ValueError(message)
    def nonempty(value):
        return isinstance(value, str) and bool(value.strip())
    def evidence(items):
        require(isinstance(items, list) and bool(items), "Evidence required")
        for e in items:
            require(isinstance(e, dict) and set(e) == {"frame", "camera", "observation"},
                    "Evidence fields")
            require(type(e["frame"]) is int and e["frame"] in ep["sampled_frames"]
                    and e["camera"] in ("agent", "wrist") and nonempty(e["observation"]),
                    "Evidence must cite supplied frames and cameras")
    require(isinstance(value, dict) and set(value) == {"episode_id", "identity_check", "segments"}
            and value["episode_id"] == ep["id"], "Episode envelope")
    identity = value["identity_check"]
    require(isinstance(identity, dict) and set(identity) == {
        "requested_object", "manipulated_object", "match_status", "evidence", "uncertainty"
    }, "Identity fields")
    require(all(nonempty(identity[k]) for k in ("requested_object", "manipulated_object")),
            "Identity descriptions")
    require(identity["match_status"] in ("supported_match", "supported_mismatch", "unresolved")
            and isinstance(identity["uncertainty"], str), "Identity status")
    require(identity["match_status"] != "unresolved" or nonempty(identity["uncertainty"]),
            "Unresolved identity needs an explanation")
    evidence(identity["evidence"])
    segments = value["segments"]
    require(isinstance(segments, list) and bool(segments), "Complete nonempty partition required")
    cursor, action_count, previous, known_referents = 0, 0, None, {}
    for segment in segments:
        require(isinstance(segment, dict) and set(segment) == FIELDS, "Atomic segment fields")
        a, b = segment["start_frame"], segment["end_frame"]
        require(type(a) is int and type(b) is int and a == cursor and a < b <= ep["n_frames"],
                "Gap, overlap, or invalid boundary")
        cursor = b
        require(all(nonempty(segment[k]) for k in (
            "action_id", "action", "observed_result", "boundary_reason", "what_happened"
        )), "Action, result, description, and boundary explanation required")
        require(re.fullmatch(r"[a-z][a-z_]*", segment["action"]) is not None,
                "Action must be a lower-case operation identifier")
        for k in ("object_id", "destination_id"):
            require(segment[k] is None or nonempty(segment[k]), "Stable referent identifiers")
        require(type(segment["progress"]) is bool and type(segment["failure"]) is bool,
                "Independent boolean task labels required")
        cats = segment["categories"]
        require(isinstance(cats, list) and all(isinstance(c, str) for c in cats)
                and set(cats) <= CATEGORIES and len(cats) == len(set(cats)), "Failure categories")
        if segment["failure"]:
            require(bool(cats) and nonempty(segment["recovery"]), "Failure needs category and recovery")
            require(segment["severity"] in ("low", "medium", "catastrophic")
                    and segment["resetability"] in ("immediate", "minor_reset", "hard_reset", "unresettable"),
                    "Failure consequence fields")
        else:
            require(not cats and all(segment[k] is None for k in ("recovery", "severity", "resetability")),
                    "Nonfailure fields must be empty/null")
        evidence(segment["evidence"])
        require(type(segment["review_required"]) is bool and isinstance(segment["uncertainty"], str),
                "Uncertainty metadata")
        require(nonempty(segment["uncertainty"]) if segment["review_required"] else segment["uncertainty"] == "",
                "Review flag and uncertainty explanation must agree")
        require(segment["action"] != "unknown" or segment["review_required"], "Unknown operation requires review")
        same_action = previous is not None and previous["action_id"] == segment["action_id"]
        if same_action:
            require(previous["action"] == segment["action"],
                    "Linked action pieces must retain operation")
            # Null describes an unspecified/unresolved participant, not a different
            # physical participant. Keep known identities across occluded pieces
            # so A -> null -> B cannot silently change a linked action's object.
            for k in ("object_id", "destination_id"):
                if segment[k] is not None:
                    require(known_referents.get(k) in (None, segment[k]),
                            "Linked action pieces must retain known referents")
                    known_referents[k] = segment[k]
            require((previous["progress"], previous["failure"], set(previous["categories"])) !=
                    (segment["progress"], segment["failure"], set(cats)),
                    "Adjacent identical pieces of the same action must be merged")
        else:
            action_count += 1
            require(segment["action_id"] == f"a{action_count}", "Action IDs must be chronological contiguous blocks")
            known_referents = {k: segment[k] for k in ("object_id", "destination_id")
                               if segment[k] is not None}
        previous = segment
    require(cursor == ep["n_frames"], "Unlabeled episode tail")


def report():
    records = [common.load(p) for p in sorted((OUT / "attempts").glob("*.json"))]
    result = dict(
        planned=len(IDS)*len(MODELS), attempted=len(records), validated=sum(r["status"] == "validated" for r in records),
        generation_estimate_usd=sum(r.get("generation_estimate_usd", 0) for r in records),
        preflight_allowance_usd=sum(r.get("preflight_allowance_usd", 0) for r in records),
        unknown_reservations_usd=sum(r["reserved_usd"] for r in records
                                    if r.get("generation_sent") and "generation_estimate_usd" not in r),
        records=records,
    )
    transport.atomic_json(OUT / "summary.json", result)
    return result


def execute(model_key, ep, keys):
    name = ep["id"] + "_" + model_key
    slot = OUT / "attempts" / (name + ".json")
    record = dict(model_key=model_key, model=MODELS[model_key], episode=ep["id"],
                  reserved_usd=RESERVES[model_key], status="reserved",
                  generation_sent=False, started=time.time())
    try:
        with slot.open("x", encoding="utf-8") as f:
            json.dump(record, f)
    except FileExistsError:
        print(name + ": existing reservation; not resent", flush=True)
        return
    try:
        body = payload(model_key, ep)
        count_body = {k: body[k] for k in ("model", "input", "reasoning", "text", "truncation")}
        counted = transport.post("https://api.openai.com/v1/responses/input_tokens", "openai", count_body, keys)
        common.write_json(OUT / (name + "_count.json"), counted)
        tokens = counted["input_tokens"]
        if type(tokens) is not int or not 0 < tokens <= MAX_INPUT:
            raise ValueError("Input token ceiling")
        rates = RATES[model_key]
        allowance = tokens * rates["input"] / 1e6
        worst = (tokens * max(rates["input"], rates["cache_write"]) + MAX_OUTPUT * rates["output"]) / 1e6
        if allowance + worst > RESERVES[model_key]:
            raise ValueError("Preflight plus generation bound exceeds reservation")
        record.update(counted_input_tokens=tokens, preflight_allowance_usd=allowance,
                      maximum_generation_usd=worst, generation_sent=True, status="in_flight")
        transport.atomic_json(slot, record)
        print(f"{name}: generating ({tokens} input tokens)", flush=True)
        raw = transport.post("https://api.openai.com/v1/responses", "openai", body, keys)
        raw = json.loads(transport.redact(json.dumps(raw), keys))
        common.write_json(OUT / (name + "_raw.json"), raw)
        usage = transport.usage_cost("openai", raw)
        usage["generation_estimate_usd"] = round((
            (usage["input_tokens"] - usage["cached_tokens"] - usage["cache_write_tokens"]) * rates["input"]
            + usage["cached_tokens"] * rates["cached"] + usage["cache_write_tokens"] * rates["cache_write"]
            + usage["output_tokens_including_thinking"] * rates["output"]
        ) / 1e6, 8)
        record.update(usage, returned_model=raw.get("model"))
        answer, complete = transport.extract_answer("openai", raw)
        (OUT / (name + "_answer.txt")).write_text(answer, encoding="utf-8")
        if not complete:
            raise ValueError("Incomplete response; no retry")
        value = json.loads(answer)
        validate(value, ep)
        common.write_json(OUT / (name + "_annotation.json"), value)
        record["status"] = "validated"
    except Exception as exc:
        record.update(status="stopped", error=transport.redact(str(exc), keys)[:800])
    finally:
        record["elapsed_seconds"] = round(time.time() - record["started"], 2)
        transport.atomic_json(slot, record)
    print(name + ": " + record["status"] + (": " + record.get("error", "") if record["status"] != "validated" else ""), flush=True)


def run():
    manifest = verify()
    keys = {"openai": common.credentials()["openai"]}
    (OUT / "attempts").mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=2) as executor:
        jobs = [executor.submit(execute, model, ep, keys)
                for ep in manifest["episodes"] for model in MODELS]
        for job in as_completed(jobs):
            job.result()
            report()
    result = report()
    print(json.dumps({k: v for k, v in result.items() if k != "records"}), flush=True)
    return result["validated"] == result["planned"]



def validate(value, ep):
    if SCHEME == "atomic":
        return validate_atomic(value, ep)
    from annotation_states import validate
    return validate(value, ep)


def configure(out, source=None, episodes=None, scheme=None):
    global OUT, SOURCE, IDS, SCHEME, PROMPT_DIR
    OUT = Path(out).resolve()
    if source is not None:
        SOURCE = Path(source).resolve()
    if scheme is None and (OUT / "manifest.json").exists():
        scheme = common.load(OUT / "manifest.json")["settings"]["scheme"]
    SCHEME = scheme or "atomic"
    PROMPT_DIR = ROOT / "scripts/annotate/prompts" / ("atomic_actions" if SCHEME == "atomic" else "state_segments")
    if episodes is None and (OUT / "manifest.json").exists():
        episodes = [e["id"] for e in common.load(OUT / "manifest.json")["episodes"]]
    IDS = tuple(episodes or ("R001", "R002", "R003", "R004"))
    if len(set(IDS)) != len(IDS) or any(not re.fullmatch(r"[A-Za-z0-9_-]+", e) for e in IDS):
        raise ValueError("Episode IDs must be unique simple directory names")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("prepare", "verify", "run", "report"))
    p.add_argument("--scheme", choices=("atomic", "states"))
    p.add_argument("--source", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--episodes", nargs="+")
    args = p.parse_args()
    configure(args.out, args.source, args.episodes, args.scheme)
    result = globals()[args.command]()
    if args.command == "run" and not result:
        raise SystemExit(1)
