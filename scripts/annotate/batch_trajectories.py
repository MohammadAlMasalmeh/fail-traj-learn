"""Freeze, submit, retrieve and validate Sol/V8 annotations through the Batch API.

Existing synchronous runs and prompt bytes remain untouched. Submissions are
journaled before each billable create call; uncertain calls are never resent.
Prepare/verify are offline. Submit spends API credits; collect is read-only remotely.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time
import urllib.error
import urllib.request
import uuid

import annotate_trajectories as annotation
import annotation_runtime as io

V8_SHA256 = "e7ed5aae4443ce1ee721bacf6746c9abef4b1c2d11f18039414d5a00811ce20e"
TERMINAL = {"completed", "failed", "expired", "cancelled"}
BATCH_RATES = dict(input=1.0, cached=.05, cache_write=1.25, output=5.0)


def configure(out):
    annotation.configure(out, models=["sol"], scheme="atomic", reasoning="low")


def freeze_shards(out, max_bytes=150_000_000, max_requests=50):
    configure(out)
    manifest = annotation.verify()
    if io.digest(out / "prompt.txt") != V8_SHA256:
        raise ValueError("Prompt differs from the frozen atomic V8 prompt")
    if (out / "batch_manifest.json").exists():
        raise ValueError("Batch inputs already frozen")
    shards_dir = out / "batch_inputs"
    shards_dir.mkdir(exist_ok=False)
    shards = []
    stream, size, ids = None, 0, []

    def close_shard():
        if stream is not None:
            stream.close()
            shards.append(dict(path=path.relative_to(out).as_posix(), bytes=size,
                               sha256=io.digest(path), custom_ids=list(ids)))

    try:
        for ep in manifest["episodes"]:
            body = annotation.payload("sol", ep)
            # Batch selects its processing tier at the batch endpoint.
            body.pop("service_tier", None)
            custom_id = ep["id"] + "_sol"
            line = (json.dumps(dict(custom_id=custom_id, method="POST", url="/v1/responses", body=body),
                               separators=(",", ":")) + "\n").encode("utf-8")
            if len(line) > max_bytes:
                raise ValueError(f"One request exceeds the shard size limit: {custom_id}")
            if stream is None or size+len(line)>max_bytes or len(ids)>=max_requests:
                close_shard()
                path = shards_dir / f"batch_{len(shards)+1:03d}.jsonl"
                stream, size, ids = path.open("xb"), 0, []
            stream.write(line)
            size += len(line)
            ids.append(custom_id)
        close_shard()
        stream = None
    finally:
        if stream is not None:
            stream.close()
    batch = dict(version="sol_v8_batch_v1", run_id=uuid.uuid4().hex,
                 prepared_utc=datetime.now(timezone.utc).isoformat(),
                 model="gpt-6.1-sol", endpoint="/v1/responses", completion_window="24h",
                 prompt_sha256=V8_SHA256, source_manifest_sha256=io.digest(out/"manifest.json"),
                 runner_sha256=io.digest(Path(__file__)), rates_per_million=BATCH_RATES,
                 pricing_checked="2026-10-15",
                 pricing_source="https://developers.openai.com/api/docs/models/gpt-6.1-sol",
                 selection_sha256=io.digest(out/"selection.json") if (out/"selection.json").exists() else None,
                 planned=len(manifest["episodes"]), shards=shards, automatic_retries=0,
                 limitations="Teacher-generated proposals, not human ground truth. Review flags must remain excluded from definitive supervision until adjudicated.")
    io.write_json(out / "batch_manifest.json", batch)
    return batch


def verify(out):
    configure(out)
    source = annotation.verify()
    batch = io.load(out / "batch_manifest.json")
    if (batch["source_manifest_sha256"] != io.digest(out / "manifest.json")
            or batch["prompt_sha256"] != V8_SHA256
            or io.digest(out / "prompt.txt") != V8_SHA256
            or batch["runner_sha256"] != io.digest(Path(__file__))
            or batch["rates_per_million"] != BATCH_RATES):
        raise ValueError("Frozen batch provenance changed")
    if batch.get("selection_sha256") is not None and io.digest(out/"selection.json") != batch["selection_sha256"]:
        raise ValueError("Frozen episode selection changed")
    ids = []
    for shard in batch["shards"]:
        path = (out / shard["path"]).resolve()
        if not path.is_relative_to(out.resolve()) or io.digest(path) != shard["sha256"]:
            raise ValueError("Frozen Batch request bytes changed")
        if path.stat().st_size != shard["bytes"] or shard["bytes"] > 200_000_000:
            raise ValueError("Invalid Batch shard size")
        actual = []
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if (row["method"] != "POST" or row["url"] != "/v1/responses"
                        or row["body"]["model"] != "gpt-6.1-sol"):
                    raise ValueError("Unexpected Batch request")
                actual.append(row["custom_id"])
        if actual != shard["custom_ids"]:
            raise ValueError("Batch custom IDs differ from frozen inventory")
        ids.extend(actual)
    expected = [ep["id"]+"_sol" for ep in source["episodes"]]
    if ids != expected or len(set(ids)) != len(ids) or batch["planned"] != len(ids):
        raise ValueError("Duplicate, missing or unexpected episode requests")
    return source, batch


def api(method, path, key, body=None, content_type="application/json", raw=False, request_id=None):
    if not re.fullmatch(r"/v1/(?:files|batches)(?:/[A-Za-z0-9_-]+)?(?:/content)?(?:\?limit=100(?:&after=[A-Za-z0-9_-]+)?)?", path):
        raise ValueError("Unexpected API path")
    data = json.dumps(body).encode() if isinstance(body, dict) else body
    headers = {"Authorization": "Bearer "+key}
    if data is not None:
        headers["Content-Type"] = content_type
    if request_id:
        headers["X-Client-Request-Id"] = request_id
        headers["Idempotency-Key"] = request_id
    request = urllib.request.Request("https://api.openai.com"+path, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            result = response.read()
        return result if raw else json.loads(result)
    except urllib.error.HTTPError as exc:
        message = io.redact(exc.read().decode("utf-8", "replace"), {"openai":key})
        raise RuntimeError(f"API HTTP {exc.code}: {message[:1200]}") from None
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"API {type(exc).__name__}; outcome may be unknown; no automatic resend") from None


def upload(path, key):
    boundary = "batch-"+uuid.uuid4().hex
    prefix = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\nbatch\r\n"
              f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{path.name}\"\r\n"
              "Content-Type: application/jsonl\r\n\r\n").encode()
    body = prefix + path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    return api("POST", "/v1/files", key, body, f"multipart/form-data; boundary={boundary}")


def reconcile(out, batch, key):
    """Read-only recovery of a create response lost after remote acceptance."""
    pending = {Path(s["path"]).stem:s for s in batch["shards"]
               if (out / "batch_jobs" / (Path(s["path"]).stem+".json")).exists()}
    if not pending:
        return
    after = None
    while True:
        page = api("GET", "/v1/batches?limit=100"+("&after="+after if after else ""), key)
        for remote in page["data"]:
            metadata = remote.get("metadata") or {}
            shard_id = metadata.get("shard")
            if metadata.get("run_id") != batch["run_id"] or shard_id not in pending:
                continue
            state_path = out / "batch_jobs" / (shard_id+".json")
            state = io.load(state_path)
            if state.get("batch") and state["batch"]["id"] != remote["id"]:
                raise ValueError("Multiple remote batches found for one frozen shard")
            if remote["input_file_id"] != state["input_file_id"]:
                raise ValueError("Remote batch input differs from submission journal")
            state["batch"] = remote
            state["status"] = "submitted"
            io.atomic_json(state_path, state)
        if not page.get("has_more"):
            break
        after = page["data"][-1]["id"]


def submit(out):
    _, batch = verify(out)
    key = io.credentials()["openai"]
    jobs = out / "batch_jobs"
    jobs.mkdir(exist_ok=True)
    for shard in batch["shards"]:
        stem = Path(shard["path"]).stem
        state_path = jobs / (stem+".json")
        if state_path.exists():
            state = io.load(state_path)
            if state.get("batch"):
                print(f"{stem}: already submitted as {state['batch']['id']}", flush=True)
                continue
            raise ValueError(f"{stem}: prior submission outcome unresolved; run reconcile, never blindly resend")
        upload_path = jobs / (stem+"_upload.json")
        if upload_path.exists():
            uploaded = io.load(upload_path)
        else:
            uploaded = upload(out / shard["path"], key)
            io.atomic_json(upload_path, uploaded)
        state = dict(status="submitting", input_file_id=uploaded["id"],
                     custom_ids=shard["custom_ids"], run_id=batch["run_id"],
                     request_id=str(uuid.uuid4()), started_utc=datetime.now(timezone.utc).isoformat())
        # Exclusive journal before the billable mutation prevents concurrent or
        # accidental repeated runs from buying duplicate annotations.
        with state_path.open("x", encoding="utf-8") as stream:
            json.dump(state, stream, indent=2)
        body = dict(input_file_id=uploaded["id"], endpoint=batch["endpoint"],
                    completion_window="24h", metadata=dict(run_id=batch["run_id"], shard=stem,
                    protocol="atomic_v8", model="gpt-6.1-sol"))
        try:
            state["batch"] = api("POST", "/v1/batches", key, body, request_id=state["request_id"])
            state["status"] = "submitted"
        except Exception as exc:
            state.update(status="submission_unresolved", error=str(exc))
            raise
        finally:
            io.atomic_json(state_path, state)
        print(f"{stem}: {len(shard['custom_ids'])} requests submitted as {state['batch']['id']}", flush=True)
    return report(out)


def estimate_usage(body):
    usage = io.usage_cost("openai", body)
    usage["generation_estimate_usd"] = round((
        (usage["input_tokens"]-usage["cached_tokens"]-usage["cache_write_tokens"])*BATCH_RATES["input"]
        + usage["cached_tokens"]*BATCH_RATES["cached"]
        + usage["cache_write_tokens"]*BATCH_RATES["cache_write"]
        + usage["output_tokens_including_thinking"]*BATCH_RATES["output"])/1e6, 8)
    return usage


def ingest(out, source, batch):
    episodes = {ep["id"]+"_sol":ep for ep in source["episodes"]}
    expected_by_file = {}
    for shard in batch["shards"]:
        stem = Path(shard["path"]).stem
        for suffix in ("output", "error"):
            expected_by_file[out/"batch_results"/f"{stem}_{suffix}.jsonl"] = set(shard["custom_ids"])
    rows = {}
    for path, expected in expected_by_file.items():
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                custom_id = row.get("custom_id")
                if custom_id not in expected or custom_id in rows:
                    raise ValueError("Unknown or duplicate result custom_id")
                rows[custom_id] = row
    for custom_id, row in rows.items():
        ep = episodes[custom_id]
        record = dict(episode=ep["id"], model_key="sol", model="gpt-6.1-sol",
                      transport="batch", status="stopped", custom_id=custom_id)
        response = row.get("response") or {}
        body = response.get("body")
        io.atomic_json(out/(custom_id+"_batch_result.json"), row)
        try:
            if response.get("status_code") != 200 or row.get("error"):
                raise ValueError("Batch request error: "+json.dumps(row.get("error") or body)[:800])
            io.atomic_json(out/(custom_id+"_raw.json"), body)
            record.update(estimate_usage(body), returned_model=body.get("model"))
            if body.get("model") != "gpt-6.1-sol" and not str(body.get("model", "")).startswith("gpt-6.1-sol-"):
                raise ValueError("Unexpected returned model")
            answer, completed = io.extract_answer("openai", body)
            (out/(custom_id+"_answer.txt")).write_text(answer, encoding="utf-8")
            if not completed:
                raise ValueError("Incomplete response; no automatic retry")
            value = json.loads(answer)
            annotation.validate(value, ep)
            io.atomic_json(out/(custom_id+"_annotation.json"), value)
            record.update(status="validated", review_required_segments=sum(s["review_required"] for s in value["segments"]),
                          segments=len(value["segments"]))
        except (ValueError, KeyError, TypeError) as exc:
            record["error"] = str(exc)[:1000]
        io.atomic_json(out/"batch_results"/(custom_id+"_record.json"), record)


def collect(out):
    source, batch = verify(out)
    key = io.credentials()["openai"]
    results = out / "batch_results"
    results.mkdir(exist_ok=True)
    for shard in batch["shards"]:
        stem = Path(shard["path"]).stem
        path = out/"batch_jobs"/(stem+".json")
        if not path.exists() or not io.load(path).get("batch"):
            continue
        state = io.load(path)
        remote = api("GET", "/v1/batches/"+state["batch"]["id"], key)
        state["batch"] = remote
        io.atomic_json(path, state)
        print(f"{stem}: {remote['status']} {remote.get('request_counts', {})}", flush=True)
        if remote["status"] not in TERMINAL:
            continue
        for kind in ("output", "error"):
            file_id = remote.get(kind+"_file_id")
            target = results / f"{stem}_{kind}.jsonl"
            receipt = results / f"{stem}_{kind}_receipt.json"
            if file_id and not target.exists():
                content = api("GET", "/v1/files/"+file_id+"/content", key, raw=True)
                temp = target.with_suffix(".tmp")
                temp.write_bytes(content)
                temp.replace(target)
                io.atomic_json(receipt, dict(file_id=file_id, sha256=io.digest(target)))
            elif file_id and (not receipt.exists() or io.load(receipt) != dict(file_id=file_id, sha256=io.digest(target))):
                raise ValueError("Downloaded result bytes or file provenance changed")
    ingest(out, source, batch)
    return report(out)


def report(out):
    batch = io.load(out/"batch_manifest.json")
    states = [io.load(out/"batch_jobs"/(Path(s["path"]).stem+".json"))
              for s in batch["shards"] if (out/"batch_jobs"/(Path(s["path"]).stem+".json")).exists()]
    records = [io.load(p) for p in sorted((out/"batch_results").glob("*_record.json"))]
    result = dict(planned=batch["planned"], shards=len(batch["shards"]),
                  submitted=sum(len(s["custom_ids"]) for s in states if s.get("batch")),
                  remote_statuses=dict((s["batch"]["id"],s["batch"]["status"]) for s in states if s.get("batch")),
                  validated=sum(r["status"]=="validated" for r in records),
                  stopped=sum(r["status"]=="stopped" for r in records),
                  not_yet_returned=batch["planned"]-len(records),
                  generation_estimate_usd=sum(r.get("generation_estimate_usd",0) for r in records),
                  review_required_segments=sum(r.get("review_required_segments",0) for r in records),
                  records=records)
    io.atomic_json(out/"batch_summary.json", result)
    print(json.dumps({k:v for k,v in result.items() if k!="records"}), flush=True)
    return result


def watch(out):
    """Retrieve the already submitted jobs; this never submits paid requests."""
    while True:
        try:
            result = collect(out)
            if (result["submitted"] == result["planned"]
                    and len(result["remote_statuses"]) == result["shards"]
                    and all(status in TERMINAL for status in result["remote_statuses"].values())):
                print("All submitted batches reached a terminal status; inspect batch_summary.json for validation and errors.", flush=True)
                return
        except (OSError, RuntimeError) as exc:
            print(f"Read-only collection interrupted: {exc}; will check again", flush=True)
        time.sleep(60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify", "submit", "reconcile", "collect", "report", "watch"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--selection", type=Path)
    args = parser.parse_args()
    out = args.out.resolve()
    if args.command == "prepare":
        if args.source is None or args.selection is None:
            parser.error("prepare requires --source and --selection")
        selection = io.load(args.selection)
        ids = [row["id"] for row in selection["splits"]["train"]]
        if len(ids) != 500 or selection["count"] != 500:
            raise ValueError("This authorized run requires exactly 500 distinct training episodes")
        split_keys = {}
        for split, rows in selection["splits"].items():
            keys = {(r["dataset"],r["episode_index"]) for r in rows}
            if len(keys) != len(rows) or any(keys & previous for previous in split_keys.values()):
                raise ValueError("Duplicate episode identity or split leakage")
            split_keys[split] = keys
        annotation.configure(out, source=args.source, episodes=ids, scheme="atomic", models=["sol"], reasoning="low")
        annotation.prepare()
        (out/"selection.json").write_bytes(args.selection.read_bytes())
        batch = freeze_shards(out)
        io.write_json(out/"selection_provenance.json", dict(path=str(args.selection.resolve()), sha256=io.digest(args.selection)))
        print(json.dumps(dict(planned=batch["planned"], shards=len(batch["shards"]),
                             bytes=sum(s["bytes"] for s in batch["shards"]))), flush=True)
    elif args.command == "verify":
        _, batch = verify(out)
        print(json.dumps(dict(planned=batch["planned"], shards=len(batch["shards"]), verified=True)))
    elif args.command == "reconcile":
        _, batch = verify(out)
        reconcile(out, batch, io.credentials()["openai"])
        report(out)
    else:
        globals()[args.command](out)
