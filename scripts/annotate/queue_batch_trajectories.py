"""Run frozen Batch shards sequentially under an account's enqueued-token limit.

Only batches rejected for token_limit_exceeded before any request ran may be
re-enqueued. Completed requests, semantic errors and uncertain submissions are
never retried. Original rejection records are retained verbatim in queue_history.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
import uuid

import batch_trajectories as batch


def capacity_rejection(remote):
    errors = (remote.get("errors") or {}).get("data") or []
    counts = remote.get("request_counts") or {}
    return (remote.get("status") == "failed" and bool(errors)
            and all(e.get("code") == "token_limit_exceeded" for e in errors)
            and counts.get("completed") == 0 and counts.get("failed") == 0)


def states(out, manifest):
    rows = []
    for shard in manifest["shards"]:
        stem = Path(shard["path"]).stem
        path = out/"batch_jobs"/(stem+".json")
        state = batch.io.load(path) if path.exists() else None
        rows.append((shard, path, state))
    return rows


def queue_status(out, manifest):
    active, pending, complete, failed, unresolved = 0, 0, 0, 0, 0
    for shard, _, state in states(out, manifest):
        n = len(shard["custom_ids"])
        if state is None:
            pending += n
        elif not state.get("batch"):
            unresolved += n
        elif capacity_rejection(state["batch"]):
            pending += n
        elif state["batch"]["status"] not in batch.TERMINAL:
            active += n
        elif state["batch"]["status"] == "completed":
            complete += n
        else:
            failed += n
    summary_path = out/"batch_summary.json"
    summary = batch.io.load(summary_path) if summary_path.exists() else {}
    result = dict(planned=manifest["planned"], active_requests=active,
                  queued_locally=pending, completed_batch_requests=complete,
                  terminal_failed_requests=failed, unresolved_submission_requests=unresolved,
                  validated=summary.get("validated",0), stopped=summary.get("stopped",0),
                  generation_estimate_usd=summary.get("generation_estimate_usd",0),
                  updated_utc=datetime.now(timezone.utc).isoformat(),
                  controller_pid=os.getpid(), controller_sha256=batch.io.digest(Path(__file__)))
    batch.io.atomic_json(out/"queue_status.json",result)
    print(json.dumps(result),flush=True)
    return result


def enqueue_next(out, manifest, key):
    rows = states(out, manifest)
    if any(state and state.get("batch") and state["batch"]["status"] not in batch.TERMINAL
           for _,_,state in rows):
        return False
    if any(state is not None and not state.get("batch") for _,_,state in rows):
        raise ValueError("Unresolved prior submission; reconcile before making another paid request")
    for shard, state_path, state in rows:
        if state is not None and not capacity_rejection(state["batch"]):
            continue
        stem = Path(shard["path"]).stem
        history = out/"queue_history"
        history.mkdir(exist_ok=True)
        previous = []
        if state is not None:
            archive = history/(stem+"_"+state["batch"]["id"]+".json")
            if archive.exists() and batch.io.load(archive) != state:
                raise ValueError("A historical rejection changed")
            if not archive.exists():
                batch.io.write_json(archive,state)
            previous = state.get("prior_capacity_rejections",[]) + [archive.relative_to(out).as_posix()]
        upload_path = state_path.with_name(stem+"_upload.json")
        if upload_path.exists():
            uploaded = batch.io.load(upload_path)
        else:
            uploaded = batch.upload(out/shard["path"],key)
            batch.io.atomic_json(upload_path,uploaded)
        next_state = dict(status="submitting",input_file_id=uploaded["id"],
                          custom_ids=shard["custom_ids"],run_id=manifest["run_id"],
                          prior_capacity_rejections=previous,
                          request_id=str(uuid.uuid4()),started_utc=datetime.now(timezone.utc).isoformat())
        if state is None:
            with state_path.open("x",encoding="utf-8") as stream:
                json.dump(next_state,stream,indent=2)
        else:
            batch.io.atomic_json(state_path,next_state)
        body = dict(input_file_id=uploaded["id"],endpoint=manifest["endpoint"],completion_window="24h",
                    metadata=dict(run_id=manifest["run_id"],shard=stem,protocol="atomic_v8",
                                  capacity_attempt=str(len(previous)+1)))
        try:
            next_state["batch"] = batch.api("POST","/v1/batches",key,body,request_id=next_state["request_id"])
            next_state["status"] = "submitted"
        except Exception as exc:
            next_state.update(status="submission_unresolved",error=str(exc))
            raise
        finally:
            batch.io.atomic_json(state_path,next_state)
        print(f"{stem}: enqueued {len(shard['custom_ids'])} frozen requests as {next_state['batch']['id']}",flush=True)
        return True
    return False


def run(out, interval=60):
    _,manifest = batch.verify(out)
    (out/"batch_jobs").mkdir(exist_ok=True)
    lock = out/"queue_worker.lock"
    with lock.open("x",encoding="utf-8") as stream:
        json.dump(dict(pid=os.getpid(),started_utc=datetime.now(timezone.utc).isoformat()),stream)
    try:
        key = batch.io.credentials()["openai"]
        while True:
            try:
                batch.collect(out)
            except (OSError,RuntimeError) as exc:
                print(f"Read-only collection interrupted: {exc}; checking again later",flush=True)
                time.sleep(interval)
                continue
            status = queue_status(out,manifest)
            if status["unresolved_submission_requests"]:
                raise ValueError("Unresolved submission; no automatic resend")
            if not status["active_requests"] and not status["queued_locally"]:
                print("Queue finished; inspect validation counts in batch_summary.json",flush=True)
                return
            if not status["active_requests"]:
                enqueue_next(out,manifest,key)
                queue_status(out,manifest)
            time.sleep(interval)
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--interval",type=int,default=60)
    args = parser.parse_args()
    run(args.out.resolve(),args.interval)
