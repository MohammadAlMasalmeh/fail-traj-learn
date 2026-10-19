"""Explicitly resume the authorized Sol/V8 run after reloading API credits.

Preserve rejected submission journals, retry only unsuccessful episodes in a
separate frozen run, then create an index that references original artifacts.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

import queue_batch_trajectories as queue

batch = queue.batch


def recover_billing_journal(path):
    state = batch.io.load(path)
    if state.get("batch"):
        return False
    prefix = "API HTTP 400: "
    error = state.get("error", "")
    if not error.startswith(prefix):
        raise ValueError("Ambiguous submission outcome; cannot resend")
    details = json.loads(error[len(prefix):]).get("error", {})
    if details.get("code") != "billing_hard_limit_reached" or state.get("status") != "submission_unresolved":
        raise ValueError("Only a definite HTTP 400 billing rejection may be reset")
    history = path.parent.parent/"queue_history"
    history.mkdir(exist_ok=True)
    archived = history/(path.stem+"_billing_rejected_"+state["request_id"]+".json")
    if archived.exists():
        raise ValueError("Submission journal has already been archived; inspect before proceeding")
    path.rename(archived)
    print(f"Preserved definite billing rejection: {archived.name}", flush=True)
    return True


def request_hashes(out, wanted):
    result = {}
    for path in sorted((out/"batch_inputs").glob("*.jsonl")):
        with path.open("rb") as stream:
            for line in stream:
                cid = json.loads(line)["custom_id"]
                if cid in wanted:
                    if cid in result:
                        raise ValueError("Duplicate frozen request")
                    result[cid] = hashlib.sha256(line).hexdigest()
    return result


def prepare(main, retry):
    source, manifest = batch.verify(main)
    if (main/"queue_worker.lock").exists():
        raise ValueError("A queue lock exists; verify that no worker is active before resuming")
    # Read-only remote reconciliation first: never assume a lost create reply
    # means the service did not accept a paid batch.
    batch.reconcile(main, manifest, batch.io.credentials()["openai"])
    for _, path, state in queue.states(main, manifest):
        if state is not None and not state.get("batch"):
            recover_billing_journal(path)
    failed = []
    for path in sorted((main/"batch_results").glob("*_record.json")):
        record = batch.io.load(path)
        if record["status"] != "validated":
            if (main/(record["episode"]+"_sol_annotation.json")).exists():
                raise ValueError("Cannot retry an episode with an existing accepted annotation")
            failed.append(record)
    ids = {r["episode"] for r in failed}
    if retry.exists():
        retry_source, _ = batch.verify(retry)
        if {e["id"] for e in retry_source["episodes"]} != ids:
            raise ValueError("Existing recovery run differs from unsuccessful episode set")
        return
    if not ids:
        raise ValueError("No failed episodes for a separate recovery run")
    retry.mkdir()
    subset = deepcopy(source)
    subset["episodes"] = [e for e in source["episodes"] if e["id"] in ids]
    subset["settings"]["maximum_reserved_usd"] = len(ids)*sum(subset["settings"]["reservations"].values())
    subset["files"] = {name:sha for name,sha in source["files"].items()
                       if name == "prompt.txt" or name.split("/",1)[0] in ids}
    for name in subset["files"]:
        target = retry/name
        target.parent.mkdir(parents=True,exist_ok=True)
        os.link(main/name,target)
    batch.io.write_json(retry/"manifest.json",subset)
    batch.io.write_json(retry/"recovery_provenance.json",dict(
        parent=str(main),parent_batch_manifest_sha256=batch.io.digest(main/"batch_manifest.json"),
        authorization="User reloaded the balance and explicitly requested resume on 2026-10-18.",
        unsuccessful_original_records=failed,
        limitation="One fresh attempt for each unsuccessful episode; original answers, failures and accepted annotations remain unchanged."))
    frozen = batch.freeze_shards(retry,max_requests=10)
    wanted = {e+"_sol" for e in ids}
    if request_hashes(main,wanted) != request_hashes(retry,wanted):
        raise ValueError("Recovery requests differ from original frozen request bytes")
    print(json.dumps(dict(recovery_episodes=sorted(ids),requests=frozen["planned"],
                          same_request_bytes=True)),flush=True)


def completion_index(main,retry):
    manifest = batch.io.load(main/"manifest.json")
    rows,missing = [],[]
    for ep in manifest["episodes"]:
        name = ep["id"]+"_sol_annotation.json"
        candidates = [p for root in (main,retry) if (p:=root/name).exists()]
        if len(candidates)>1:
            raise ValueError("Two accepted annotations found for one episode")
        if not candidates:
            missing.append(ep["id"])
        else:
            path = candidates[0]
            rows.append(dict(episode=ep["id"],annotation=str(path),sha256=batch.io.digest(path),
                             source_run=str(path.parent),quality="schema_validated_teacher_proposal"))
    result = dict(planned=len(manifest["episodes"]),validated=len(rows),missing=missing,
                  generation_estimate_usd=sum(batch.io.load(root/"batch_summary.json")["generation_estimate_usd"]
                                              for root in (main,retry)),episodes=rows)
    batch.io.atomic_json(main.parent/"completion.json",result)
    print(json.dumps({k:v for k,v in result.items() if k!="episodes"}),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("prepare","run"))
    parser.add_argument("--main",type=Path,required=True)
    parser.add_argument("--retry",type=Path,required=True)
    args = parser.parse_args()
    main,retry = args.main.resolve(),args.retry.resolve()
    if args.command == "prepare":
        prepare(main,retry)
    else:
        queue.run(main)
        queue.run(retry)
        completion_index(main,retry)
