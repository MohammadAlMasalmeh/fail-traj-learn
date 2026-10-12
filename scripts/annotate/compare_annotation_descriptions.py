"""Blinded, order-balanced LLM agreement pilot on saved atomic descriptions.

G-Eval-inspired rubric evaluation, not the original probability-weighted metric.
No annotation regeneration, prompt edits, video accuracy judging, or retries.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time

import annotate_trajectories as source

common, transport = source.common, source.transport
ROOT = source.ROOT
OUT = ROOT / "outputs/annotation_description_comparison"
RUBRIC = ROOT / "scripts/annotate/prompts/description_equivalence.txt"
MODEL = "gpt-6-astra"
MAX_INPUT, MAX_OUTPUT, RESERVE = 10000, 8192, .70
VERDICTS = {5: "equivalent", 4: "minor_difference", 3: "material_difference",
            2: "contradiction", 1: "unrelated"}


def settings():
    return dict(model=MODEL, reasoning="medium", max_input_tokens=MAX_INPUT,
                max_output_tokens=MAX_OUTPUT, reservation_per_call_usd=RESERVE,
                workers=2, retries=0, input_orders=["ab", "ba"],
                rates_per_million=source.RATES["astra"],
                pricing_source="https://developers.openai.com/api/docs/models/gpt-6-astra",
                price_checked="2026-10-10")


def hashes():
    paths = [Path(__file__), RUBRIC, Path(common.__file__), Path(transport.__file__),
             source.PROMPT_DIR / "annotation.txt", source.OUT / "prompt.txt",
             source.OUT / "manifest.json"]
    paths += [source.OUT / f"{eid}_{model}_annotation.json"
              for eid in source.IDS for model in source.MODELS]
    return {p.relative_to(ROOT).as_posix(): common.digest(p) for p in paths}


def overlapping_pairs(eid, left, right):
    pairs = []
    for i, a in enumerate(left["segments"]):
        for j, b in enumerate(right["segments"]):
            start = max(a["start_frame"], b["start_frame"])
            end = min(a["end_frame"], b["end_frame"])
            if start < end:
                pairs.append(dict(pair_id=f"p{len(pairs)+1:03d}", episode=eid,
                                  left_index=i, right_index=j, start_frame=start,
                                  end_frame=end, left=a["what_happened"],
                                  right=b["what_happened"]))
    return pairs


def controls():
    a = "The robot grasps carton_1 and retains control while moving it toward basket_1."
    cases = [
        ("paraphrase", a,
         "The robot takes hold of carton_1 and carries it toward basket_1 without losing its grip.", 5, 5),
        ("negation", "The robot grasps carton_1 successfully.",
         "The robot fails to grasp carton_1.", 1, 2),
        ("wrong_object", "The robot grasps carton_1.",
         "The robot grasps bottle_1 instead of carton_1.", 1, 2),
        ("attempt_vs_success", "The robot attempts to grasp carton_1 but acquires no control.",
         "The robot grasps carton_1 and establishes control.", 1, 2),
        ("task_vs_physical", "The robot grasps the wrong carton successfully, without advancing the task.",
         "The robot grasps the requested carton successfully and advances the task.", 1, 2),
    ]
    return [dict(pair_id=f"p{i+1:03d}", name=name, left=a, right=b,
                 expected_min=lo, expected_max=hi)
            for i, (name, a, b, lo, hi) in enumerate(cases)]


def prepare():
    original = source.verify()
    if source.SCHEME != "atomic":
        raise ValueError("Description comparison currently requires atomic-action annotations")
    if OUT.exists():
        raise ValueError("Fresh evaluation directory required; existing calls are never overwritten")
    batches = []
    for ep in original["episodes"]:
        left, right = [common.load(source.OUT / f"{ep['id']}_{m}_annotation.json")
                       for m in ("sol", "astra")]
        for value in (left, right):
            source.validate(value, ep)
        pairs = overlapping_pairs(ep["id"], left, right)
        if sum(p["end_frame"] - p["start_frame"] for p in pairs) != ep["n_frames"]:
            raise ValueError("Paired descriptions must partition the original frames")
        batches.append(dict(id=ep["id"], task=ep["task"], pairs=pairs))
    batches.append(dict(id="controls", task="Put the requested carton in basket_1.", pairs=controls()))
    OUT.mkdir(parents=True)
    (OUT / "attempts").mkdir()
    shutil.copyfile(RUBRIC, OUT / "rubric.txt")
    common.write_json(OUT / "manifest.json", dict(
        settings=settings(), source_hashes=hashes(), batches=batches,
        prepared_utc=datetime.now(timezone.utc).isoformat(),
        method="G-Eval-inspired rubric judge; integer ordinal scores, averaged over input order; no token probability weighting",
        method_source="https://aclanthology.org/2023.emnlp-main.153/",

        limitations="Description agreement, not video accuracy. Judge is also one of the annotation models; blinding and order balancing do not remove all judge bias.",
    ))
    print(json.dumps(dict(pairs={b["id"]: len(b["pairs"]) for b in batches},
                          calls=len(batches)*2, maximum_reserved_usd=len(batches)*2*RESERVE)))


def verify():
    manifest = common.load(OUT / "manifest.json")
    if manifest["settings"] != settings() or manifest["source_hashes"] != hashes():
        raise ValueError("Frozen evaluator, rubric, annotation prompt, or saved predictions changed")
    if (OUT / "rubric.txt").read_bytes() != RUBRIC.read_bytes():
        raise ValueError("Frozen rubric changed")
    return manifest


def payload(batch, order):
    pairs = [dict(pair_id=p["pair_id"], A=p["left" if order == "ab" else "right"],
                  B=p["right" if order == "ab" else "left"]) for p in batch["pairs"]]
    # No model names, structured annotation labels, expected control scores,
    # segment numbers, or previous judgments reach the judge.
    content = json.dumps(dict(task_context=batch["task"], pairs=pairs), ensure_ascii=False)
    return dict(model=MODEL, input=[
        dict(role="developer", content=(OUT / "rubric.txt").read_text(encoding="utf-8")),
        dict(role="user", content=content)], reasoning=dict(effort="medium"),
        max_output_tokens=MAX_OUTPUT, text=dict(format=dict(type="json_object")),
        store=False, service_tier="default", truncation="disabled")


def validate(value, batch):
    if not isinstance(value, dict) or set(value) != {"evaluations"}:
        raise ValueError("Judge envelope")
    rows = value["evaluations"]
    if not isinstance(rows, list) or len(rows) != len(batch["pairs"]):
        raise ValueError("Judge pair count")
    found = set()
    for r in rows:
        if not isinstance(r, dict) or set(r) != {"pair_id", "score", "verdict", "explanation", "differences"}:
            raise ValueError("Judge fields")
        if type(r["score"]) is not int or r["score"] not in VERDICTS or r["verdict"] != VERDICTS[r["score"]]:
            raise ValueError("Score and verdict")
        if not isinstance(r["pair_id"], str) or r["pair_id"] in found:
            raise ValueError("Duplicate or invalid pair ID")
        found.add(r["pair_id"])
        if not isinstance(r["explanation"], str) or not r["explanation"].strip():
            raise ValueError("Missing explanation")
        if not isinstance(r["differences"], list) or any(not isinstance(x, str) or not x.strip() for x in r["differences"]):
            raise ValueError("Differences format")
        if (r["score"] == 5) != (len(r["differences"]) == 0):
            raise ValueError("Difference list conflicts with score")
    if found != {p["pair_id"] for p in batch["pairs"]}:
        raise ValueError("Unexpected pair IDs")


def execute(batch, order, keys):
    name = batch["id"] + "_" + order
    slot = OUT / "attempts" / (name + ".json")
    record = dict(id=name, reserved_usd=RESERVE, status="reserved",
                  generation_sent=False, started=time.time())
    try:
        with slot.open("x", encoding="utf-8") as f:
            json.dump(record, f)
    except FileExistsError:
        print(name + ": existing reservation; not resent", flush=True)
        return
    try:
        body = payload(batch, order)
        common.write_json(OUT / (name + "_request.json"), body)
        count_body = {k: body[k] for k in ("model", "input", "reasoning", "text", "truncation")}
        count = transport.post("https://api.openai.com/v1/responses/input_tokens", "openai", count_body, keys)
        common.write_json(OUT / (name + "_count.json"), count)
        tokens = count["input_tokens"]
        if type(tokens) is not int or not 0 < tokens <= MAX_INPUT:
            raise ValueError("Input token ceiling")
        rates = settings()["rates_per_million"]
        allowance = tokens*rates["input"]/1e6
        worst = (tokens*max(rates["input"], rates["cache_write"]) + MAX_OUTPUT*rates["output"])/1e6
        if allowance + worst > RESERVE:
            raise ValueError("Preflight plus generation bound exceeds reservation")
        record.update(counted_input_tokens=tokens, preflight_allowance_usd=allowance,
                      generation_sent=True, status="in_flight")
        transport.atomic_json(slot, record)
        print(name + ": generating", flush=True)
        raw = transport.post("https://api.openai.com/v1/responses", "openai", body, keys)
        raw = json.loads(transport.redact(json.dumps(raw), keys))
        common.write_json(OUT / (name + "_raw.json"), raw)
        usage = transport.usage_cost("openai", raw)
        usage["generation_estimate_usd"] = round((
            (usage["input_tokens"]-usage["cached_tokens"]-usage["cache_write_tokens"])*rates["input"]
            + usage["cached_tokens"]*rates["cached"] + usage["cache_write_tokens"]*rates["cache_write"]
            + usage["output_tokens_including_thinking"]*rates["output"])/1e6, 8)
        record.update(usage, returned_model=raw.get("model"))
        answer, complete = transport.extract_answer("openai", raw)
        (OUT / (name + "_answer.txt")).write_text(answer, encoding="utf-8")
        if not complete:
            raise ValueError("Incomplete response; no retry")
        value = json.loads(answer)
        validate(value, batch)
        common.write_json(OUT / (name + "_evaluation.json"), value)
        record["status"] = "validated"
    except Exception as exc:
        record.update(status="stopped", error=transport.redact(str(exc), keys)[:800])
    finally:
        record["elapsed_seconds"] = round(time.time()-record["started"], 2)
        transport.atomic_json(slot, record)
    print(name + ": " + record["status"] + (": " + record.get("error", "") if record["status"] != "validated" else ""), flush=True)


def report():
    records = [common.load(p) for p in sorted((OUT / "attempts").glob("*.json"))]
    summary = dict(planned=len(common.load(OUT / "manifest.json")["batches"])*2, attempted=len(records),
                   validated=sum(r["status"] == "validated" for r in records),
                   generation_estimate_usd=sum(r.get("generation_estimate_usd", 0) for r in records),
                   preflight_allowance_usd=sum(r.get("preflight_allowance_usd", 0) for r in records),
                   unknown_reservations_usd=sum(r["reserved_usd"] for r in records
                       if r["generation_sent"] and "generation_estimate_usd" not in r), records=records)
    transport.atomic_json(OUT / "summary.json", summary)
    return summary


def combine():
    manifest = verify()
    rows, calibration = [], []
    artifact_hashes = {}
    for batch in manifest["batches"]:
        evaluated = []
        for order in ("ab", "ba"):
            name = batch["id"] + "_" + order
            slot = OUT / "attempts" / (name + ".json")
            if common.load(slot)["status"] != "validated":
                raise ValueError("Both orders must validate before publishing scores")
            raw = common.load(OUT / (name + "_raw.json"))
            answer, complete = transport.extract_answer("openai", raw)
            value = common.load(OUT / (name + "_evaluation.json"))
            if not complete or json.loads(answer) != value:
                raise ValueError("Evaluation differs from completed raw response")
            validate(value, batch)
            evaluated.append({r["pair_id"]: r for r in value["evaluations"]})
            for suffix in ("_request.json", "_raw.json", "_evaluation.json"):
                path = OUT / (name + suffix)
                artifact_hashes[path.name] = common.digest(path)
            artifact_hashes["attempts/" + slot.name] = common.digest(slot)
        for p in batch["pairs"]:
            ab, ba = [x[p["pair_id"]] for x in evaluated]
            result = dict(**p, score=(ab["score"]+ba["score"])/2,
                          order_disagreement=ab["score"] != ba["score"], judgments=dict(ab=ab, ba=ba))
            if batch["id"] == "controls":
                result["passed"] = all(p["expected_min"] <= r["score"] <= p["expected_max"] for r in (ab, ba))
                calibration.append(result)
            else:
                rows.append(result)
    summary = report()
    result = dict(run="atomic_actions", model=MODEL, method=manifest["method"],
                  limitations=manifest["limitations"], scale=VERDICTS,
                  source_hashes=manifest["source_hashes"], artifact_hashes=artifact_hashes,
                  calibration=calibration, calibration_passed=all(c["passed"] for c in calibration),
                  cost={k: v for k, v in summary.items() if k != "records"}, pairs=rows)
    common.write_json(OUT / "llm_semantic_similarity.json", result)
    print(json.dumps(dict(pairs=len(rows), controls_passed=result["calibration_passed"],
                          order_disagreements=sum(r["order_disagreement"] for r in rows), **result["cost"])))
    return result


def export_viewer(target):
    path = OUT / "llm_semantic_similarity.json"
    if not path.exists():
        return
    manifest = verify()
    value = common.load(path)
    if value["source_hashes"] != manifest["source_hashes"]:
        raise ValueError("Evaluation source mismatch")
    for name, digest in value["artifact_hashes"].items():
        if common.digest(OUT / name) != digest:
            raise ValueError("Judge artifacts changed")
    shutil.copyfile(path, target / path.name)
    shutil.copyfile(OUT / "rubric.txt", target / "description_equivalence_rubric.txt")


def run():
    manifest = verify()
    keys = {"openai": common.credentials()["openai"]}
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(execute, b, order, keys) for b in manifest["batches"] for order in ("ab", "ba")]
        for job in as_completed(jobs):
            job.result()
            report()
    if report()["validated"] != len(manifest["batches"])*2:
        raise ValueError("Some calls stopped; retained for review, not retried")
    combine()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "combine"))
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source.configure(args.annotations)
    OUT = args.out.resolve()
    globals()[args.command]()
