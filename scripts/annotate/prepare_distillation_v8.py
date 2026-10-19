"""Select fresh object-corpus episodes and export observable-only V8 source packs.

Run with the LeRobot Python environment on the machine holding the raw recordings.
Selection uses task, collection condition and recorded outcome only for sampling;
none of that private sampling metadata is added to the model's observable input.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE / "bench"))


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def identities(value):
    if isinstance(value, dict):
        if "dataset" in value and "episode_index" in value:
            yield value["dataset"], int(value["episode_index"])
        for child in value.values():
            yield from identities(child)
    elif isinstance(value, list):
        for child in value:
            yield from identities(child)


def balanced_take(rows, count, rng):
    """Equal outcome allocation; spread each outcome over tasks and datasets."""
    selected = []
    for outcome, quota in ((False, count // 2), (True, count - count // 2)):
        groups = defaultdict(lambda: defaultdict(list))
        for row in rows:
            if row["recorded_success"] == outcome:
                groups[row["task"]][row["dataset"]].append(row)
        if sum(len(v) for group in groups.values() for v in group.values()) < quota:
            raise ValueError(f"Insufficient fresh episodes for outcome={outcome}: need {quota}")
        for group in groups.values():
            for values in group.values():
                rng.shuffle(values)
        task_counts, dataset_counts = Counter(), Counter()
        for _ in range(quota):
            tasks = [t for t, group in groups.items() if any(group.values())]
            rng.shuffle(tasks)
            task = min(tasks, key=lambda t: task_counts[t])
            datasets = [d for d, values in groups[task].items() if values]
            rng.shuffle(datasets)
            dataset = min(datasets, key=lambda d: dataset_counts[d])
            selected.append(groups[task][dataset].pop())
            task_counts[task] += 1
            dataset_counts[dataset] += 1
    rng.shuffle(selected)
    return selected


def select(data, out, count, seed):
    out.mkdir(parents=True, exist_ok=True)
    selection_path = out / "selection.json"
    if selection_path.exists():
        selection = load(selection_path)
        if selection["count"] != count or selection["seed"] != seed or selection["data"] != str(data):
            raise ValueError("Existing selection has different arguments")
        return selection
    excluded, exclusion_hashes = set(), {}
    patterns = ["**/*private_manifest.json", "**/oracle_audit_r6/manifest.json"]
    paths = set()
    for pattern in patterns:
        paths.update((ROOT / "outputs").glob(pattern))
    paths.update(p for p in (data.parent / "bench").glob("episodes*.json"))
    for path in sorted(paths):
        if path.is_relative_to(out):
            continue
        excluded.update(identities(load(path)))
        exclusion_hashes[str(path)] = digest(path)
    corrections_path = ROOT / "scripts/analysis/exclusions.json"
    corrections = load(corrections_path) if corrections_path.exists() else {}
    rows = []
    inventory_hashes = {}
    for folder in sorted(data.glob("full_*")):
        index = folder / "episodes.jsonl"
        if not index.exists() or not (folder / "meta/info.json").exists():
            continue
        inventory_hashes[str(index)] = digest(index)
        for text in index.read_text(encoding="utf-8").splitlines():
            row = json.loads(text)
            key = folder.name, int(row["episode_index"])
            if row["suite"] != "libero_object" or key in excluded:
                continue
            outcome = bool(row["success"])
            dropped = False
            for entry in corrections.get(folder.name, []):
                if entry["episode_index"] == key[1]:
                    if entry["action"] == "drop":
                        dropped = True
                    elif entry["action"] in ("relabel_success", "relabel_failure"):
                        outcome = entry["action"] == "relabel_success"
            if dropped:
                continue
            rows.append(dict(dataset=folder.name, episode_index=key[1],
                             recorded_success=outcome, task=row["task_language"],
                             n_frames=int(row["length"])))
    rng = random.Random(seed)
    remaining = list(rows)
    splits = {}
    # Reserve references first; they are neither exported nor sent for labeling.
    for split, size in (("test", 50), ("development", 50), ("train", count)):
        chosen = balanced_take(remaining, size, rng)
        keys = {(r["dataset"], r["episode_index"]) for r in chosen}
        remaining = [r for r in remaining if (r["dataset"], r["episode_index"]) not in keys]
        prefix = {"train": "D", "development": "V", "test": "T"}[split]
        splits[split] = [dict(row, id=f"{prefix}{i:04d}") for i, row in enumerate(chosen, 1)]
    selection = dict(version="sol_v8_distillation_500_v1", data=str(data), count=count,
                     seed=seed, suite="libero_object", excluded_identities=len(excluded),
                     exclusion_manifests_sha256=exclusion_hashes,
                     inventory_sha256=inventory_hashes,
                     corrections_sha256=digest(corrections_path), splits=splits,
                     remaining_unlabeled=len(remaining),
                     limitation="Outcome-balanced, task/collection-spread sample, not a natural-frequency test set. Recorded outcomes are sampling metadata, not local failure truth. Holdouts are episode-disjoint, not task- or policy-disjoint.")
    save(selection_path, selection)
    print(json.dumps(dict(selection_sha256=digest(selection_path), eligible=len(rows),
                          excluded=len(excluded), remaining=len(remaining),
                          splits={k:len(v) for k,v in splits.items()},
                          training_tasks=dict(Counter(r["task"] for r in splits["train"])),
                          training_datasets=len({r["dataset"] for r in splits["train"]}))), flush=True)
    return selection


def export(data, out, selection, worker_index=0, workers=1):
    import numpy as np
    import common
    from review_oracle import decode_episode
    common.DATA = data
    packs = out / "source"
    packs.mkdir(exist_ok=True)
    base_audit = out / "source_export.json"
    audit_path = base_audit if workers == 1 else out / f"source_export_worker_{worker_index}.json"
    audit = load(base_audit) if base_audit.exists() else {}
    if audit_path != base_audit and audit_path.exists():
        audit.update(load(audit_path))
    dataset = None
    previous_name = None
    for number, row in enumerate(sorted(selection["splits"]["train"], key=lambda r:(r["dataset"],r["episode_index"])), 1):
        if (number-1) % workers != worker_index:
            continue
        folder = packs / row["id"]
        folder.mkdir(exist_ok=True)
        if row["id"] in audit:
            for name, sha in audit[row["id"]]["files_sha256"].items():
                if digest(folder / name) != sha:
                    raise ValueError(f"Export changed: {row['id']}/{name}")
            continue
        if (out / "run/manifest.json").exists():
            raise ValueError("Cannot regenerate source after annotation inputs were frozen")
        if previous_name != row["dataset"]:
            dataset = common.open_dataset(row["dataset"])
            previous_name = row["dataset"]
        episode = common.Episode(dataset, row["dataset"], row["episode_index"])
        if episode.n != row["n_frames"] or episode.task != row["task"]:
            raise ValueError("Episode metadata differs from selection")
        agent, wrist, mode = decode_episode(dataset, episode)
        for camera in (agent, wrist):
            if camera.shape != (episode.n, 256, 256, 3) or camera.dtype != np.uint8:
                raise ValueError("Unexpected decoded camera shape or dtype")
        state, action = episode.col("observation.state"), episode.col("action")
        if not np.isfinite(state).all() or not np.isfinite(action).all():
            raise ValueError("Nonfinite robot telemetry")
        telemetry = [dict(frame=t, eef_xyz_m=state[t,:3].round(5).tolist(),
                          finger_qpos_m=state[t,6:8].round(5).tolist(),
                          gripper_command="close" if action[t,6]>0 else "open") for t in range(episode.n)]
        np.savez_compressed(folder / "cameras.npz", agent=agent, wrist=wrist)
        save(folder / "observable.json", dict(id=row["id"], task=episode.task,
             n_frames=episode.n, fps=episode.fps, telemetry=telemetry,
             frame_bounds="zero-based, end-exclusive",
             notes="Finger positions and command are not proof of contact or possession."))
        audit[row["id"]] = dict(dataset=row["dataset"], episode_index=row["episode_index"],
            decoder=mode, selection_sha256=digest(out / "selection.json"),
            sidecar_sha256=digest(episode.root / "sidecar" / f"episode_{episode.ep:06d}.json"),
            files_sha256={name:digest(folder/name) for name in ("observable.json","cameras.npz")})
        save(audit_path, audit)
        print(f"{number}/{len(selection['splits']['train'])} {row['id']}: {episode.n} frames exported ({mode})", flush=True)
    if workers == 1 and len(audit) != len(selection["splits"]["train"]):
        raise ValueError("Export count differs from selected training count")
    print(f"Export worker {worker_index} complete", flush=True)


def export_parallel(data, out, selection, workers):
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=workers) as pool:
        tasks = [pool.submit(export, data, out, selection, i, workers) for i in range(workers)]
        for task in tasks:
            task.result()
    audit = {}
    for i in range(workers):
        path = out / f"source_export_worker_{i}.json"
        if path.exists():
            for eid, row in load(path).items():
                if eid in audit and audit[eid] != row:
                    raise ValueError("Conflicting worker exports")
                audit[eid] = row
    expected = {r["id"] for r in selection["splits"]["train"]}
    if set(audit) != expected:
        raise ValueError("Incomplete parallel export")
    save(out / "source_export.json", audit)
    print(f"Export complete: {len(audit)} distinct episodes", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("select", "export"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    selected = select(args.data.resolve(), args.out.resolve(), args.count, args.seed)
    if args.command == "export":
        if not 1 <= args.workers <= 4:
            parser.error("workers must be between one and four")
        if args.workers == 1:
            export(args.data.resolve(), args.out.resolve(), selected)
        else:
            export_parallel(args.data.resolve(), args.out.resolve(), selected, args.workers)
