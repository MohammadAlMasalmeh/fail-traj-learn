"""Precompute symmetric STS DistilRoBERTa scores for overlapping descriptions."""
import argparse
from importlib.metadata import version
import json
import math
from pathlib import Path

MODEL = "cross-encoder/stsb-distilroberta-base"
MODEL_REVISION = "6b71347df6e2b34246b53e06d6bce70ef67de368"


def summarize(data, scores):
    """Average cached pair scores over source frames; expose incomplete coverage."""
    manifest = scores.get('annotation_manifest_sha256')
    if manifest and manifest != data.get('annotation_manifest_sha256'):
        raise ValueError('Text scores belong to a different annotation batch')
    lookup = {}
    for row in scores['pairs']:
        score = row['score']
        if type(score) not in (int, float) or not math.isfinite(score):
            raise ValueError('Text scores must be finite numbers')
        lookup[(row['left'], row['right'])] = score

    def timeline(annotation, n):
        values, occupied = [None]*n, [False]*n
        if annotation is None:
            return values
        for segment in annotation['segments']:
            start, end = segment['start_frame'], segment['end_frame']
            if type(start) is not int or type(end) is not int or not 0 <= start < end <= n:
                raise ValueError('Invalid description interval')
            for frame in range(start, end):
                if occupied[frame]:
                    raise ValueError('Overlapping descriptions cannot define a frame score')
                occupied[frame] = True
                values[frame] = segment.get('what_happened') or None
        for interval in annotation.get('uncertain_intervals', []):
            for frame in range(max(0, interval['start_frame']), min(n, interval['end_frame'])):
                values[frame] = None
        return values

    def mean_row(total, covered, weighted):
        return dict(total_frames=total, scored_frames=covered,
                    coverage_fraction=covered/total if total else None,
                    mean_all_frames=weighted/total if total and covered == total else None,
                    mean_scored_frames=weighted/covered if covered else None)

    runs = []
    for run in data['runs']:
        rows, total, covered, weighted = [], 0, 0, 0.
        for episode in data['episodes']:
            n = episode['n_frames']
            annotations = episode['versions'].get(run['id'], {})
            left, right = [timeline(annotations.get(p), n) for p in ('openai', 'gemini')]
            count, subtotal = 0, 0.
            for a, b in zip(left, right):
                score = lookup.get((a, b)) if a and b else None
                if score is not None:
                    count += 1
                    subtotal += score
            rows.append(dict(episode=episode['id'], **mean_row(n, count, subtotal)))
            total += n
            covered += count
            weighted += subtotal
        runs.append(dict(run=run['id'], episodes=rows, overall=mean_row(total, covered, weighted)))
    return dict(model=scores['model'], annotation_manifest_sha256=manifest,
                aggregation='Mean over source frames, weighted by frames covered by each description pair',
                limitations='Text similarity only. Missing scores are not zero; an all-frame mean requires complete coverage.',
                runs=runs)


def save_summary(folder, data, scores):
    summary = summarize(data, scores)
    (folder / 'text_similarity_summary.json').write_text(
        json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    for run in summary['runs']:
        print(run['run'], [(r['episode'], r['mean_all_frames']) for r in run['episodes']],
              'overall:', run['overall']['mean_all_frames'])
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, nargs="?",
                        default=Path("outputs/taxonomy_pilot_v1/web_review"))
    parser.add_argument("--local-files-only", action="store_true",
                        help="Use cached model files without downloading")
    parser.add_argument("--summarize-only", action="store_true",
                        help="Aggregate existing pair scores without loading a model")
    args = parser.parse_args()
    data = json.loads((args.folder / "data.json").read_text(encoding="utf-8"))
    if args.summarize_only:
        scores = json.loads((args.folder / 'text_similarity.json').read_text(encoding='utf-8'))
        save_summary(args.folder, data, scores)
        return
    pairs = set()
    for episode in data["episodes"]:
        for annotations in episode["versions"].values():
            left, right = annotations.get("openai"), annotations.get("gemini")
            if not left or not right:
                continue
            for a in left["segments"]:
                for b in right["segments"]:
                    if max(a["start_frame"], b["start_frame"]) < min(a["end_frame"], b["end_frame"]):
                        if a.get("what_happened") and b.get("what_happened"):
                            pairs.add((a["what_happened"], b["what_happened"]))
    texts = sorted({text for pair in pairs for text in pair})
    import torch
    from sentence_transformers import CrossEncoder
    from huggingface_hub import try_to_load_from_cache
    torch.set_num_threads(4)
    model = CrossEncoder(MODEL, revision=MODEL_REVISION, device="cpu", local_files_only=args.local_files_only)
    cached_config = try_to_load_from_cache(MODEL, "config.json", revision=MODEL_REVISION)
    revision = Path(cached_config).parent.name if isinstance(cached_config, str) else None
    pairs = sorted(pairs)
    limit = min(model.max_seq_length or 512, model.tokenizer.model_max_length)
    if any(len(model.tokenizer.encode(a, b, add_special_tokens=True)) > limit for a,b in pairs):
        raise ValueError("A description pair exceeds the model context; refusing silent truncation")
    predictions = model.predict(pairs + [(b,a) for a,b in pairs],
                                batch_size=8, show_progress_bar=True)
    scores = (predictions[:len(pairs)] + predictions[len(pairs):]) / 2
    payload = {
        "model": MODEL,
        "annotation_manifest_sha256": data.get("annotation_manifest_sha256"),
        "revision": revision,
        "packages": {name: version(name) for name in ("sentence-transformers", "transformers", "torch")},
        "field": "what_happened",
        "metric": "semantic_textual_similarity",
        "aggregation": "mean of both input orders",
        "pairs": [{"left": a, "right": b,
                   "score": float(score)}
                  for (a, b), score in zip(pairs, scores)],
    }
    (args.folder / "text_similarity.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Scored {len(pairs)} overlapping description pairs from {len(texts)} unique texts")
    save_summary(args.folder, data, payload)


if __name__ == "__main__":
    main()
