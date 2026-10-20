from __future__ import annotations

import argparse
import json
import statistics
import tempfile
from pathlib import Path

import numpy as np
import torchaudio

from loopscape.audio import SAMPLE_RATE, audio_metrics, read_stereo
from loopscape.manifest import records


METRIC_NAMES = (
    "seam_jump",
    "seam_jump_ratio",
    "seam_spectral_flux_ratio",
    "edge_to_center_rms",
    "rms",
    "peak",
)


def select_wav_directory(paths: list[Path], parent: Path) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    for index, source in enumerate(paths):
        if not source.is_file():
            raise FileNotFoundError(source)
        (parent / f"{index:06d}.wav").symlink_to(source.resolve())
    return parent


def quality_metrics(samples: list[dict], generated_dir: Path, reference_paths: list[Path]) -> dict:
    try:
        from frechet_audio_distance import CLAPScore, FrechetAudioDistance
    except ImportError as error:
        raise RuntimeError("Install the optional quality dependencies: pip install -e '.[quality]'") from error
    clap = CLAPScore(submodel_name="630k-audioset", verbose=False, audio_load_worker=2)
    captions = [str(item["caption"]) for item in samples]
    audio_arrays = []
    for item in samples:
        audio = read_stereo(generated_dir / item["file"]).mean(dim=0)
        resampled = torchaudio.functional.resample(audio, SAMPLE_RATE, 48_000)
        audio_arrays.append(resampled.numpy())
    text_vectors = clap.get_text_embeddings(captions)
    audio_vectors = clap.get_audio_embeddings(audio_arrays, 48_000)
    if text_vectors.shape != audio_vectors.shape:
        raise ValueError("CLAP embeddings have incompatible shapes")
    numerator = np.sum(text_vectors * audio_vectors, axis=1)
    denominator = np.linalg.norm(text_vectors, axis=1) * np.linalg.norm(audio_vectors, axis=1)
    clap_scores = numerator / np.maximum(denominator, 1e-8)

    with tempfile.TemporaryDirectory(prefix="loopscape-fad-") as temporary:
        root = Path(temporary)
        reference_dir = select_wav_directory(reference_paths, root / "reference")
        candidate_dir = select_wav_directory(
            [generated_dir / item["file"] for item in samples], root / "generated"
        )
        fad = FrechetAudioDistance(
            model_name="vggish",
            sample_rate=16_000,
            use_pca=False,
            use_activation=False,
            verbose=False,
            audio_load_worker=2,
        ).score(str(reference_dir), str(candidate_dir), dtype="float32")
    if not np.isfinite(fad) or fad < 0:
        raise RuntimeError(f"FAD did not return a valid score: {fad}")
    return {"clap_mean": float(np.mean(clap_scores)), "clap_per_sample": clap_scores.tolist(), "fad_vggish": float(fad)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generated loops and optional CLAP/FAD")
    parser.add_argument("--generated-dir", type=Path, required=True)
    parser.add_argument("--compare-dir", type=Path)
    parser.add_argument("--reference-manifest", type=Path)
    parser.add_argument("--quality", action="store_true")
    args = parser.parse_args()
    metadata = json.loads((args.generated_dir / "metadata.json").read_text(encoding="utf-8"))
    samples = metadata["samples"]
    if not samples:
        raise ValueError("No generated samples found")
    per_sample = []
    for item in samples:
        candidate = args.generated_dir / item["file"]
        current = audio_metrics(read_stereo(candidate))
        entry = {"id": item["id"], "file": item["file"], **current}
        if args.compare_dir:
            comparison = audio_metrics(read_stereo(args.compare_dir / item["file"]))
            entry["delta_vs_compare"] = {
                name: current[name] - comparison[name] for name in METRIC_NAMES
            }
        per_sample.append(entry)
    result = {
        "count": len(per_sample),
        "mean": {
            name: statistics.mean(item[name] for item in per_sample) for name in METRIC_NAMES
        },
        "per_sample": per_sample,
    }
    if args.quality:
        if not args.reference_manifest:
            raise ValueError("--quality requires --reference-manifest")
        references = [
            Path(item["path"])
            for item in records(args.reference_manifest)
            if item.get("split") == "test"
        ]
        if len(references) < 2:
            raise ValueError("FAD needs at least two reference test clips")
        result["quality"] = quality_metrics(samples, args.generated_dir, references)
    output = args.generated_dir / "evaluation.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"count": result["count"], "mean": result["mean"], "quality": result.get("quality")}, indent=2))


if __name__ == "__main__":
    main()
