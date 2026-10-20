from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torchaudio

from loopscape.audio import SAMPLE_RATE, audio_metrics, make_continuous_loop, read_stereo
from loopscape.manifest import records, split_for_source, write_records


ALLOWED_LICENSES = {"cc0", "cc-by-4.0", "cc-by-nc-4.0"}


def prepare(input_path: Path, output_dir: Path, duration_s: float, overlap_s: float) -> None:
    if duration_s <= 0 or overlap_s <= 0:
        raise ValueError("duration and overlap must be positive")
    prepared: list[dict] = []
    rejected: list[dict] = []
    seen_ids: set[str] = set()
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    for item in records(input_path):
        source_id = str(item.get("source_id", ""))
        required = (
            "path", "caption", "source_url", "author", "license",
            "ai_training_allowed", "permission_checked_at",
        )
        missing = [key for key in required if key not in item or item[key] in (None, "")]
        reason = None
        if not source_id or missing:
            reason = f"missing metadata: {missing or ['source_id']}"
        elif source_id in seen_ids:
            reason = "duplicate source_id"
        elif item["ai_training_allowed"] is not True:
            reason = "AI training permission not recorded as allowed"
        elif str(item["license"]).lower() not in ALLOWED_LICENSES:
            reason = "license not in the reviewed allowlist"
        if reason:
            rejected.append({"source_id": source_id, "reason": reason})
            continue
        seen_ids.add(source_id)
        source_path = Path(item["path"])
        try:
            audio = read_stereo(source_path)
            clip = make_continuous_loop(audio, duration_s, overlap_s)
            metrics = audio_metrics(clip)
            if metrics["rms"] < 0.005 or metrics["peak"] > 1.05:
                raise ValueError("clip is nearly silent or has invalid peak level")
        except (OSError, RuntimeError, ValueError) as error:
            rejected.append({"source_id": source_id, "reason": str(error)})
            continue
        filename = hashlib.sha256(source_id.encode()).hexdigest()[:20] + ".wav"
        destination = audio_dir / filename
        torchaudio.save(str(destination), clip, SAMPLE_RATE)
        prepared.append(
            {
                "source_id": source_id,
                "group_id": str(item.get("group_id", source_id)),
                "source_url": item["source_url"],
                "author": item["author"],
                "license": item["license"],
                "ai_training_allowed": True,
                "permission_checked_at": item["permission_checked_at"],
                "caption": item["caption"],
                "path": str(destination),
                "split": split_for_source(str(item.get("group_id", source_id))),
                "duration_s": duration_s,
                "overlap_s": overlap_s,
                "source_rms": metrics["rms"],
            }
        )
    write_records(output_dir / "manifest.jsonl", prepared)
    write_records(output_dir / "rejected.jsonl", rejected)
    print(f"prepared={len(prepared)} rejected={len(rejected)} output={output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare traceable environmental loops")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--overlap", type=float, default=0.5)
    args = parser.parse_args()
    prepare(args.input, args.output_dir, args.duration, args.overlap)


if __name__ == "__main__":
    main()
