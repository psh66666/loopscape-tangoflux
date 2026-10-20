from __future__ import annotations

import argparse
import csv
import json
from datetime import date
from pathlib import Path

from loopscape.manifest import write_records
from loopscape.prepare import ALLOWED_LICENSES


DEFAULT_KEYWORDS = (
    "rain", "wind", "forest", "river", "stream", "waterfall", "ocean",
    "sea", "waves", "creek", "brook", "thunderstorm", "ambience",
    "ambient", "nature", "fireplace", "crickets", "insects", "birds",
    "traffic", "engine hum", "air conditioner", "ventilation",
)
REVIEW_COLUMNS = (
    "source_id", "source_url", "author", "caption", "duration_s",
    "audio_path", "license", "ai_training_allowed", "permission_checked_at",
    "reviewer_notes",
)
AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".ogg", ".aif", ".aiff"}


def audio_index(root: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in AUDIO_SUFFIXES:
            continue
        stem = path.stem
        identifier = stem.split("__", 1)[0]
        if identifier.isdecimal():
            index.setdefault(identifier, []).append(path.resolve())
    return index


def reviewed_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        missing = set(REVIEW_COLUMNS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Review CSV is missing columns: {sorted(missing)}")
        result: dict[str, dict[str, str]] = {}
        for row in reader:
            source_id = (row.get("source_id") or "").strip()
            if not source_id or source_id in result:
                raise ValueError(f"Empty or duplicate source_id in {path}: {source_id!r}")
            result[source_id] = row
    return result


def curate(
    metadata: Path,
    audio_root: Path,
    output_dir: Path,
    review_csv: Path | None,
    keywords: tuple[str, ...],
    min_duration: float,
) -> None:
    if min_duration <= 0:
        raise ValueError("min_duration must be positive")
    template_path = output_dir / "review_template.csv"
    if review_csv and review_csv.resolve() == template_path.resolve():
        raise ValueError("Copy review_template.csv to a separate review.csv before filling it")
    with metadata.open(encoding="utf-8") as stream:
        catalog = json.load(stream)
    if not isinstance(catalog, dict) or not isinstance(catalog.get("data"), list):
        raise ValueError("Expected WavCaps Freesound JSON with a data array")
    found_audio = audio_index(audio_root)
    reviews = reviewed_rows(review_csv) if review_csv else {}
    candidates: list[dict[str, str]] = []
    approved: list[dict] = []
    rejected: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for item in catalog["data"]:
        if not isinstance(item, dict):
            continue
        try:
            duration = float(item.get("duration", 0))
        except (TypeError, ValueError):
            continue
        if duration < min_duration:
            continue
        searchable = " ".join(
            str(value) for value in (
                item.get("caption", ""), item.get("description", ""),
                " ".join(map(str, item.get("tags") or [])),
            )
        ).casefold()
        if keywords and not any(word.casefold() in searchable for word in keywords):
            continue
        identifier = str(item.get("id", "")).strip()
        if not identifier.isdecimal():
            continue
        source_id = f"freesound-{identifier}"
        if source_id in seen_ids:
            rejected.append({"source_id": source_id, "reason": "duplicate metadata ID"})
            continue
        seen_ids.add(source_id)
        href = str(item.get("href", ""))
        source_url = f"https://freesound.org{href}" if href.startswith("/people/") else f"https://freesound.org/s/{identifier}/"
        paths = found_audio.get(identifier, [])
        candidate = {
            "source_id": source_id,
            "source_url": source_url,
            "author": str(item.get("author", "")).strip(),
            "caption": str(item.get("caption", "")).strip(),
            "duration_s": str(duration),
            "audio_path": str(paths[0]) if len(paths) == 1 else "",
            "license": "",
            "ai_training_allowed": "",
            "permission_checked_at": "",
            "reviewer_notes": "",
        }
        candidates.append(candidate)
        review = reviews.get(source_id)
        if review is None:
            continue
        reason = None
        if review.get("source_url", "").strip() != source_url:
            reason = "source URL differs from current metadata"
        elif review.get("author", "").strip() != candidate["author"]:
            reason = "author differs from current metadata"
        elif review.get("caption", "").strip() != candidate["caption"]:
            reason = "caption differs from current metadata"
        elif review.get("license", "").strip().lower() not in ALLOWED_LICENSES:
            reason = "license missing or outside project allowlist"
        elif review.get("ai_training_allowed", "").strip().lower() != "true":
            reason = "AI training permission has not been affirmed"
        elif not review.get("permission_checked_at", "").strip():
            reason = "permission check date missing"
        else:
            try:
                date.fromisoformat(review["permission_checked_at"].strip())
            except ValueError:
                reason = "permission check date is not ISO format"
        reviewed_path = Path(review.get("audio_path", "").strip()) if review.get("audio_path", "").strip() else None
        if reason is None and (len(paths) != 1 and reviewed_path is None):
            reason = "audio missing or ambiguous; set audio_path in review CSV"
        selected_path = reviewed_path or (paths[0] if len(paths) == 1 else None)
        if reason is None and (selected_path is None or not selected_path.is_file()):
            reason = "reviewed audio_path does not exist"
        if reason is None and not candidate["author"]:
            reason = "author missing from metadata"
        if reason is None and not candidate["caption"]:
            reason = "caption missing from metadata"
        if reason:
            rejected.append({"source_id": source_id, "reason": reason})
            continue
        approved.append({
            "source_id": source_id,
            "group_id": candidate["author"],
            "source_url": source_url,
            "author": candidate["author"],
            "license": review["license"].strip().lower(),
            "ai_training_allowed": True,
            "permission_checked_at": review["permission_checked_at"].strip(),
            "path": str(selected_path.resolve()),
            "caption": candidate["caption"],
        })
    unknown_reviews = set(reviews) - seen_ids
    for source_id in sorted(unknown_reviews):
        rejected.append({"source_id": source_id, "reason": "reviewed ID absent from selected metadata"})
    output_dir.mkdir(parents=True, exist_ok=True)
    with template_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        writer.writerows(candidates)
    write_records(output_dir / "approved_manifest.jsonl", approved)
    write_records(output_dir / "review_rejected.jsonl", rejected)
    print(f"candidates={len(candidates)} approved={len(approved)} rejected_reviews={len(rejected)} output={output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Curate WavCaps Freesound candidates for permission review")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/curated"))
    parser.add_argument("--review-csv", type=Path)
    parser.add_argument("--keywords", nargs="*", default=list(DEFAULT_KEYWORDS))
    parser.add_argument("--min-duration", type=float, default=10.5)
    args = parser.parse_args()
    curate(args.metadata, args.audio_root, args.output_dir, args.review_csv, tuple(args.keywords), args.min_duration)


if __name__ == "__main__":
    main()
