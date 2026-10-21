"""Extract a reproducible CC0 soundscape subset from Zenodo using HTTP ranges."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import http.client
import io
import json
import random
import re
import threading
import time
import urllib.request
import wave
import zipfile
from datetime import date
from pathlib import Path


RECORD_URL = "https://zenodo.org/records/18911328"
ARCHIVE_URL = (
    "https://zenodo.org/api/records/18911328/files/"
    "42937__sensingtheforest__natural-soundscape-dataset-part-2.zip/content"
)
README_URL = f"{RECORD_URL}/files/README.txt?download=1"
README_MD5 = "3e17209751cb59dbf10d2cadd78f00bb"
ARCHIVE_BYTES = 20_698_813_016
PUBLISHED_ARCHIVE_MD5 = "01feafa071d8dd2e5f1a60df362c6cc3"
BLOCK_BYTES = 1024 * 1024
WORKER_STATE = threading.local()
FILENAME = re.compile(
    r"^(\d+)__sensingtheforest__(\d{4}-\d{2}-\d{2})_(\d{4})\.wav(?:\.wav)?$"
)
LICENSED_SOUND = re.compile(
    r"(?m)^\s+\* (\d+)__sensingtheforest__[^\n]*\n"
    r"\s+\* url: https://freesound.org/s/\1/\n"
    r"\s+\* license: Creative Commons 0$"
)


class RangeReader(io.RawIOBase):
    def __init__(self, url: str, size: int) -> None:
        self.url = url
        self.size = size
        self.position = 0
        self.block_start = -1
        self.block = b""
        self.fetched_bytes = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self.position + offset
        elif whence == io.SEEK_END:
            position = self.size + offset
        else:
            raise ValueError(f"invalid whence: {whence}")
        if position < 0:
            raise ValueError("negative seek position")
        self.position = position
        return position

    def _fetch_block(self, start: int) -> None:
        end = min(self.size, start + BLOCK_BYTES) - 1
        request = urllib.request.Request(
            self.url, headers={"Range": f"bytes={start}-{end}"}
        )
        for attempt in range(4):
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    if response.status != 206:
                        raise OSError(f"server ignored HTTP Range: {response.status}")
                    block = response.read()
                if len(block) != end - start + 1:
                    raise OSError("short HTTP Range response")
                self.block_start = start
                self.block = block
                self.fetched_bytes += len(block)
                return
            except (OSError, TimeoutError, http.client.IncompleteRead):
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = self.size - self.position
        size = min(size, self.size - self.position)
        output = bytearray()
        while size > 0:
            if not self.block_start <= self.position < self.block_start + len(self.block):
                self._fetch_block((self.position // BLOCK_BYTES) * BLOCK_BYTES)
            offset = self.position - self.block_start
            length = min(size, len(self.block) - offset)
            output.extend(self.block[offset:offset + length])
            self.position += length
            size -= length
        return bytes(output)


def caption_for_hour(hhmm: str) -> str:
    hour = int(hhmm[:2])
    if 5 <= hour < 11:
        period = "morning"
    elif 11 <= hour < 16:
        period = "daytime"
    elif 16 <= hour < 21:
        period = "evening"
    else:
        period = "nighttime"
    return f"A continuous {period} forest ambience in Surrey."


def licensed_sound_ids(output_dir: Path) -> set[str]:
    readme_path = output_dir / "README.txt"
    if readme_path.is_file():
        content = readme_path.read_bytes()
    else:
        with urllib.request.urlopen(README_URL, timeout=120) as response:
            content = response.read()
    if hashlib.md5(content).hexdigest() != README_MD5:
        raise ValueError("Zenodo README MD5 does not match the published value")
    readme_path.write_bytes(content)
    return set(LICENSED_SOUND.findall(content.decode("utf-8")))


def extract_clip(info: zipfile.ZipInfo, destination: Path) -> str:
    if not hasattr(WORKER_STATE, "archive"):
        reader = RangeReader(ARCHIVE_URL, ARCHIVE_BYTES)
        WORKER_STATE.archive = zipfile.ZipFile(reader)
    for attempt in range(3):
        try:
            with WORKER_STATE.archive.open(info) as member, wave.open(member, "rb") as audio:
                channels = audio.getnchannels()
                sample_width = audio.getsampwidth()
                sample_rate = audio.getframerate()
                if audio.getcomptype() != "NONE":
                    raise ValueError(f"compressed WAV: {info.filename}")
                audio.readframes(5 * sample_rate)
                frames = int(10.5 * sample_rate)
                payload = audio.readframes(frames)
                if len(payload) != frames * channels * sample_width:
                    raise ValueError(f"short WAV: {info.filename}")
            temporary = destination.with_suffix(".wav.tmp")
            with wave.open(str(temporary), "wb") as output:
                output.setnchannels(channels)
                output.setsampwidth(sample_width)
                output.setframerate(sample_rate)
                output.writeframes(payload)
            temporary.replace(destination)
            return hashlib.sha256(destination.read_bytes()).hexdigest()
        except (OSError, TimeoutError, http.client.IncompleteRead, zipfile.BadZipFile):
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def extract_subset(output_dir: Path, count: int, seed: int, workers: int) -> None:
    if count <= 0:
        raise ValueError("count must be positive")
    if workers <= 0:
        raise ValueError("workers must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    licensed_ids = licensed_sound_ids(output_dir)
    audio_dir = output_dir / "clips"
    audio_dir.mkdir(exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"
    completed: set[str] = set()
    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    record = json.loads(line)
                    if Path(record["path"]).is_file():
                        completed.add(record["source_id"])

    remote = RangeReader(ARCHIVE_URL, ARCHIVE_BYTES)
    with zipfile.ZipFile(remote) as archive:
        entries = [
            (info, match)
            for info in archive.infolist()
            if (match := FILENAME.fullmatch(Path(info.filename).name))
        ]
        if len(entries) < count:
            raise ValueError(f"archive has only {len(entries)} matching WAV files")
        selected = random.Random(seed).sample(entries, count)
        selected.sort(key=lambda entry: entry[0].filename)
        unlicensed = [match.group(1) for _, match in selected
                      if match.group(1) not in licensed_ids]
        if unlicensed:
            raise ValueError(f"missing CC0 evidence in README: {unlicensed[:5]}")
        pending = [(info, match) for info, match in selected
                   if f"zenodo-18911328-{match.group(1)}" not in completed]
        print(
            f"selected={len(selected)} eligible={len(entries)} "
            f"completed={len(selected) - len(pending)} workers={workers}",
            flush=True,
        )
        failures = []
        with (manifest_path.open("a", encoding="utf-8") as manifest,
              concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool):
            futures = {
                pool.submit(extract_clip, info, audio_dir / f"{match.group(1)}.wav"): (info, match)
                for info, match in pending
            }
            for future in concurrent.futures.as_completed(futures):
                info, match = futures[future]
                sound_id, recording_day, hhmm = match.groups()
                source_id = f"zenodo-18911328-{sound_id}"
                try:
                    digest = future.result()
                except Exception as error:
                    failures.append(source_id)
                    print(f"failed={source_id} error={error!r}", flush=True)
                    continue
                destination = audio_dir / f"{sound_id}.wav"
                record = {
                    "source_id": source_id,
                    "group_id": recording_day,
                    "source_url": RECORD_URL,
                    "original_sound_url": f"https://freesound.org/s/{sound_id}/",
                    "author": "Sensing the Forest",
                    "license": "cc0",
                    "license_evidence_url": README_URL,
                    "ai_training_allowed": True,
                    "permission_checked_at": date.today().isoformat(),
                    "path": str(destination.resolve()),
                    "caption": caption_for_hour(hhmm),
                    "recording_day": recording_day,
                    "recording_time": hhmm,
                    "source_archive_member": info.filename,
                    "source_archive_published_md5": PUBLISHED_ARCHIVE_MD5,
                    "clip_start_s": 5.0,
                    "clip_duration_s": 10.5,
                    "clip_sha256": digest,
                }
                manifest.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                manifest.flush()
                completed.add(source_id)
                if len(completed) % 10 == 0 or len(completed) == len(selected):
                    print(
                        f"saved={len(completed)}/{len(selected)}",
                        flush=True,
                    )
        if failures:
            raise RuntimeError(f"{len(failures)} clips failed; rerun to resume")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a CC0 forest soundscape subset")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/sensing_forest"))
    parser.add_argument("--count", type=int, default=240)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    extract_subset(args.output_dir, args.count, args.seed, args.workers)


if __name__ == "__main__":
    main()
