from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

VIDEO_EXTENSIONS = {".mp4", ".m4v", ".mov", ".mkv", ".avi", ".webm", ".flv"}


@dataclass(frozen=True)
class MediaPaths:
    video_path: str
    audio_path: str | None
    use_audio_in_video: bool


def _with_mp4(video_dir: str | Path, video_id: str) -> Path:
    video_id_path = Path(video_id)
    if video_id_path.suffix.lower() in VIDEO_EXTENSIONS:
        return Path(video_dir) / video_id_path.name
    return Path(video_dir) / f"{video_id}.mp4"


def _media_stem(video_id: str) -> str:
    video_id_path = Path(video_id)
    if video_id_path.suffix.lower() in VIDEO_EXTENSIONS:
        return video_id_path.stem
    return video_id_path.name


def _resolve_audio(audio_dir: str | Path | None, video_id: str) -> str | None:
    if not audio_dir:
        return None
    base = Path(audio_dir)
    stem = _media_stem(video_id)
    for suffix in (".mp3", ".wav"):
        candidate = base / f"{stem}{suffix}"
        if candidate.exists():
            return str(candidate)
    return None


def resolve_media_paths(
    video_id: str,
    video_dir: str | Path,
    audio_dir: str | Path | None = None,
    use_audio_in_video: bool = False,
    audio_from_video_if_missing: bool = True,
) -> MediaPaths:
    video_path = _with_mp4(video_dir, video_id)
    audio_path = _resolve_audio(audio_dir, video_id)
    if audio_path is None and use_audio_in_video and audio_from_video_if_missing:
        audio_path = str(video_path)
    return MediaPaths(
        video_path=str(video_path),
        audio_path=audio_path,
        use_audio_in_video=use_audio_in_video,
    )
