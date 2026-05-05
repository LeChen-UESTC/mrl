from __future__ import annotations

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qwen_omni_retrieval.data.media_paths import resolve_media_paths


def test_plain_video_id_gets_mp4_suffix() -> None:
    media = resolve_media_paths("video0", "/videos")
    assert media.video_path == "/videos/video0.mp4"


def test_dotted_vast_video_id_gets_mp4_suffix() -> None:
    media = resolve_media_paths("TYBUpSwGryk.28", "/videos")
    assert media.video_path == "/videos/TYBUpSwGryk.28.mp4"


def test_existing_video_extension_is_preserved() -> None:
    media = resolve_media_paths("video0.mp4", "/videos")
    assert media.video_path == "/videos/video0.mp4"


def test_dotted_video_id_audio_path_keeps_full_id() -> None:
    with TemporaryDirectory() as tmpdir:
        audio_dir = Path(tmpdir)
        (audio_dir / "TYBUpSwGryk.28.mp3").touch()
        media = resolve_media_paths("TYBUpSwGryk.28", "/videos", audio_dir=audio_dir)
        assert media.audio_path == str(audio_dir / "TYBUpSwGryk.28.mp3")


def test_video_extension_audio_path_uses_stem() -> None:
    with TemporaryDirectory() as tmpdir:
        audio_dir = Path(tmpdir)
        (audio_dir / "video0.mp3").touch()
        media = resolve_media_paths("video0.mp4", "/videos", audio_dir=audio_dir)
        assert media.audio_path == str(audio_dir / "video0.mp3")


if __name__ == "__main__":
    test_plain_video_id_gets_mp4_suffix()
    test_dotted_vast_video_id_gets_mp4_suffix()
    test_existing_video_extension_is_preserved()
    test_dotted_video_id_audio_path_keeps_full_id()
    test_video_extension_audio_path_uses_stem()
    print("ok")
