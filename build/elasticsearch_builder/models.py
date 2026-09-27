from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FrameRecord:
    video_id: str
    shot_id: int
    frame_id: int
    ocr_text: str
    asr_text: str
    objects: tuple[tuple[str, int], ...]

    @property
    def keyframe_id(self) -> str:
        return f"{self.video_id}_s{self.shot_id:06d}_f{self.frame_id:09d}"

    def to_source(self) -> dict[str, object]:
        return {
            "video_id": self.video_id,
            "shot_id": self.shot_id,
            "frame_id": self.frame_id,
            "ocr_text": self.ocr_text,
            "asr_text": self.asr_text,
            "objects": dict(self.objects),
        }


@dataclass(frozen=True)
class BuildResult:
    index_name: str
    document_count: int
