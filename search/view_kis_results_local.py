"""Display Textual KIS CSV results from a local keyframe folder.

Edit CSV_PATH, KEYFRAME_ROOT, and KEYFRAME_INDEX_PATH, then run this file
directly. This version does not download anything from Kaggle.

Controls:
    Space / Right Arrow  next result
    Left Arrow           previous result
    Q / Escape           close viewer
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path


# ============================================================
# Configuration: edit these paths.
# ============================================================
csv_name = "test"
CSV_PATH = Path(
    rf"C:\linhtinh\HCMC AI\deloy\submission\{csv_name}.csv"
)
KEYFRAME_ROOT = Path(r"C:\linhtinh\HCMC AI\deloy\keyframes")
KEYFRAME_INDEX_PATH = Path(
    r"C:\linhtinh\HCMC AI\deloy\keyframes.jsonl"
)

MAX_IMAGE_WIDTH = 1400
MAX_IMAGE_HEIGHT = 850
BACKGROUND_COLOR = "#111111"
TEXT_COLOR = "#f2f2f2"
SUPPORTED_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")


KISResult = dict[str, object]
FrameKey = tuple[str, int]


def configure_console_utf8() -> None:
    """Allow Vietnamese messages in legacy Windows terminals."""

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="backslashreplace")


def load_kis_csv(path: Path) -> list[KISResult]:
    """Read headerless website-format rows: video_id,frame_id."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"KIS CSV does not exist: {path}")

    results: list[KISResult] = []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream)
            for line_number, row in enumerate(reader, start=1):
                if not row or all(not value.strip() for value in row):
                    continue
                if len(row) != 2:
                    raise ValueError(
                        f"CSV line {line_number} must contain exactly "
                        "video_id,frame_id"
                    )

                video_id = row[0].strip()
                raw_frame_id = row[1].strip()
                if not video_id:
                    raise ValueError(
                        f"CSV line {line_number} has an empty video_id"
                    )
                try:
                    frame_id = int(raw_frame_id)
                except ValueError as error:
                    raise ValueError(
                        f"CSV line {line_number} frame_id must be an integer"
                    ) from error
                if frame_id < 0:
                    raise ValueError(
                        f"CSV line {line_number} frame_id must be non-negative"
                    )

                results.append(
                    {
                        "rank": len(results) + 1,
                        "video_id": video_id,
                        "frame_id": frame_id,
                    }
                )
    except (OSError, UnicodeError, csv.Error) as error:
        raise RuntimeError(f"Cannot read KIS CSV '{path}': {error}") from error

    if not results:
        raise ValueError(f"KIS CSV contains no result rows: {path}")
    return results


def keyframe_video_prefix(video_id: str) -> str:
    """Convert website ID L21_V001 to keyframe prefix l21-v001."""

    normalized = video_id.strip()
    if normalized.casefold().endswith(".mp4"):
        normalized = normalized[:-4]
    match = re.fullmatch(r"[lL](\d+)[-_][vV](\d+)", normalized)
    if match is not None:
        return f"l{match.group(1)}-v{match.group(2)}"
    return normalized.casefold().replace("_", "-")


def result_key(result: Mapping[str, object], source: str) -> FrameKey:
    """Return the normalized lookup key shared by CSV and JSONL records."""

    video_id = result.get("video_id")
    frame_id = result.get("frame_id")
    if not isinstance(video_id, str) or not video_id.strip():
        raise ValueError(f"{source} video_id must be a non-empty string")
    if isinstance(frame_id, bool) or not isinstance(frame_id, int) or frame_id < 0:
        raise ValueError(f"{source} frame_id must be a non-negative integer")
    return keyframe_video_prefix(video_id), frame_id


def image_path_from_metadata(
    metadata: Mapping[str, object],
    keyframe_root: Path,
    line_number: int,
) -> Path | None:
    """Build an existing local image path from one JSONL metadata record."""

    video_prefix, frame_id = result_key(
        metadata, f"Keyframe JSONL line {line_number}"
    )
    shot_id = metadata.get("shot_id")
    if isinstance(shot_id, bool) or not isinstance(shot_id, int) or shot_id < 0:
        raise ValueError(
            f"Keyframe JSONL line {line_number} shot_id must be "
            "a non-negative integer"
        )

    stem = f"{video_prefix}_s{shot_id:06d}_f{frame_id:09d}"
    for suffix in SUPPORTED_IMAGE_SUFFIXES:
        candidate = keyframe_root / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def resolve_results(
    results: Sequence[Mapping[str, object]],
    keyframe_root: Path,
    keyframe_index_path: Path,
) -> list[KISResult]:
    """Resolve all requested images by streaming the local JSONL index once."""

    keyframe_root = Path(keyframe_root)
    keyframe_index_path = Path(keyframe_index_path)
    if not keyframe_root.is_dir():
        raise FileNotFoundError(
            f"Keyframe directory does not exist: {keyframe_root}"
        )
    if not keyframe_index_path.is_file():
        raise FileNotFoundError(
            f"Keyframe JSONL index does not exist: {keyframe_index_path}"
        )

    wanted = {result_key(result, "CSV result") for result in results}
    found: dict[FrameKey, Path] = {}
    try:
        with keyframe_index_path.open("r", encoding="utf-8") as stream:
            for line_number, raw_line in enumerate(stream, start=1):
                if not raw_line.strip():
                    continue
                try:
                    metadata = json.loads(raw_line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON at keyframe index line "
                        f"{line_number}: {error}"
                    ) from error
                if not isinstance(metadata, dict):
                    raise ValueError(
                        f"Keyframe JSONL line {line_number} must be an object"
                    )

                key = result_key(
                    metadata, f"Keyframe JSONL line {line_number}"
                )
                if key not in wanted or key in found:
                    continue
                image_path = image_path_from_metadata(
                    metadata, keyframe_root, line_number
                )
                if image_path is not None:
                    found[key] = image_path
                    if len(found) == len(wanted):
                        break
    except (OSError, UnicodeError) as error:
        raise RuntimeError(
            f"Cannot read keyframe JSONL index "
            f"'{keyframe_index_path}': {error}"
        ) from error

    resolved: list[KISResult] = []
    for result in results:
        image_path = found.get(result_key(result, "CSV result"))
        if image_path is None:
            print(
                "Warning: keyframe not found for "
                f"rank={result['rank']} video={result['video_id']} "
                f"frame={result['frame_id']}"
            )
            continue
        resolved.append({**result, "image_path": image_path})

    if not resolved:
        raise FileNotFoundError(
            "None of the CSV results matched a local keyframe image"
        )
    return resolved


class KISKeyframeViewer:
    """A single-window, keyboard-controlled KIS result viewer."""

    def __init__(
        self,
        root: object,
        results: Sequence[Mapping[str, object]],
    ):
        import tkinter as tk

        self.root = root
        self.results = list(results)
        self.index = 0
        self.photo: object | None = None

        root.configure(background=BACKGROUND_COLOR)
        self.image_label = tk.Label(root, background=BACKGROUND_COLOR)
        self.image_label.pack(
            fill="both", expand=True, padx=10, pady=(10, 4)
        )

        self.info_label = tk.Label(
            root,
            background=BACKGROUND_COLOR,
            foreground=TEXT_COLOR,
            font=("Segoe UI", 13, "bold"),
        )
        self.info_label.pack(fill="x", padx=10, pady=(2, 2))

        self.help_label = tk.Label(
            root,
            text="Space / Right: next    Left: previous    Q / Esc: close",
            background=BACKGROUND_COLOR,
            foreground="#bdbdbd",
            font=("Segoe UI", 10),
        )
        self.help_label.pack(fill="x", padx=10, pady=(2, 10))

        root.bind("<space>", self.next_result)
        root.bind("<Right>", self.next_result)
        root.bind("<Left>", self.previous_result)
        root.bind("<Escape>", self.close)
        root.bind("<KeyPress-q>", self.close)
        root.bind("<KeyPress-Q>", self.close)
        root.protocol("WM_DELETE_WINDOW", root.destroy)

        self.show_current()

    def show_current(self) -> None:
        from PIL import Image, ImageOps, ImageTk

        result = self.results[self.index]
        image_path = result["image_path"]
        if not isinstance(image_path, Path):
            raise TypeError("Resolved result image_path must be a Path")

        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        screen_width = max(320, int(self.root.winfo_screenwidth()) - 80)
        screen_height = max(240, int(self.root.winfo_screenheight()) - 180)
        image.thumbnail(
            (
                min(MAX_IMAGE_WIDTH, screen_width),
                min(MAX_IMAGE_HEIGHT, screen_height),
            ),
            Image.Resampling.LANCZOS,
        )

        self.photo = ImageTk.PhotoImage(image)
        self.image_label.configure(image=self.photo)

        position = self.index + 1
        total = len(self.results)
        info = (
            f"Result {position}/{total}  |  Rank {result['rank']}  |  "
            f"{result['video_id']}  |  Frame {result['frame_id']}"
        )
        self.info_label.configure(text=info)
        self.root.title(info)

    def next_result(self, _event: object = None) -> str:
        if self.index < len(self.results) - 1:
            self.index += 1
            self.show_current()
        else:
            self.root.bell()
        return "break"

    def previous_result(self, _event: object = None) -> str:
        if self.index > 0:
            self.index -= 1
            self.show_current()
        else:
            self.root.bell()
        return "break"

    def close(self, _event: object = None) -> str:
        self.root.destroy()
        return "break"


def main() -> None:
    import tkinter as tk

    configure_console_utf8()
    results = load_kis_csv(CSV_PATH.resolve())
    resolved = resolve_results(
        results,
        KEYFRAME_ROOT.resolve(),
        KEYFRAME_INDEX_PATH.resolve(),
    )
    print(
        f"Loaded {len(results)} CSV rows; "
        f"displaying {len(resolved)} local keyframes"
    )

    root = tk.Tk()
    KISKeyframeViewer(root, resolved)
    root.after(50, root.focus_force)
    root.mainloop()


if __name__ == "__main__":
    main()
