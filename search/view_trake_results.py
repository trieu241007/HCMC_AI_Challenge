"""Display TRAKE CSV sequences as keyboard-navigable keyframe contact sheets.

Edit CSV_PATH and the keyframe paths below, then run this file directly.
There is no CLI.

Controls:
    Space / Right Arrow  next result
    Left Arrow           previous result
    Home / End           first / last result
    Q / Escape           close viewer
"""

from __future__ import annotations

import csv
import json
import math
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path


# ========================= Configuration =========================
name_output = 'query-p1-13-kis'
CSV_PATH = Path(f"C:\linhtinh\HCMC AI\deloy\submission\{name_output}.csv")
KEYFRAME_ROOT = Path(r"C:\linhtinh\HCMC AI\deloy\keyframes")
KEYFRAME_INDEX_PATH = Path(r"C:\linhtinh\HCMC AI\deloy\keyframes.jsonl")

MAX_COLUMNS = 4
KIS_RANKS_PER_VIDEO = 10
TILE_WIDTH = 360
TILE_HEIGHT = 240
LABEL_HEIGHT = 34
TILE_GAP = 10
SHEET_PADDING = 10
MAX_DISPLAY_WIDTH = 1800
MAX_DISPLAY_HEIGHT = 920
START_MAXIMIZED = True

BACKGROUND_COLOR = "#111111"
TILE_COLOR = (17, 17, 17)
TEXT_COLOR = "#f2f2f2"
MUTED_TEXT_COLOR = "#bdbdbd"
EVENT_COLORS = ("#37a2ff", "#62d26f", "#ffb347", "#d98cff", "#ff6b6b", "#63d5da")
SUPPORTED_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")

TRAKE_RESULT = dict[str, object]
FRAME_KEY = tuple[str, int]


def configure_console_utf8() -> None:
    """Allow Vietnamese paths and messages in legacy Windows terminals."""

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="backslashreplace")


def load_trake_csv(path: Path) -> list[TRAKE_RESULT]:
    """Read headerless rows: video_id,frame_E1,...,frame_EN."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"TRAKE CSV does not exist: {path}")

    results: list[TRAKE_RESULT] = []
    event_count: int | None = None
    previous_video: str | None = None
    video_rank = 0
    kis_rank = 0
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            for line_number, row in enumerate(csv.reader(stream), start=1):
                if not row or all(not value.strip() for value in row):
                    continue
                if len(row) < 3:
                    raise ValueError(
                        f"CSV line {line_number} must contain video_id and at least two event frames"
                    )
                current_count = len(row) - 1
                if event_count is None:
                    event_count = current_count
                elif current_count != event_count:
                    raise ValueError("Every CSV row must contain the same number of event frames")

                video_id = row[0].strip()
                if not video_id:
                    raise ValueError(f"CSV line {line_number} has an empty video_id")
                frame_ids: list[int] = []
                for event_number, value in enumerate(row[1:], start=1):
                    try:
                        frame_id = int(value.strip())
                    except ValueError as error:
                        raise ValueError(
                            f"CSV line {line_number} E{event_number} frame must be an integer"
                        ) from error
                    if frame_id < 0:
                        raise ValueError(
                            f"CSV line {line_number} E{event_number} frame must be non-negative"
                        )
                    frame_ids.append(frame_id)
                video_key = keyframe_video_prefix(video_id)
                if video_key != previous_video:
                    video_rank += 1
                    kis_rank = 1
                    previous_video = video_key
                else:
                    kis_rank += 1
                results.append({
                    "rank": len(results) + 1,
                    "video_rank": video_rank,
                    "kis_rank": kis_rank,
                    "video_id": video_id,
                    "frame_ids": frame_ids,
                })
    except (OSError, UnicodeError, csv.Error) as error:
        raise RuntimeError(f"Cannot read TRAKE CSV '{path}': {error}") from error

    if not results:
        raise ValueError(f"TRAKE CSV contains no result rows: {path}")
    return results


def keyframe_video_prefix(video_id: str) -> str:
    """Convert a website ID such as L24_V037 to keyframe prefix l24-v037."""

    normalized = video_id.strip()
    if normalized.casefold().endswith(".mp4"):
        normalized = normalized[:-4]
    match = re.fullmatch(r"[lL](\d+)[-_][vV](\d+)", normalized)
    if match is not None:
        return f"l{match.group(1)}-v{match.group(2)}"
    return normalized.casefold().replace("_", "-")


def result_keys(result: Mapping[str, object], source: str) -> list[FRAME_KEY]:
    """Return normalized (video, frame) keys in event order."""

    video_id = result.get("video_id")
    frame_ids = result.get("frame_ids")
    if not isinstance(video_id, str) or not video_id.strip():
        raise ValueError(f"{source} video_id must be a non-empty string")
    if not isinstance(frame_ids, list) or len(frame_ids) < 2:
        raise ValueError(f"{source} must contain at least two frame IDs")
    keys: list[FRAME_KEY] = []
    for frame_id in frame_ids:
        if isinstance(frame_id, bool) or not isinstance(frame_id, int) or frame_id < 0:
            raise ValueError(f"{source} frame IDs must be non-negative integers")
        keys.append((keyframe_video_prefix(video_id), frame_id))
    return keys


def metadata_key(metadata: Mapping[str, object], line_number: int) -> FRAME_KEY:
    video_id = metadata.get("video_id")
    frame_id = metadata.get("frame_id")
    if not isinstance(video_id, str) or not video_id.strip():
        raise ValueError(f"Keyframe JSONL line {line_number} has invalid video_id")
    if isinstance(frame_id, bool) or not isinstance(frame_id, int) or frame_id < 0:
        raise ValueError(f"Keyframe JSONL line {line_number} has invalid frame_id")
    return keyframe_video_prefix(video_id), frame_id


def image_path_from_metadata(
    metadata: Mapping[str, object], keyframe_root: Path, line_number: int
) -> Path | None:
    """Build the exact image path from a keyframe metadata record."""

    video_prefix, frame_id = metadata_key(metadata, line_number)
    shot_id = metadata.get("shot_id")
    if isinstance(shot_id, bool) or not isinstance(shot_id, int) or shot_id < 0:
        raise ValueError(f"Keyframe JSONL line {line_number} has invalid shot_id")
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
) -> list[TRAKE_RESULT]:
    """Resolve all requested keyframes by streaming the JSONL index once."""

    keyframe_root = Path(keyframe_root)
    keyframe_index_path = Path(keyframe_index_path)
    if not keyframe_root.is_dir():
        raise FileNotFoundError(f"Keyframe directory does not exist: {keyframe_root}")
    if not keyframe_index_path.is_file():
        raise FileNotFoundError(f"Keyframe JSONL index does not exist: {keyframe_index_path}")

    wanted = {
        key
        for result in results
        for key in result_keys(result, "CSV result")
    }
    found: dict[FRAME_KEY, Path] = {}
    try:
        with keyframe_index_path.open("r", encoding="utf-8") as stream:
            for line_number, raw_line in enumerate(stream, start=1):
                if not raw_line.strip():
                    continue
                try:
                    metadata = json.loads(raw_line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON at keyframe index line {line_number}: {error}"
                    ) from error
                if not isinstance(metadata, dict):
                    raise ValueError(f"Keyframe JSONL line {line_number} must be an object")
                key = metadata_key(metadata, line_number)
                if key not in wanted or key in found:
                    continue
                image_path = image_path_from_metadata(metadata, keyframe_root, line_number)
                if image_path is not None:
                    found[key] = image_path
                    if len(found) == len(wanted):
                        break
    except (OSError, UnicodeError) as error:
        raise RuntimeError(f"Cannot read keyframe index '{keyframe_index_path}': {error}") from error

    resolved: list[TRAKE_RESULT] = []
    for result in results:
        keys = result_keys(result, "CSV result")
        missing = [key for key in keys if key not in found]
        if missing:
            for video, frame_id in missing:
                print(
                    f"Warning: skipping rank {result['rank']}; missing image {video} frame {frame_id}"
                )
            continue
        resolved.append({**result, "image_paths": [found[key] for key in keys]})
    if not resolved:
        raise FileNotFoundError("No complete TRAKE result has all event keyframe images")
    return resolved


def compose_contact_sheet(
    result: Mapping[str, object],
    *,
    max_columns: int = MAX_COLUMNS,
    tile_width: int = TILE_WIDTH,
    tile_height: int = TILE_HEIGHT,
    label_height: int = LABEL_HEIGHT,
    gap: int = TILE_GAP,
    padding: int = SHEET_PADDING,
):
    """Create one RGB contact sheet with event labels in temporal order."""

    from PIL import Image, ImageDraw, ImageOps

    frame_ids = result.get("frame_ids")
    image_paths = result.get("image_paths")
    if not isinstance(frame_ids, list) or not isinstance(image_paths, list):
        raise ValueError("Result must contain frame_ids and image_paths lists")
    if not frame_ids or len(frame_ids) != len(image_paths):
        raise ValueError("frame_ids and image_paths must have equal non-zero length")
    dimensions = (max_columns, tile_width, tile_height, label_height)
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in dimensions):
        raise ValueError("Contact-sheet dimensions must be positive integers")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in (gap, padding)):
        raise ValueError("Contact-sheet gap and padding must be non-negative integers")

    columns = min(max_columns, len(image_paths))
    rows = math.ceil(len(image_paths) / columns)
    cell_height = tile_height + label_height
    width = 2 * padding + columns * tile_width + (columns - 1) * gap
    height = 2 * padding + rows * cell_height + (rows - 1) * gap
    sheet = Image.new("RGB", (width, height), TILE_COLOR)
    draw = ImageDraw.Draw(sheet)

    for index, (raw_path, frame_id) in enumerate(zip(image_paths, frame_ids)):
        image_path = Path(raw_path)
        column = index % columns
        row = index // columns
        x = padding + column * (tile_width + gap)
        y = padding + row * (cell_height + gap)
        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        image.thumbnail((tile_width, tile_height), Image.Resampling.LANCZOS)
        image_x = x + (tile_width - image.width) // 2
        image_y = y + (tile_height - image.height) // 2
        sheet.paste(image, (image_x, image_y))
        color = EVENT_COLORS[index % len(EVENT_COLORS)]
        draw.rectangle((x, y, x + tile_width - 1, y + tile_height - 1), outline=color, width=3)
        draw.text((x + 6, y + tile_height + 8), f"E{index + 1}  |  Frame {frame_id}", fill=color)
    return sheet


class TRAKEKeyframeViewer:
    """A single-window viewer for complete ordered TRAKE sequences."""

    def __init__(self, root: object, results: Sequence[Mapping[str, object]]):
        import tkinter as tk

        self.root = root
        self.results = list(results)
        self.index = 0
        self.photo: object | None = None

        root.configure(background=BACKGROUND_COLOR)
        self.image_label = tk.Label(root, background=BACKGROUND_COLOR)
        self.image_label.pack(fill="both", expand=True, padx=10, pady=(10, 4))
        self.info_label = tk.Label(
            root,
            background=BACKGROUND_COLOR,
            foreground=TEXT_COLOR,
            font=("Segoe UI", 12, "bold"),
        )
        self.info_label.pack(fill="x", padx=10, pady=(2, 2))
        self.help_label = tk.Label(
            root,
            text="Space / Right: next    Left: previous    Home / End: first / last    Q / Esc: close",
            background=BACKGROUND_COLOR,
            foreground=MUTED_TEXT_COLOR,
            font=("Segoe UI", 10),
        )
        self.help_label.pack(fill="x", padx=10, pady=(2, 10))

        root.bind("<space>", self.next_result)
        root.bind("<Right>", self.next_result)
        root.bind("<Left>", self.previous_result)
        root.bind("<Home>", self.first_result)
        root.bind("<End>", self.last_result)
        root.bind("<Escape>", self.close)
        root.bind("<KeyPress-q>", self.close)
        root.bind("<KeyPress-Q>", self.close)
        root.protocol("WM_DELETE_WINDOW", root.destroy)
        self.show_current()

    def show_current(self) -> None:
        from PIL import Image, ImageTk

        result = self.results[self.index]
        sheet = compose_contact_sheet(result)
        screen_width = max(640, int(self.root.winfo_screenwidth()) - 80)
        screen_height = max(480, int(self.root.winfo_screenheight()) - 170)
        sheet.thumbnail(
            (min(MAX_DISPLAY_WIDTH, screen_width), min(MAX_DISPLAY_HEIGHT, screen_height)),
            Image.Resampling.LANCZOS,
        )
        self.photo = ImageTk.PhotoImage(sheet)
        self.image_label.configure(image=self.photo)

        frame_ids = result["frame_ids"]
        frames = " → ".join(f"E{i}: {frame}" for i, frame in enumerate(frame_ids, start=1))
        info = (
            f"Result {self.index + 1}/{len(self.results)}  |  "
            f"Video rank {result['video_rank']}  |  "
            f"KIS rank {result['kis_rank']}/{KIS_RANKS_PER_VIDEO}  |  "
            f"{result['video_id']}  |  {frames}"
        )
        self.info_label.configure(text=info)
        self.root.title(
            f"TRAKE Viewer — Video {result['video_rank']} — "
            f"KIS {result['kis_rank']}/{KIS_RANKS_PER_VIDEO} — {result['video_id']}"
        )

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

    def first_result(self, _event: object = None) -> str:
        self.index = 0
        self.show_current()
        return "break"

    def last_result(self, _event: object = None) -> str:
        self.index = len(self.results) - 1
        self.show_current()
        return "break"

    def close(self, _event: object = None) -> str:
        self.root.destroy()
        return "break"


def main() -> None:
    import tkinter as tk

    configure_console_utf8()
    results = load_trake_csv(CSV_PATH.resolve())
    resolved = resolve_results(
        results,
        KEYFRAME_ROOT.resolve(),
        KEYFRAME_INDEX_PATH.resolve(),
    )
    event_count = len(results[0]["frame_ids"])
    print(
        f"Loaded {len(results)} CSV rows with {event_count} events; "
        f"displaying {len(resolved)} complete sequences"
    )

    root = tk.Tk()
    root.minsize(640, 480)
    if START_MAXIMIZED:
        try:
            root.state("zoomed")
        except tk.TclError:
            pass
    TRAKEKeyframeViewer(root, resolved)
    root.after(50, root.focus_force)
    root.mainloop()


if __name__ == "__main__":
    main()
