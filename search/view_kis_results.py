"""Display Textual KIS CSV results as keyboard-navigable keyframes.

Edit CSV_PATH, KEYFRAME_CACHE_ROOT, and KEYFRAME_INDEX_PATH, then run this
file directly. Missing result images are downloaded individually from Kaggle.
There is no CLI.

Controls:
    Space / Right Arrow  next result
    Left Arrow           previous result
    Q / Escape           close viewer
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
import time
import tempfile
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath
import search_kis

# ============================================================
# Configuration: edit these paths.
# ============================================================
csv_name = search_kis.name_output
CSV_PATH = Path(rf"C:\linhtinh\HCMC AI\deloy\log\{csv_name}.csv")
KAGGLE_DATASET = "trieu241007/keyframes-p1"
KEYFRAME_CACHE_ROOT = Path(r"C:\linhtinh\HCMC AI\deloy\keyframe_cache")
KEYFRAME_INDEX_PATH = Path(r"C:\linhtinh\HCMC AI\deloy\keyframes.jsonl")
MAX_DOWNLOAD_WORKERS = 100

MAX_IMAGE_WIDTH = 1400
MAX_IMAGE_HEIGHT = 850
BACKGROUND_COLOR = "#111111"
TEXT_COLOR = "#f2f2f2"


KISResult = dict[str, object]
KaggleDownloader = Callable[[str, Path], Path]


def format_load_summary(image_count: int, elapsed_seconds: float) -> str:
    """Format batch duration and throughput for terminal output."""

    if image_count < 0:
        raise ValueError("image_count must be non-negative")
    if elapsed_seconds < 0:
        raise ValueError("elapsed_seconds must be non-negative")

    average = elapsed_seconds / image_count if image_count else 0.0
    speed = image_count / elapsed_seconds if elapsed_seconds else 0.0
    return (
        f"Image loading finished: {image_count} images in "
        f"{elapsed_seconds:.2f} seconds\n"
        f"Average: {average:.3f} seconds/image | "
        f"Speed: {speed:.2f} images/second"
    )


def configure_console_utf8() -> None:
    """Allow Vietnamese messages in legacy Windows terminals."""

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="backslashreplace")


def load_kis_csv(path: Path) -> list[KISResult]:
    """Read headerless website-format rows: video_id,frame_id[,shot_id]."""

    if not path.is_file():
        raise FileNotFoundError(f"KIS CSV does not exist: {path}")

    results: list[KISResult] = []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream)
            for line_number, row in enumerate(reader, start=1):
                if not row or all(not value.strip() for value in row):
                    continue
                if len(row) not in (2, 3):
                    raise ValueError(
                        f"CSV line {line_number} must contain "
                        "video_id,frame_id[,shot_id]"
                    )

                video_id = row[0].strip()
                raw_frame_id = row[1].strip()
                if not video_id:
                    raise ValueError(f"CSV line {line_number} has an empty video_id")
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

                shot_id = None
                if len(row) == 3 and row[2].strip():
                    try:
                        shot_id = int(row[2].strip())
                        if shot_id < 0:
                            shot_id = None
                    except ValueError:
                        shot_id = None

                item: KISResult = {
                    "rank": len(results) + 1,
                    "video_id": video_id,
                    "frame_id": frame_id,
                }
                if shot_id is not None:
                    item["shot_id"] = shot_id

                results.append(item)
    except (OSError, UnicodeError, csv.Error) as error:
        raise RuntimeError(f"Cannot read KIS CSV '{path}': {error}") from error

    if not results:
        raise ValueError(f"KIS CSV contains no result rows: {path}")
    return results


def keyframe_video_prefix(video_id: str) -> str:
    """Convert website ID L21_V001 to the keyframe prefix l21-v001."""

    normalized = video_id.strip()
    if normalized.lower().endswith(".mp4"):
        normalized = normalized[:-4]
    match = re.fullmatch(r"[lL](\d+)[-_][vV](\d+)", normalized)
    if match is None:
        raise ValueError(
            f"Unsupported video_id {video_id!r}; expected the supported "
            "L<digits>_V<digits> format"
        )
    return f"l{match.group(1)}-v{match.group(2)}"


def result_key(result: Mapping[str, object], source: str) -> tuple[str, int, int | None]:
    """Return the normalized lookup key shared by CSV and JSONL records."""

    video_id = result.get("video_id")
    frame_id = result.get("frame_id")
    shot_id = result.get("shot_id")
    if not isinstance(video_id, str) or not video_id.strip():
        raise ValueError(f"{source} video_id must be a non-empty string")
    if isinstance(frame_id, bool) or not isinstance(frame_id, int) or frame_id < 0:
        raise ValueError(f"{source} frame_id must be a non-negative integer")
    if shot_id is not None and (isinstance(shot_id, bool) or not isinstance(shot_id, int) or shot_id < 0):
        raise ValueError(f"{source} shot_id must be a non-negative integer or None")
    return keyframe_video_prefix(video_id), frame_id, shot_id


def kaggle_path_from_metadata(
    metadata: Mapping[str, object],
    line_number: int,
) -> str:
    """Build the exact Kaggle dataset path from one JSONL metadata record."""

    video_prefix, frame_id, _ = result_key(
        metadata, f"Keyframe JSONL line {line_number}"
    )
    shot_id = metadata.get("shot_id")
    if isinstance(shot_id, bool) or not isinstance(shot_id, int) or shot_id < 0:
        raise ValueError(
            f"Keyframe JSONL line {line_number} shot_id must be "
            "a non-negative integer"
        )

    stem = f"{video_prefix}_s{shot_id:06d}_f{frame_id:09d}"
    return f"keyframes/{stem}.jpg"


def download_kaggle_file(dataset_path: str, output_dir: Path) -> Path:
    """Download one public Kaggle dataset file into the local cache with retries."""

    try:
        import kagglehub
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "kagglehub>=1.0.2 is required. Install it with: "
            "pip install -U \"kagglehub>=1.0.2\""
        ) from error

    last_err = None
    for attempt in range(3):
        try:
            downloaded_path = kagglehub.dataset_download(
                KAGGLE_DATASET,
                path=dataset_path,
                output_dir=str(output_dir),
            )
            p = Path(downloaded_path)
            if p.is_file() and is_decodable_jpeg(p):
                return p
        except TypeError as error:
            if "output_dir" in str(error):
                raise RuntimeError(
                    "Upgrade kagglehub to a version supporting output_dir with: "
                    "pip install -U \"kagglehub>=1.0.2\""
                ) from error
            last_err = error
        except Exception as error:
            last_err = error
        time.sleep(1)

    raise RuntimeError(
        f"Cannot download '{dataset_path}' from Kaggle dataset "
        f"'{KAGGLE_DATASET}': {last_err}"
    )


def is_decodable_jpeg(path: Path) -> bool:
    """Fully decode a JPEG so corrupt cache entries are never reused."""

    try:
        from PIL import Image
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Pillow is required. Install it with: pip install -U pillow"
        ) from error
    try:
        with Image.open(path) as image:
            if image.format != "JPEG":
                return False
            image.load()
    except (OSError, SyntaxError, ValueError):
        return False
    return True


def resolve_exact_cache_path(
    metadata: Mapping[str, object],
    keyframe_cache_root: Path,
) -> Path | None:
    """Return the exact cached image path constructed from metadata if file exists and is valid JPEG."""

    try:
        video_id = metadata.get("video_id")
        frame_id = metadata.get("frame_id")
        shot_id = metadata.get("shot_id")
        if not isinstance(video_id, str) or not video_id.strip():
            return None
        if isinstance(frame_id, bool) or not isinstance(frame_id, int) or frame_id < 0:
            return None
        if isinstance(shot_id, bool) or not isinstance(shot_id, int) or shot_id < 0:
            return None

        v_prefix = keyframe_video_prefix(video_id)
        stem = f"{v_prefix}_s{shot_id:06d}_f{frame_id:09d}"
        dataset_path = f"keyframes/{stem}.jpg"
        resolved_cache_root = keyframe_cache_root.resolve()
        cached_path = resolved_cache_root.joinpath(
            *PurePosixPath(dataset_path).parts
        ).resolve()
        
        cached_path.relative_to(resolved_cache_root)
        if cached_path.is_file() and is_decodable_jpeg(cached_path):
            return cached_path
    except Exception:
        pass
    return None


def image_path_from_metadata(
    metadata: Mapping[str, object],
    keyframe_cache_root: Path,
    line_number: int,
    downloader: KaggleDownloader,
) -> Path:
    """Return a cached keyframe path, downloading only when it is missing."""

    dataset_path = kaggle_path_from_metadata(metadata, line_number)
    resolved_cache_root = keyframe_cache_root.resolve()
    cached_path = resolved_cache_root.joinpath(
        *PurePosixPath(dataset_path).parts
    ).resolve()
    try:
        cached_path.relative_to(resolved_cache_root)
    except ValueError as error:
        raise ValueError(
            f"Resolved Kaggle cache path escapes its root: {cached_path}"
        ) from error

    if cached_path.is_file():
        if is_decodable_jpeg(cached_path):
            return cached_path
        print(f"Discarding invalid cached image: {cached_path}")
        cached_path.unlink()

    print(f"Downloading Kaggle image: {dataset_path}")
    downloaded_path = Path(
        downloader(dataset_path, resolved_cache_root)
    ).resolve()
    try:
        downloaded_path.relative_to(resolved_cache_root)
    except ValueError as error:
        raise ValueError(
            f"Kaggle downloader returned a path outside the cache: "
            f"{downloaded_path}"
        ) from error

    if downloaded_path == cached_path and is_decodable_jpeg(cached_path):
        return cached_path
    if downloaded_path.is_file() and not is_decodable_jpeg(downloaded_path):
        downloaded_path.unlink()
    raise FileNotFoundError(
        "Kaggle download did not create a complete JPEG at the expected path: "
        f"{cached_path}"
    )


def resolve_results(
    results: Sequence[Mapping[str, object]],
    keyframe_cache_root: Path,
    keyframe_index_path: Path,
    downloader: KaggleDownloader | None = None,
    max_workers: int = MAX_DOWNLOAD_WORKERS,
) -> list[KISResult]:
    """Resolve CSV rows and fetch their Kaggle images as one worker batch."""

    if not keyframe_index_path.is_file():
        raise FileNotFoundError(
            f"Keyframe JSONL index does not exist: {keyframe_index_path}"
        )
    if (
        isinstance(max_workers, bool)
        or not isinstance(max_workers, int)
        or max_workers < 1
    ):
        raise ValueError("max_workers must be a positive integer")
    keyframe_cache_root.mkdir(parents=True, exist_ok=True)
    active_downloader = downloader or download_kaggle_file

    wanted = {result_key(result, "CSV result") for result in results}
    matched_metadata: dict[
        tuple[str, int, int | None], tuple[Mapping[str, object], int]
    ] = {}
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
                    raise ValueError(
                        f"Keyframe JSONL line {line_number} must be an object"
                    )

                v_prefix, frame_id, meta_shot_id = result_key(metadata, f"Keyframe JSONL line {line_number}")
                
                # Check for exact key match first
                exact_key = (v_prefix, frame_id, meta_shot_id)
                no_shot_key = (v_prefix, frame_id, None)

                matched_any = False
                if exact_key in wanted and exact_key not in matched_metadata:
                    matched_metadata[exact_key] = (metadata, line_number)
                    matched_any = True
                if no_shot_key in wanted and no_shot_key not in matched_metadata:
                    matched_metadata[no_shot_key] = (metadata, line_number)
                    matched_any = True

                if len(matched_metadata) == len(wanted):
                    break
    except (OSError, UnicodeError) as error:
        raise RuntimeError(
            f"Cannot read keyframe JSONL index '{keyframe_index_path}': {error}"
        ) from error

    found: dict[tuple[str, int, int | None], Path] = {}
    if matched_metadata:
        worker_count = min(max_workers, len(matched_metadata))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_key = {
                executor.submit(
                    image_path_from_metadata,
                    metadata,
                    keyframe_cache_root,
                    line_number,
                    active_downloader,
                ): key
                for key, (metadata, line_number) in matched_metadata.items()
            }
            total = len(future_to_key)
            for completed, future in enumerate(
                as_completed(future_to_key),
                start=1,
            ):
                key = future_to_key[future]
                try:
                    found[key] = future.result()
                except Exception as error:
                    for pending_future in future_to_key:
                        pending_future.cancel()
                    executor.shutdown(wait=True, cancel_futures=True)
                    metadata, line_number = matched_metadata[key]
                    failed_path = kaggle_path_from_metadata(
                        metadata,
                        line_number,
                    )
                    raise RuntimeError(
                        f"Kaggle batch failed for '{failed_path}': {error}"
                    ) from error
                print(f"Kaggle images ready: {completed}/{total}")

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
        raise FileNotFoundError("None of the CSV results matched a keyframe image")
    return resolved


class KISKeyframeViewer:
    """A single-window, keyboard-controlled KIS result viewer."""

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


def build_search_neighbors_sidecar(
    raw_rows: Sequence[Mapping[str, object]],
    output_sidecar_path: Path,
    keyframe_index_path: Path = KEYFRAME_INDEX_PATH,
    radius: int = 10,
) -> None:
    """Scan keyframes.jsonl ONCE for all video_ids in raw_rows and write atomic neighbors sidecar JSON."""

    if not keyframe_index_path.is_file():
        raise FileNotFoundError(
            f"Keyframe JSONL index does not exist: {keyframe_index_path}"
        )

    # Collect target keys and unique video prefixes
    targets: dict[tuple[str, int], dict[str, object]] = {}
    target_prefixes: set[str] = set()
    for row in raw_rows[:100]:
        v_id = str(row["video_id"]).strip()
        f_id = int(row["frame_id"])
        prefix = keyframe_video_prefix(v_id)
        targets[(prefix, f_id)] = {"video_id": v_id, "frame_id": f_id}
        target_prefixes.add(prefix)

    if not targets:
        sidecar_data = {}
    else:
        # One pass scan over keyframes.jsonl for all target_prefixes
        prefix_items: dict[str, list[dict[str, object]]] = {p: [] for p in target_prefixes}
        try:
            with keyframe_index_path.open("r", encoding="utf-8") as stream:
                for line_number, raw_line in enumerate(stream, start=1):
                    if not raw_line.strip():
                        continue
                    try:
                        metadata = json.loads(raw_line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(metadata, dict):
                        continue
                    
                    try:
                        v_prefix, frame_id, _ = result_key(metadata, f"Keyframe JSONL line {line_number}")
                    except ValueError:
                        continue

                    if v_prefix in prefix_items:
                        prefix_items[v_prefix].append({
                            "video_id": metadata.get("video_id"),
                            "frame_id": frame_id,
                            "shot_id": metadata.get("shot_id"),
                        })
        except (OSError, UnicodeError) as error:
            raise RuntimeError(
                f"Cannot read keyframe JSONL index '{keyframe_index_path}': {error}"
            ) from error

        # Build neighbor maps for each target and emit entry for all neighbor items as well
        sidecar_data: dict[str, list[dict[str, object]]] = {}
        for (prefix, f_id), target_info in targets.items():
            items = prefix_items.get(prefix, [])
            if not items:
                canonical_key = f"{target_info['video_id']}:{f_id}"
                sidecar_data[canonical_key] = []
                continue

            items.sort(key=lambda x: int(x["frame_id"]))
            target_idx = -1
            for idx, item in enumerate(items):
                if int(item["frame_id"]) == f_id:
                    target_idx = idx
                    break

            if target_idx != -1:
                start_idx = max(0, target_idx - radius)
                end_idx = min(len(items), target_idx + radius + 1)
                selected = items[start_idx:end_idx]
            else:
                selected = items[: 2 * radius + 1]

            neighbor_list = [
                {
                    "video_id": item["video_id"] or target_info["video_id"],
                    "frame_id": int(item["frame_id"]),
                    "shot_id": item.get("shot_id"),
                }
                for item in selected
            ]

            # Emit canonical key for target
            canonical_key = f"{target_info['video_id']}:{f_id}"
            sidecar_data[canonical_key] = neighbor_list

            # Emit sidecar entries for neighbor frames within the same video cluster if not present
            for item in selected:
                item_v_id = item["video_id"] or target_info["video_id"]
                item_f_id = int(item["frame_id"])
                item_key = f"{item_v_id}:{item_f_id}"
                if item_key not in sidecar_data:
                    item_target_idx = -1
                    for idx, sub_item in enumerate(items):
                        if int(sub_item["frame_id"]) == item_f_id:
                            item_target_idx = idx
                            break
                    if item_target_idx != -1:
                        s_idx = max(0, item_target_idx - radius)
                        e_idx = min(len(items), item_target_idx + radius + 1)
                        sub_selected = items[s_idx:e_idx]
                    else:
                        sub_selected = selected
                    sidecar_data[item_key] = [
                        {
                            "video_id": sub["video_id"] or target_info["video_id"],
                            "frame_id": int(sub["frame_id"]),
                            "shot_id": sub.get("shot_id"),
                        }
                        for sub in sub_selected
                    ]

    # Atomic write to sidecar JSON file
    output_sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    temp_sidecar: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_sidecar_path.parent,
            prefix=f".{output_sidecar_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temp_sidecar = Path(stream.name)
            json.dump(sidecar_data, stream, ensure_ascii=False, indent=2)
        os.replace(temp_sidecar, output_sidecar_path)
    except Exception as error:
        if temp_sidecar and temp_sidecar.exists():
            temp_sidecar.unlink(missing_ok=True)
        raise RuntimeError(f"Cannot write neighbors sidecar '{output_sidecar_path}': {error}") from error


def load_neighbor_keyframes(
    video_id: str,
    target_frame_id: int,
    keyframe_index_path: Path = KEYFRAME_INDEX_PATH,
    radius: int = 10,
) -> list[KISResult]:
    """Stream keyframes.jsonl to find all records for target video, returning up to 2*radius+1 centered items."""

    if not keyframe_index_path.is_file():
        raise FileNotFoundError(
            f"Keyframe JSONL index does not exist: {keyframe_index_path}"
        )
    if radius <= 0 or radius > 10:
        raise ValueError("radius must be between 1 and 10")

    target_prefix = keyframe_video_prefix(video_id)
    same_video_items: list[dict[str, object]] = []

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
                    continue
                
                v_prefix, frame_id, _ = result_key(metadata, f"Keyframe JSONL line {line_number}")
                if v_prefix == target_prefix:
                    same_video_items.append({
                        "video_id": metadata.get("video_id", video_id),
                        "frame_id": frame_id,
                        "shot_id": metadata.get("shot_id"),
                    })
    except (OSError, UnicodeError) as error:
        raise RuntimeError(
            f"Cannot read keyframe JSONL index '{keyframe_index_path}': {error}"
        ) from error

    if not same_video_items:
        return []

    same_video_items.sort(key=lambda item: int(item["frame_id"]))

    target_index = -1
    for idx, item in enumerate(same_video_items):
        if int(item["frame_id"]) == target_frame_id:
            target_index = idx
            break

    if target_index != -1:
        start_idx = max(0, target_index - radius)
        end_idx = min(len(same_video_items), target_index + radius + 1)
        selected = same_video_items[start_idx:end_idx]
    else:
        # Fallback to closest frames
        selected = same_video_items[: 2 * radius + 1]

    results: list[KISResult] = []
    for rank, item in enumerate(selected, start=1):
        results.append({
            "rank": rank,
            "video_id": item["video_id"],
            "frame_id": item["frame_id"],
            "shot_id": item.get("shot_id"),
        })

    return results


def main() -> None:
    import tkinter as tk

    configure_console_utf8()
    csv_path = CSV_PATH.resolve()
    keyframe_cache_root = KEYFRAME_CACHE_ROOT.resolve()
    keyframe_index_path = KEYFRAME_INDEX_PATH.resolve()
    results = load_kis_csv(csv_path)

    print(f"Preparing {len(results)} CSV result images...")
    load_started_at = time.perf_counter()
    resolved = resolve_results(results, keyframe_cache_root, keyframe_index_path)
    load_elapsed_seconds = time.perf_counter() - load_started_at
    print(format_load_summary(len(resolved), load_elapsed_seconds))
    print(f"Loaded {len(results)} CSV rows; displaying {len(resolved)} keyframes")

    root = tk.Tk()
    KISKeyframeViewer(root, resolved)
    root.after(50, root.focus_force)
    root.mainloop()


if __name__ == "__main__":
    main()
