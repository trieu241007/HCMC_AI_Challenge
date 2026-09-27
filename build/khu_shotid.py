import csv
import json
import math
from pathlib import Path
from typing import TextIO

import cv2
import numpy as np


INPUT_FILE = Path(r"pe_core_search_results.jsonl")
OUTPUT_FILE = Path("results_visual_dedup.csv")
LOG_FILE = Path("visual_dedup.log")
KEYFRAME_ROOT = Path(r"C:\linhtinh\HCMC AI\project\data\processed\keyframes")
ANSWER_COUNT = 100
VISUAL_DUPLICATE_THRESHOLD = 0.8
PHASH_MAX_DISTANCE = 24
SSIM_IMAGE_SIZE = 128


def read_jsonl(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Line {line_number} is not a JSON object")
            records.append(record)
    return records


def validate_hits(hits: list[object]) -> list[dict[str, object]]:
    validated: list[dict[str, object]] = []
    for position, raw_hit in enumerate(hits, start=1):
        if not isinstance(raw_hit, dict):
            raise ValueError(f"Result {position} is not a JSON object")
        hit = dict(raw_hit)
        score = hit.get("score")
        frame_id = hit.get("frame_id")
        if (
            not isinstance(score, (int, float))
            or isinstance(score, bool)
            or not math.isfinite(float(score))
        ):
            raise ValueError(f"Result {position} has an invalid score")
        if not isinstance(frame_id, int) or isinstance(frame_id, bool):
            raise ValueError(f"Result {position} has an invalid frame_id")
        if "video_id" not in hit or "shot_id" not in hit:
            raise ValueError(f"Result {position} is missing video_id or shot_id")

        hit["score"] = float(score)
        validated.append(hit)
    return validated


def keyframe_path(keyframe_root: Path, hit: dict[str, object]) -> Path:
    video_id = hit.get("video_id")
    shot_id = hit.get("shot_id")
    frame_id = hit.get("frame_id")
    if not isinstance(video_id, str) or not video_id:
        raise ValueError("Result has an invalid video_id")
    if not isinstance(shot_id, int) or isinstance(shot_id, bool):
        raise ValueError("Result has an invalid shot_id")
    if not isinstance(frame_id, int) or isinstance(frame_id, bool):
        raise ValueError("Result has an invalid frame_id")
    return keyframe_root / (
        f"{video_id}_s{shot_id:06d}_f{frame_id:09d}.jpg"
    )


def load_grayscale_keyframe(path: Path, image_size: int) -> np.ndarray:
    if image_size <= 0:
        raise ValueError("SSIM image size must be positive")
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read keyframe: {path}")
    resized = cv2.resize(
        image,
        (image_size, image_size),
        interpolation=cv2.INTER_AREA,
    )
    return resized.astype(np.float64)


def ssim_similarity(first: np.ndarray, second: np.ndarray) -> float:
    if first.shape != second.shape:
        raise ValueError("SSIM images must have the same shape")

    constant_1 = (0.01 * 255.0) ** 2
    constant_2 = (0.03 * 255.0) ** 2
    mean_first = cv2.GaussianBlur(first, (11, 11), 1.5)
    mean_second = cv2.GaussianBlur(second, (11, 11), 1.5)
    mean_first_squared = mean_first * mean_first
    mean_second_squared = mean_second * mean_second
    mean_product = mean_first * mean_second

    variance_first = (
        cv2.GaussianBlur(first * first, (11, 11), 1.5)
        - mean_first_squared
    )
    variance_second = (
        cv2.GaussianBlur(second * second, (11, 11), 1.5)
        - mean_second_squared
    )
    covariance = (
        cv2.GaussianBlur(first * second, (11, 11), 1.5)
        - mean_product
    )

    numerator = (2.0 * mean_product + constant_1) * (
        2.0 * covariance + constant_2
    )
    denominator = (
        mean_first_squared + mean_second_squared + constant_1
    ) * (variance_first + variance_second + constant_2)
    score = float(np.mean(numerator / denominator))
    return max(0.0, min(1.0, score))


def perceptual_hash(image: np.ndarray) -> np.ndarray:
    resized = cv2.resize(
        image,
        (32, 32),
        interpolation=cv2.INTER_AREA,
    ).astype(np.float32)
    low_frequency = cv2.dct(resized)[:8, :8]
    median = float(np.median(low_frequency.reshape(-1)[1:]))
    return low_frequency > median


def perceptual_hash_distance(first: np.ndarray, second: np.ndarray) -> int:
    if first.shape != second.shape:
        raise ValueError("Perceptual hashes must have the same shape")
    return int(np.count_nonzero(first != second))


def filter_visual_duplicates(
    hits: list[dict[str, object]],
    *,
    keyframe_root: Path,
    threshold: float,
    image_size: int = SSIM_IMAGE_SIZE,
    phash_max_distance: int = PHASH_MAX_DISTANCE,
    query_id: str | None = None,
    log_handle: TextIO | None = None,
) -> list[dict[str, object]]:
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("Visual duplicate threshold must be between 0 and 1")
    if not 0 <= phash_max_distance <= 64:
        raise ValueError("pHash maximum distance must be between 0 and 64")

    filtered: list[dict[str, object]] = []
    anchor_images: dict[tuple[object, object], np.ndarray] = {}
    anchor_hashes: dict[tuple[object, object], np.ndarray] = {}
    anchor_paths: dict[tuple[object, object], Path] = {}
    image_cache: dict[Path, np.ndarray] = {}
    hash_cache: dict[Path, np.ndarray] = {}

    for hit in hits:
        group = (hit.get("video_id"), hit.get("shot_id"))
        path = keyframe_path(keyframe_root, hit)
        image = image_cache.get(path)
        if image is None:
            image = load_grayscale_keyframe(path, image_size)
            image_cache[path] = image
        image_hash = hash_cache.get(path)
        if image_hash is None:
            image_hash = perceptual_hash(image)
            hash_cache[path] = image_hash

        anchor = anchor_images.get(group)
        if anchor is None:
            anchor_images[group] = image
            anchor_hashes[group] = image_hash
            anchor_paths[group] = path
            filtered.append(hit)
            continue

        similarity = ssim_similarity(anchor, image)
        hash_distance = perceptual_hash_distance(
            anchor_hashes[group],
            image_hash,
        )
        has_visual_detail = float(np.std(anchor)) >= 5.0 and float(
            np.std(image)
        ) >= 5.0
        ssim_duplicate = similarity >= threshold
        phash_duplicate = (
            has_visual_detail and hash_distance <= phash_max_distance
        )
        is_duplicate = ssim_duplicate or phash_duplicate
        reasons: list[str] = []
        if ssim_duplicate:
            reasons.append("SSIM")
        if phash_duplicate:
            reasons.append("PHASH")
        reason = "+".join(reasons) if reasons else "NONE"
        if log_handle is not None:
            anchor_path = anchor_paths[group]
            log_handle.write(
                f'query={query_id or "UNKNOWN"} image="{path.name}" '
                f'compare_to="{anchor_path.name}" ssim={similarity:.6f} '
                f'phash_distance={hash_distance} '
                f'result={"DELETE" if is_duplicate else "KEEP"} '
                f'reason={reason}\n'
            )

        if not is_duplicate:
            filtered.append(hit)

    return filtered


def prepare_ranked_hits(
    query_row: int,
    results: list[object],
    *,
    keyframe_root: Path = KEYFRAME_ROOT,
    duplicate_threshold: float = VISUAL_DUPLICATE_THRESHOLD,
    query_id: str | None = None,
    log_handle: TextIO | None = None,
) -> list[dict[str, object]]:
    validated = validate_hits(results)
    ranked = sorted(
        validated,
        key=lambda hit: float(hit["score"]),
        reverse=True,
    )
    return filter_visual_duplicates(
        ranked,
        keyframe_root=keyframe_root,
        threshold=duplicate_threshold,
        query_id=query_id,
        log_handle=log_handle,
    )


def process_to_wide_csv(
    input_path: Path,
    output_path: Path,
    *,
    log_path: Path = LOG_FILE,
    keyframe_root: Path = KEYFRAME_ROOT,
) -> None:
    queries = read_jsonl(input_path)
    headers = ["query_id"] + [
        f"answer_{index:03d}" for index in range(1, ANSWER_COUNT + 1)
    ]

    with (
        output_path.open("w", newline="", encoding="utf-8") as handle,
        log_path.open("w", encoding="utf-8") as log_handle,
    ):
        writer = csv.writer(handle)
        writer.writerow(headers)

        for query in queries:
            query_row = query.get("query_row")
            if not isinstance(query_row, int) or isinstance(query_row, bool):
                raise ValueError("query_row must be an integer")
            formatted_query_id = f"Q_{query_row + 1:04d}"

            results = query.get("results")
            if not isinstance(results, list):
                raise ValueError(f"results for query_row {query_row} must be a list")
            hits = prepare_ranked_hits(
                query_row,
                results,
                keyframe_root=keyframe_root,
                query_id=formatted_query_id,
                log_handle=log_handle,
            )

            answers: list[str] = []
            for hit in hits:
                video_id = hit["video_id"]
                frame_id = hit["frame_id"]

                answers.append(f"{video_id}.mp4,{frame_id}")
                if len(answers) == ANSWER_COUNT:
                    break

            padding = answers[-1] if answers else "video0000.mp4,0"
            answers.extend([padding] * (ANSWER_COUNT - len(answers)))
            writer.writerow([formatted_query_id, *answers])

    print(f"Converted JSONL search results to: {output_path}")
    print(f"Wrote visual comparison log to: {log_path}")


if __name__ == "__main__":
    process_to_wide_csv(INPUT_FILE, OUTPUT_FILE, log_path=LOG_FILE)
