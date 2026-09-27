from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import cast

from .models import FrameRecord


REQUIRED_FIELDS = frozenset(
    {"video_id", "shot_id", "frame_id", "ocr_text", "asr_text", "objects"}
)
ELASTICSEARCH_LONG_MAX = 2**63 - 1
ELASTICSEARCH_INTEGER_MAX = 2**31 - 1
ELASTICSEARCH_ID_MAX_BYTES = 512


class _DuplicateJsonMember(ValueError):
    def __init__(self, key: str) -> None:
        super().__init__(key)
        self.key = key


def _reject_duplicate_json_members(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise _DuplicateJsonMember(key)
        payload[key] = value
    return payload


def _record_error(path: Path, line_number: int, message: str) -> ValueError:
    return ValueError(f"{path}: {message} at line {line_number}")


def _normalize_identifier(
    value: object, field: str, path: Path, line_number: int
) -> int:
    message = f"{field} must be a non-negative integer or digit string"
    if isinstance(value, bool):
        raise _record_error(path, line_number, message)
    if isinstance(value, int) and 0 <= value <= ELASTICSEARCH_LONG_MAX:
        return value
    if isinstance(value, str) and value and value.isascii() and value.isdigit():
        normalized = value.lstrip("0") or "0"
        maximum = str(ELASTICSEARCH_LONG_MAX)
        if len(normalized) <= len(maximum) and (
            len(normalized) < len(maximum) or normalized <= maximum
        ):
            return int(normalized)
    raise _record_error(path, line_number, message)


def _parse_record(
    payload: object,
    *,
    path: Path,
    line_number: int,
    labels: frozenset[str],
) -> FrameRecord:
    if not isinstance(payload, dict):
        raise _record_error(path, line_number, "line must contain a JSON object")
    keys = frozenset(payload)
    missing = sorted(REQUIRED_FIELDS - keys)
    unknown = sorted(keys - REQUIRED_FIELDS)
    if missing:
        raise _record_error(path, line_number, f"missing fields: {', '.join(missing)}")
    if unknown:
        raise _record_error(path, line_number, f"unknown fields: {', '.join(unknown)}")

    video_id = payload["video_id"]
    if not isinstance(video_id, str):
        raise _record_error(path, line_number, "video_id must be a string")
    if not video_id.strip():
        raise _record_error(path, line_number, "video_id must be nonblank")
    for field in ("ocr_text", "asr_text"):
        if not isinstance(payload[field], str):
            raise _record_error(path, line_number, f"{field} must be a string")

    raw_objects = payload["objects"]
    if not isinstance(raw_objects, dict):
        raise _record_error(path, line_number, "objects must be an object")
    normalized_objects: list[tuple[str, int]] = []
    for label, count in raw_objects.items():
        if not isinstance(label, str) or label not in labels:
            raise _record_error(path, line_number, f"unknown object label '{label}'")
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or count < 0
            or count > ELASTICSEARCH_INTEGER_MAX
        ):
            raise _record_error(
                path,
                line_number,
                f"object count for '{label}' must be a non-negative integer",
            )
        normalized_objects.append((label, count))

    record = FrameRecord(
        video_id=video_id,
        shot_id=_normalize_identifier(payload["shot_id"], "shot_id", path, line_number),
        frame_id=_normalize_identifier(payload["frame_id"], "frame_id", path, line_number),
        ocr_text=cast(str, payload["ocr_text"]),
        asr_text=cast(str, payload["asr_text"]),
        objects=tuple(sorted(normalized_objects)),
    )
    try:
        keyframe_id_bytes = len(record.keyframe_id.encode("utf-8"))
    except UnicodeEncodeError as error:
        raise _record_error(
            path, line_number, "video_id must be UTF-8 encodable"
        ) from error
    if keyframe_id_bytes > ELASTICSEARCH_ID_MAX_BYTES:
        raise _record_error(
            path,
            line_number,
            "frame identity exceeds 512 UTF-8 bytes",
        )
    return record


def iter_records(path: Path, labels: tuple[str, ...]) -> Iterator[FrameRecord]:
    label_set = frozenset(labels)
    row_count = 0
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, raw_line in enumerate(stream, start=1):
                if not raw_line.strip():
                    raise _record_error(path, line_number, f"blank line {line_number}")
                try:
                    payload = json.loads(
                        raw_line, object_pairs_hook=_reject_duplicate_json_members
                    )
                except _DuplicateJsonMember as error:
                    raise _record_error(
                        path,
                        line_number,
                        f"duplicate JSON member '{error.key}'",
                    ) from error
                except json.JSONDecodeError as error:
                    raise _record_error(
                        path, line_number, f"invalid JSON at line {line_number}: {error}"
                    ) from error
                except ValueError as error:
                    raise _record_error(
                        path, line_number, f"invalid JSON at line {line_number}: {error}"
                    ) from error
                row_count += 1
                yield _parse_record(
                    payload, path=path, line_number=line_number, labels=label_set
                )
    except (OSError, UnicodeError) as error:
        raise RuntimeError(f"cannot read JSONL '{path}': {error}") from error
    if row_count == 0:
        raise ValueError(f"JSONL '{path}' must contain at least one record")


def preflight_jsonl(path: Path, labels: tuple[str, ...]) -> int:
    identities: set[str] = set()
    row_count = 0
    for row_count, record in enumerate(iter_records(path, labels), start=1):
        if record.keyframe_id in identities:
            raise ValueError(
                f"{path}: duplicate frame identity '{record.keyframe_id}' at line {row_count}"
            )
        identities.add(record.keyframe_id)
    return row_count


def iter_bulk_actions(
    path: Path, labels: tuple[str, ...], index_name: str
) -> Iterator[dict[str, object]]:
    for record in iter_records(path, labels):
        yield {
            "_op_type": "create",
            "_index": index_name,
            "_id": record.keyframe_id,
            "_source": record.to_source(),
        }


def load_labels(path: Path) -> tuple[str, ...]:
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise RuntimeError(f"cannot read label file '{path}': {error}") from error
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in label file '{path}': {error}") from error
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"label file '{path}' must contain a non-empty JSON array")

    labels: list[str] = []
    seen: set[str] = set()
    for index, label in enumerate(payload):
        if not isinstance(label, str):
            raise ValueError(f"label at index {index} must be a string")
        if not label.strip():
            raise ValueError(f"label at index {index} must be nonblank")
        if label != label.strip():
            raise ValueError(
                f"label at index {index} must not have surrounding whitespace"
            )
        if "." in label:
            raise ValueError(f"label at index {index} must not contain '.'")
        if label in seen:
            raise ValueError(f"duplicate label '{label}'")
        seen.add(label)
        labels.append(label)
    return tuple(labels)
