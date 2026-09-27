from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from elastic_transport import ApiResponse


QUERY_FIELDS = frozenset(
    {
        "query_text",
        "asr_text",
        "ocr_text",
        "asr_weight",
        "ocr_weight",
        "object",
    }
)
SOURCE_FIELDS = ["video_id", "shot_id", "frame_id"]


def _load_query(path: Path) -> tuple[str, float, str, float, dict[str, int]]:
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise RuntimeError(f"cannot read query file '{path}': {error}") from error
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid JSON in query file '{path}': {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("query JSON must contain an object")

    unknown = sorted(set(payload) - QUERY_FIELDS)
    if unknown:
        raise ValueError(f"unknown query fields: {', '.join(unknown)}")
    if "query_text" not in payload:
        raise ValueError("missing query field: query_text")

    for field in ("query_text", "asr_text", "ocr_text"):
        value = payload.get(field, "")
        if not isinstance(value, str):
            raise ValueError(f"{field} must be a string")

    asr_weight = _positive_weight(payload.get("asr_weight", 1.0), "asr_weight")
    ocr_weight = _positive_weight(payload.get("ocr_weight", 0.4), "ocr_weight")
    objects = _normalize_objects(payload.get("object"))
    return (
        payload.get("asr_text", "").strip(),
        asr_weight,
        payload.get("ocr_text", "").strip(),
        ocr_weight,
        objects,
    )


def _positive_weight(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a positive finite number")
    try:
        normalized = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{field} must be a positive finite number") from error
    if not math.isfinite(normalized) or normalized <= 0:
        raise ValueError(f"{field} must be a positive finite number")
    return normalized


def _normalize_objects(value: object) -> dict[str, int]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("object must be a JSON object or null")

    normalized: dict[str, int] = {}
    for label, count in value.items():
        if not label.strip() or label != label.strip():
            raise ValueError(f"object label '{label}' must be nonblank and trimmed")
        if "." in label:
            raise ValueError(f"object label '{label}' must not contain '.'")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(
                f"object count for '{label}' must be a non-negative integer"
            )
        if count > 0:
            normalized[label] = count
    return normalized


def _build_query(
    asr_text: str,
    asr_weight: float,
    ocr_text: str,
    ocr_weight: float,
    objects: Mapping[str, int],
) -> dict[str, object]:
    boolean: dict[str, object] = {}
    should: list[dict[str, object]] = []
    if asr_text:
        should.append(
            {
                "match": {
                    "asr_text": {
                        "query": asr_text,
                        "boost": asr_weight,
                    }
                }
            }
        )
    if ocr_text:
        should.append(
            {
                "match": {
                    "ocr_text": {
                        "query": ocr_text,
                        "boost": ocr_weight,
                    }
                }
            }
        )
    if should:
        boolean["should"] = should
        boolean["minimum_should_match"] = 1

    filters = [
        {"range": {f"objects.{label}": {"gte": count}}}
        for label, count in sorted(objects.items())
    ]
    if filters:
        boolean["filter"] = filters
    return {"bool": boolean}


def _search(
    client: object,
    index_name: str,
    query: Mapping[str, object],
    size: int,
) -> object:
    try:
        return getattr(client, "search")(
            index=index_name,
            query=query,
            size=size,
            source_includes=SOURCE_FIELDS,
            track_total_hits=False,
            allow_partial_search_results=False,
        )
    except Exception as error:
        raise RuntimeError(
            f"Elasticsearch search failed for index '{index_name}': {error}"
        ) from error


def _invalid_response(index_name: str) -> RuntimeError:
    return RuntimeError(
        f"Elasticsearch returned an invalid search response for index '{index_name}'"
    )


def _response_hits(response: object, index_name: str) -> list[object]:
    if isinstance(response, ApiResponse):
        response = response.body
    if not isinstance(response, Mapping):
        raise _invalid_response(index_name)
    if response.get("timed_out") is not False:
        raise _invalid_response(index_name)
    shards = response.get("_shards")
    if not isinstance(shards, Mapping):
        raise _invalid_response(index_name)
    validated_shards: dict[str, int] = {}
    for field in ("total", "successful", "skipped", "failed"):
        value = shards.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise _invalid_response(index_name)
        validated_shards[field] = value
    if (
        validated_shards["failed"] != 0
        or validated_shards["successful"] != validated_shards["total"]
        or validated_shards["skipped"] > validated_shards["successful"]
    ):
        raise _invalid_response(index_name)
    hits = response.get("hits")
    if not isinstance(hits, Mapping):
        raise _invalid_response(index_name)
    raw_hits = hits.get("hits")
    if not isinstance(raw_hits, list):
        raise _invalid_response(index_name)
    return raw_hits


def _normalize_hit(
    hit: object, rank: int, index_name: str
) -> dict[str, object]:
    if not isinstance(hit, Mapping):
        raise _invalid_response(index_name)
    source = hit.get("_source")
    if not isinstance(source, Mapping):
        raise _invalid_response(index_name)

    video_id = source.get("video_id")
    shot_id = source.get("shot_id")
    frame_id = source.get("frame_id")
    if not isinstance(video_id, str) or not video_id:
        raise _invalid_response(index_name)
    for identifier in (shot_id, frame_id):
        if (
            isinstance(identifier, bool)
            or not isinstance(identifier, int)
            or identifier < 0
        ):
            raise _invalid_response(index_name)

    raw_score = hit.get("_score")
    if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
        raise _invalid_response(index_name)
    try:
        score = float(raw_score)
    except (OverflowError, ValueError):
        raise _invalid_response(index_name) from None
    if not math.isfinite(score):
        raise _invalid_response(index_name)

    return {
        "frame_id": frame_id,
        "shot_id": shot_id,
        "video_id": video_id,
        "rank": rank,
        "score": score,
    }


def _write_results(
    output_path: Path, results: list[dict[str, object]]
) -> None:
    serialized = json.dumps(
        {"results": results},
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(serialized + "\n")
        os.replace(temporary_path, output_path)
    except (OSError, UnicodeError) as error:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise RuntimeError(f"cannot write search results '{output_path}': {error}") from error


def search_to_jsonl(
    client: object,
    index_name: str,
    query_path: Path,
    output_path: Path,
    size: int = 100,
) -> list[dict[str, object]]:
    if not isinstance(index_name, str) or not index_name.strip():
        raise ValueError("index_name must be nonblank")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError("size must be a positive integer")

    asr_text, asr_weight, ocr_text, ocr_weight, objects = _load_query(query_path)
    query = _build_query(
        asr_text,
        asr_weight,
        ocr_text,
        ocr_weight,
        objects,
    )
    if not query["bool"]:
        results: list[dict[str, object]] = []
    else:
        response = _search(client, index_name, query, size)
        results = [
            _normalize_hit(hit, rank, index_name)
            for rank, hit in enumerate(
                _response_hits(response, index_name),
                start=1,
            )
        ]
    _write_results(output_path, results)
    return results
