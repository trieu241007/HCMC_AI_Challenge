from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

from elasticsearch.helpers import streaming_bulk
from elastic_transport import ApiResponse

from .artifacts import iter_bulk_actions, load_labels, preflight_jsonl
from .mapping import build_index_definition
from .models import BuildResult


def run_streaming_bulk(
    client: object,
    actions: Iterable[dict[str, object]],
    batch_size: int,
) -> Iterator[tuple[bool, dict[str, object]]]:
    yield from streaming_bulk(
        client,
        actions,
        chunk_size=batch_size,
        raise_on_error=False,
        raise_on_exception=True,
    )


def _call(operation: str, index_name: str, method: Any, **kwargs: object) -> object:
    try:
        return method(**kwargs)
    except Exception as error:
        raise RuntimeError(
            f"Elasticsearch {operation} failed for index '{index_name}': {error}"
        ) from error


def _bulk_error_id(item: Mapping[str, object]) -> str:
    operation = item.get("create")
    if isinstance(operation, Mapping):
        identifier = operation.get("_id")
        if isinstance(identifier, str) and identifier:
            return identifier
    return "<unknown>"


def _invalid_response(operation: str, index_name: str) -> RuntimeError:
    return RuntimeError(
        f"Elasticsearch returned an invalid {operation} response for index "
        f"'{index_name}'"
    )


def _response_body(
    response: object, operation: str, index_name: str
) -> Mapping[str, object]:
    if isinstance(response, ApiResponse):
        response = response.body
    if not isinstance(response, Mapping):
        raise _invalid_response(operation, index_name)
    return response


def _validated_shards(
    body: Mapping[str, object],
    operation: str,
    index_name: str,
    required_fields: tuple[str, ...],
) -> dict[str, int]:
    shards = body.get("_shards")
    if not isinstance(shards, Mapping):
        raise _invalid_response(operation, index_name)

    validated: dict[str, int] = {}
    for field in required_fields:
        value = shards.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise _invalid_response(operation, index_name)
        validated[field] = value
    return validated


def _validated_refresh(response: object, index_name: str) -> None:
    body = _response_body(response, "refresh", index_name)
    shards = _validated_shards(
        body,
        "refresh",
        index_name,
        ("total", "successful", "failed"),
    )
    if shards["failed"] != 0 or shards["successful"] > shards["total"]:
        raise RuntimeError(
            f"Elasticsearch refresh shard verification failed for index '{index_name}'"
        )


def _validated_count(response: object, index_name: str) -> int:
    body = _response_body(response, "count", index_name)
    shards = _validated_shards(
        body,
        "count",
        index_name,
        ("total", "successful", "skipped", "failed"),
    )
    if (
        shards["failed"] != 0
        or shards["successful"] != shards["total"]
        or shards["skipped"] > shards["successful"]
    ):
        raise RuntimeError(
            f"Elasticsearch count shard verification failed for index '{index_name}'"
        )
    count = body.get("count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise _invalid_response("count", index_name)
    return count


def build_index(
    client: object,
    jsonl_path: Path,
    label_path: Path,
    index_name: str,
    batch_size: int = 1000,
) -> BuildResult:
    if not isinstance(index_name, str) or not index_name.strip():
        raise ValueError("index_name must be nonblank")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    labels = load_labels(label_path)
    expected_count = preflight_jsonl(jsonl_path, labels)
    indices = getattr(client, "indices")
    exists = _call("exists check", index_name, indices.exists, index=index_name)
    if bool(exists):
        raise ValueError(f"Elasticsearch index '{index_name}' already exists")

    definition = build_index_definition(labels)
    created = False
    try:
        _call(
            "create",
            index_name,
            indices.create,
            index=index_name,
            settings=definition["settings"],
            mappings=definition["mappings"],
        )
        created = True
        try:
            for ok, item in run_streaming_bulk(
                client,
                iter_bulk_actions(jsonl_path, labels, index_name),
                batch_size,
            ):
                if not ok:
                    raise RuntimeError(
                        "Elasticsearch bulk create failed for document "
                        f"'{_bulk_error_id(item)}' in index '{index_name}'"
                    )
        except (ValueError, RuntimeError):
            raise
        except Exception as error:
            raise RuntimeError(
                f"Elasticsearch bulk create failed for index '{index_name}': {error}"
            ) from error

        refresh_response = _call("refresh", index_name, indices.refresh, index=index_name)
        _validated_refresh(refresh_response, index_name)
        count_response = _call(
            "count", index_name, getattr(client, "count"), index=index_name
        )
        actual_count = _validated_count(count_response, index_name)
        if actual_count != expected_count:
            raise RuntimeError(
                f"Elasticsearch count mismatch for index '{index_name}': "
                f"expected {expected_count} documents, found {actual_count}"
            )
        return BuildResult(index_name=index_name, document_count=actual_count)
    except BaseException as original_error:
        if not created:
            raise
        try:
            _call("cleanup", index_name, indices.delete, index=index_name)
        except Exception as cleanup_error:
            if isinstance(original_error, Exception):
                raise RuntimeError(
                    f"Elasticsearch build failed for index '{index_name}'; "
                    f"cleanup also failed: {cleanup_error}"
                ) from original_error
            raise original_error from cleanup_error
        except BaseException as cleanup_error:
            if isinstance(original_error, Exception):
                raise
            raise original_error from cleanup_error
        raise
