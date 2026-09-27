from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from elastic_transport import ApiResponseMeta, HttpHeaders, NodeConfig, ObjectApiResponse

import elasticsearch_builder.builder as builder_module
from elasticsearch_builder.builder import build_index
from elasticsearch_builder.models import BuildResult


REFRESH_SHARDS = {"total": 1, "successful": 1, "failed": 0}
COUNT_SHARDS = {"total": 1, "successful": 1, "skipped": 0, "failed": 0}


def api_response(body: object) -> ObjectApiResponse:
    return ObjectApiResponse(
        body,
        ApiResponseMeta(
            200,
            "1.1",
            HttpHeaders(),
            0.01,
            NodeConfig("http", "localhost", 9200),
        ),
    )


class FakeIndices:
    def __init__(
        self,
        *,
        exists: bool = False,
        exists_error: Exception | None = None,
        create_error: Exception | None = None,
        refresh_error: Exception | None = None,
        refresh_response: object | None = None,
        delete_error: Exception | None = None,
    ):
        self.exists_result = exists
        self.exists_error = exists_error
        self.create_error = create_error
        self.refresh_error = refresh_error
        self.refresh_response = refresh_response
        self.delete_error = delete_error
        self.exists_calls: list[str] = []
        self.created: list[dict[str, object]] = []
        self.refreshed: list[str] = []
        self.deleted: list[str] = []

    def exists(self, *, index: str) -> bool:
        self.exists_calls.append(index)
        if self.exists_error is not None:
            raise self.exists_error
        return self.exists_result

    def create(self, **request: object) -> dict[str, bool]:
        self.created.append(request)
        if self.create_error is not None:
            raise self.create_error
        return {"acknowledged": True}

    def refresh(self, *, index: str) -> object:
        self.refreshed.append(index)
        if self.refresh_error is not None:
            raise self.refresh_error
        if self.refresh_response is not None:
            return self.refresh_response
        return {"_shards": REFRESH_SHARDS}

    def delete(self, *, index: str) -> dict[str, bool]:
        self.deleted.append(index)
        if self.delete_error is not None:
            raise self.delete_error
        return {"acknowledged": True}


class FakeClient:
    def __init__(
        self,
        *,
        exists: bool = False,
        count: object = 2,
        exists_error: Exception | None = None,
        create_error: Exception | None = None,
        refresh_error: Exception | None = None,
        refresh_response: object | None = None,
        count_error: Exception | None = None,
        count_shards: object = COUNT_SHARDS,
        delete_error: Exception | None = None,
    ) -> None:
        self.indices = FakeIndices(
            exists=exists,
            exists_error=exists_error,
            create_error=create_error,
            refresh_error=refresh_error,
            refresh_response=refresh_response,
            delete_error=delete_error,
        )
        self.count_value = count
        self.count_error = count_error
        self.count_shards = count_shards
        self.count_calls: list[str] = []

    def count(self, *, index: str) -> dict[str, object]:
        self.count_calls.append(index)
        if self.count_error is not None:
            raise self.count_error
        return {"count": self.count_value, "_shards": self.count_shards}


class OfficialRefreshIndices(FakeIndices):
    def refresh(self, *, index: str) -> ObjectApiResponse:
        return api_response(super().refresh(index=index))


class OfficialResponseClient(FakeClient):
    def count(self, *, index: str) -> ObjectApiResponse:
        self.count_calls.append(index)
        if self.count_error is not None:
            raise self.count_error
        return api_response(
            {"count": self.count_value, "_shards": self.count_shards}
        )


class OfficialRefreshClient(FakeClient):
    def __init__(self) -> None:
        super().__init__()
        self.indices = OfficialRefreshIndices()


def write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    labels = tmp_path / "label.json"
    labels.write_text('["person", "car"]', encoding="utf-8")
    rows = [
        {
            "video_id": "l21-v001",
            "shot_id": 1,
            "frame_id": frame,
            "ocr_text": "ocr",
            "asr_text": "asr",
            "objects": {"person": 2},
        }
        for frame in (6, 35)
    ]
    jsonl = tmp_path / "frames.jsonl"
    jsonl.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return jsonl, labels


def test_build_index_creates_bulk_loads_refreshes_and_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jsonl, labels = write_inputs(tmp_path)
    client = FakeClient(count=2)
    observed: dict[str, object] = {}

    def fake_bulk(client_arg: object, actions: object, batch_size: int):
        observed["client"] = client_arg
        observed["actions_are_iterator"] = isinstance(actions, Iterator)
        observed["actions"] = list(actions)
        observed["batch_size"] = batch_size
        yield True, {"create": {"_id": "one", "status": 201}}
        yield True, {"create": {"_id": "two", "status": 201}}

    monkeypatch.setattr(builder_module, "run_streaming_bulk", fake_bulk)

    result = build_index(client, jsonl, labels, "frames_v1", batch_size=1)

    assert result == BuildResult(index_name="frames_v1", document_count=2)
    assert observed["client"] is client
    assert observed["batch_size"] == 1
    assert observed["actions_are_iterator"] is True
    assert len(observed["actions"]) == 2
    assert client.indices.created[0]["index"] == "frames_v1"
    assert client.indices.created[0]["settings"]["similarity"]["hcmc_bm25"] == {
        "type": "BM25",
        "k1": 1.2,
        "b": 0.75,
    }
    assert client.indices.refreshed == ["frames_v1"]
    assert client.count_calls == ["frames_v1"]
    assert client.indices.deleted == []


def test_build_index_accepts_official_count_api_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jsonl, labels = write_inputs(tmp_path)
    client = OfficialResponseClient(count=2)

    def successful_bulk(client_arg: object, actions: object, batch_size: int):
        del client_arg, batch_size
        for action in actions:
            yield True, {"create": {"_id": action["_id"], "status": 201}}

    monkeypatch.setattr(builder_module, "run_streaming_bulk", successful_bulk)

    result = build_index(client, jsonl, labels, "frames_v1")

    assert result == BuildResult(index_name="frames_v1", document_count=2)
    assert client.indices.deleted == []


def test_build_index_accepts_official_refresh_api_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jsonl, labels = write_inputs(tmp_path)
    client = OfficialRefreshClient()

    def successful_bulk(client_arg: object, actions: object, batch_size: int):
        del client_arg, batch_size
        for action in actions:
            yield True, {"create": {"_id": action["_id"], "status": 201}}

    monkeypatch.setattr(builder_module, "run_streaming_bulk", successful_bulk)

    result = build_index(client, jsonl, labels, "frames_v1")

    assert result == BuildResult(index_name="frames_v1", document_count=2)
    assert client.indices.deleted == []


@pytest.mark.parametrize("batch_size", [0, -1, True])
def test_build_index_rejects_invalid_batch_size(
    tmp_path: Path, batch_size: object
) -> None:
    jsonl, labels = write_inputs(tmp_path)
    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        build_index(FakeClient(), jsonl, labels, "frames_v1", batch_size=batch_size)


def test_build_index_rejects_blank_index_name(tmp_path: Path) -> None:
    jsonl, labels = write_inputs(tmp_path)
    with pytest.raises(ValueError, match="index_name must be nonblank"):
        build_index(FakeClient(), jsonl, labels, "  ")


def test_build_index_refuses_existing_target_without_deleting(
    tmp_path: Path,
) -> None:
    jsonl, labels = write_inputs(tmp_path)
    client = FakeClient(exists=True)
    with pytest.raises(ValueError, match="index 'frames_v1' already exists"):
        build_index(client, jsonl, labels, "frames_v1")
    assert client.indices.created == []
    assert client.indices.deleted == []


def test_build_index_wraps_exists_failure_without_deleting(tmp_path: Path) -> None:
    jsonl, labels = write_inputs(tmp_path)
    client = FakeClient(exists_error=ConnectionError("cluster unavailable"))
    with pytest.raises(RuntimeError, match="exists check failed.*cluster unavailable"):
        build_index(client, jsonl, labels, "frames_v1")
    assert client.indices.created == []
    assert client.indices.deleted == []


def test_build_index_wraps_create_failure_without_claiming_cleanup(
    tmp_path: Path,
) -> None:
    jsonl, labels = write_inputs(tmp_path)
    client = FakeClient(create_error=ConnectionError("create unavailable"))
    with pytest.raises(RuntimeError, match="create failed.*create unavailable"):
        build_index(client, jsonl, labels, "frames_v1")
    assert client.indices.deleted == []


def test_build_index_deletes_only_new_target_after_bulk_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jsonl, labels = write_inputs(tmp_path)
    client = FakeClient()

    def failed_bulk(client_arg: object, actions: object, batch_size: int):
        del client_arg, actions, batch_size
        yield False, {"create": {"_id": "bad-id", "status": 400, "error": "bad"}}

    monkeypatch.setattr(builder_module, "run_streaming_bulk", failed_bulk)
    with pytest.raises(RuntimeError, match="bulk create failed.*bad-id"):
        build_index(client, jsonl, labels, "frames_v1")
    assert client.indices.deleted == ["frames_v1"]


def test_build_index_cleans_up_count_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jsonl, labels = write_inputs(tmp_path)
    client = FakeClient(count=1)

    def successful_bulk(client_arg: object, actions: object, batch_size: int):
        del client_arg, batch_size
        for action in actions:
            yield True, {"create": {"_id": action["_id"], "status": 201}}

    monkeypatch.setattr(builder_module, "run_streaming_bulk", successful_bulk)
    with pytest.raises(RuntimeError, match="expected 2 documents, found 1"):
        build_index(client, jsonl, labels, "frames_v1")
    assert client.indices.deleted == ["frames_v1"]


@pytest.mark.parametrize(
    ("client", "message"),
    [
        (
            FakeClient(refresh_error=ConnectionError("refresh unavailable")),
            "refresh failed.*refresh unavailable",
        ),
        (
            FakeClient(count_error=ConnectionError("count unavailable")),
            "count failed.*count unavailable",
        ),
    ],
)
def test_build_index_cleans_up_refresh_and_count_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    client: FakeClient,
    message: str,
) -> None:
    jsonl, labels = write_inputs(tmp_path)

    def successful_bulk(client_arg: object, actions: object, batch_size: int):
        del client_arg, batch_size
        for action in actions:
            yield True, {"create": {"_id": action["_id"], "status": 201}}

    monkeypatch.setattr(builder_module, "run_streaming_bulk", successful_bulk)
    with pytest.raises(RuntimeError, match=message):
        build_index(client, jsonl, labels, "frames_v1")
    assert client.indices.deleted == ["frames_v1"]


def test_build_index_reports_original_and_cleanup_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jsonl, labels = write_inputs(tmp_path)
    client = FakeClient(delete_error=RuntimeError("delete unavailable"))

    def failed_bulk(client_arg: object, actions: object, batch_size: int):
        del client_arg, actions, batch_size
        raise ConnectionError("bulk unavailable")
        yield

    monkeypatch.setattr(builder_module, "run_streaming_bulk", failed_bulk)
    with pytest.raises(RuntimeError, match="cleanup also failed.*delete unavailable") as caught:
        build_index(client, jsonl, labels, "frames_v1")
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert "bulk" in str(caught.value.__cause__).lower()


def test_build_index_rejects_invalid_count_response_and_cleans_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jsonl, labels = write_inputs(tmp_path)
    client = FakeClient(count=True)

    def successful_bulk(client_arg: object, actions: object, batch_size: int):
        del client_arg, batch_size
        for action in actions:
            yield True, {"create": {"_id": action["_id"], "status": 201}}

    monkeypatch.setattr(builder_module, "run_streaming_bulk", successful_bulk)
    with pytest.raises(RuntimeError, match="invalid count response"):
        build_index(client, jsonl, labels, "frames_v1")
    assert client.indices.deleted == ["frames_v1"]


def test_run_streaming_bulk_forwards_original_iterator_and_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = object()
    actions = iter([{"_id": "one"}])
    observed: dict[str, object] = {}

    def fake_streaming_bulk(
        client_arg: object, actions_arg: object, **kwargs: object
    ) -> Iterator[tuple[bool, dict[str, object]]]:
        observed["client"] = client_arg
        observed["actions"] = actions_arg
        observed["kwargs"] = kwargs
        yield True, {"create": {"_id": "one", "status": 201}}

    monkeypatch.setattr(builder_module, "streaming_bulk", fake_streaming_bulk)

    result = list(builder_module.run_streaming_bulk(client, actions, batch_size=7))

    assert result == [(True, {"create": {"_id": "one", "status": 201}})]
    assert observed["client"] is client
    assert observed["actions"] is actions
    assert observed["kwargs"] == {
        "chunk_size": 7,
        "raise_on_error": False,
        "raise_on_exception": True,
    }


def test_build_index_preflight_rejects_invalid_jsonl_before_elasticsearch(
    tmp_path: Path,
) -> None:
    jsonl, labels = write_inputs(tmp_path)
    jsonl.write_text("{\n", encoding="utf-8")
    client = FakeClient()

    with pytest.raises(ValueError, match="invalid JSON"):
        build_index(client, jsonl, labels, "frames_v1")

    assert client.indices.exists_calls == []
    assert client.indices.created == []
    assert client.indices.refreshed == []
    assert client.count_calls == []
    assert client.indices.deleted == []


def test_build_index_cleans_up_failed_refresh_shards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jsonl, labels = write_inputs(tmp_path)
    client = FakeClient(
        refresh_response={"_shards": {"total": 1, "successful": 0, "failed": 1}}
    )

    def successful_bulk(client_arg: object, actions: object, batch_size: int):
        del client_arg, batch_size
        for action in actions:
            yield True, {"create": {"_id": action["_id"], "status": 201}}

    monkeypatch.setattr(builder_module, "run_streaming_bulk", successful_bulk)

    with pytest.raises(RuntimeError, match="refresh shard verification failed"):
        build_index(client, jsonl, labels, "frames_v1")

    assert client.indices.deleted == ["frames_v1"]


def test_build_index_accepts_refresh_with_unassigned_replica(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jsonl, labels = write_inputs(tmp_path)
    client = FakeClient(
        refresh_response={"_shards": {"total": 2, "successful": 1, "failed": 0}}
    )

    def successful_bulk(client_arg: object, actions: object, batch_size: int):
        del client_arg, batch_size
        for action in actions:
            yield True, {"create": {"_id": action["_id"], "status": 201}}

    monkeypatch.setattr(builder_module, "run_streaming_bulk", successful_bulk)

    result = build_index(client, jsonl, labels, "frames_v1")

    assert result == BuildResult(index_name="frames_v1", document_count=2)
    assert client.indices.deleted == []


def test_build_index_accepts_count_with_skipped_shards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jsonl, labels = write_inputs(tmp_path)
    client = FakeClient(
        count_shards={"total": 2, "successful": 2, "skipped": 1, "failed": 0}
    )

    def successful_bulk(client_arg: object, actions: object, batch_size: int):
        del client_arg, batch_size
        for action in actions:
            yield True, {"create": {"_id": action["_id"], "status": 201}}

    monkeypatch.setattr(builder_module, "run_streaming_bulk", successful_bulk)

    result = build_index(client, jsonl, labels, "frames_v1")

    assert result == BuildResult(index_name="frames_v1", document_count=2)
    assert client.indices.deleted == []


@pytest.mark.parametrize(
    "count_shards",
    [
        {"total": 1, "successful": 0, "skipped": 0, "failed": 1},
        {"total": 2, "successful": 1, "skipped": 0, "failed": 0},
        {"total": 2, "successful": 2, "skipped": 3, "failed": 0},
    ],
    ids=["failed", "successful-less-than-total", "skipped-exceeds-successful"],
)
def test_build_index_cleans_up_invalid_count_shard_accounting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, count_shards: object
) -> None:
    jsonl, labels = write_inputs(tmp_path)
    client = FakeClient(count_shards=count_shards)

    def successful_bulk(client_arg: object, actions: object, batch_size: int):
        del client_arg, batch_size
        for action in actions:
            yield True, {"create": {"_id": action["_id"], "status": 201}}

    monkeypatch.setattr(builder_module, "run_streaming_bulk", successful_bulk)

    with pytest.raises(RuntimeError, match="count shard verification failed"):
        build_index(client, jsonl, labels, "frames_v1")

    assert client.indices.deleted == ["frames_v1"]


@pytest.mark.parametrize(
    ("client", "message"),
    [
        (
            FakeClient(refresh_response={"_shards": {"total": 1, "failed": 0}}),
            "invalid refresh response",
        ),
        (
            FakeClient(
                count_shards={
                    "total": 1,
                    "successful": True,
                    "skipped": 0,
                    "failed": 0,
                }
            ),
            "invalid count response",
        ),
    ],
    ids=["refresh-missing-successful", "count-bool-successful"],
)
def test_build_index_rejects_malformed_shard_metadata_and_cleans_up(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    client: FakeClient,
    message: str,
) -> None:
    jsonl, labels = write_inputs(tmp_path)

    def successful_bulk(client_arg: object, actions: object, batch_size: int):
        del client_arg, batch_size
        for action in actions:
            yield True, {"create": {"_id": action["_id"], "status": 201}}

    monkeypatch.setattr(builder_module, "run_streaming_bulk", successful_bulk)

    with pytest.raises(RuntimeError, match=message):
        build_index(client, jsonl, labels, "frames_v1")

    assert client.indices.deleted == ["frames_v1"]


def test_build_index_cleans_up_and_reraises_bulk_keyboard_interrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jsonl, labels = write_inputs(tmp_path)
    client = FakeClient()

    def interrupted_bulk(client_arg: object, actions: object, batch_size: int):
        del client_arg, actions, batch_size
        raise KeyboardInterrupt("stop")
        yield

    monkeypatch.setattr(builder_module, "run_streaming_bulk", interrupted_bulk)

    with pytest.raises(KeyboardInterrupt, match="stop"):
        build_index(client, jsonl, labels, "frames_v1")

    assert client.indices.deleted == ["frames_v1"]


def test_build_index_preserves_keyboard_interrupt_when_cleanup_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jsonl, labels = write_inputs(tmp_path)
    client = FakeClient(delete_error=RuntimeError("delete unavailable"))

    def interrupted_bulk(client_arg: object, actions: object, batch_size: int):
        del client_arg, actions, batch_size
        raise KeyboardInterrupt("stop")
        yield

    monkeypatch.setattr(builder_module, "run_streaming_bulk", interrupted_bulk)

    with pytest.raises(KeyboardInterrupt, match="stop") as caught:
        build_index(client, jsonl, labels, "frames_v1")

    assert isinstance(caught.value.__cause__, RuntimeError)
    assert "delete unavailable" in str(caught.value.__cause__)
