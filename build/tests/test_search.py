from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from elastic_transport import ApiResponseMeta, HttpHeaders, NodeConfig, ObjectApiResponse

import elasticsearch_builder.search as search_module
from elasticsearch_builder.search import search_to_jsonl


class FakeClient:
    def __init__(self, response: object) -> None:
        self.response = response
        self.requests: list[dict[str, object]] = []

    def search(self, **request: object) -> object:
        self.requests.append(request)
        return self.response


class FailingClient:
    def __init__(self, error: BaseException) -> None:
        self.error = error
        self.calls = 0

    def search(self, **request: object) -> object:
        del request
        self.calls += 1
        raise self.error


def write_query(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def search_response() -> dict[str, object]:
    return {
        "took": 4,
        "timed_out": False,
        "_shards": {"total": 1, "successful": 1, "skipped": 0, "failed": 0},
        "hits": {
            "total": {"value": 2, "relation": "eq"},
            "max_score": 8.72,
            "hits": [
                {
                    "_index": "video_frames_v1",
                    "_id": "l21-v001_s000010_f000000689",
                    "_score": 8.72,
                    "_source": {
                        "video_id": "l21-v001",
                        "shot_id": 10,
                        "frame_id": 689,
                    },
                },
                {
                    "_index": "video_frames_v1",
                    "_id": "l21-v002_s000004_f000000210",
                    "_score": 5.31,
                    "_source": {
                        "video_id": "l21-v002",
                        "shot_id": 4,
                        "frame_id": 210,
                    },
                },
            ],
        },
    }


def official_response(body: dict[str, object]) -> ObjectApiResponse:
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


def test_search_to_jsonl_builds_or_text_query_and_and_object_filters(
    tmp_path: Path,
) -> None:
    query_path = write_query(
        tmp_path / "query.json",
        {
            "query_text": "a man riding a motorbike",
            "asr_text": "riding through the city",
            "ocr_text": "MOTORBIKE",
            "asr_weight": 2.0,
            "ocr_weight": 0.25,
            "object": {"person": 2, "car": 1},
        },
    )
    output_path = tmp_path / "results.jsonl"
    client = FakeClient(search_response())

    results = search_to_jsonl(
        client,
        "video_frames_v1",
        query_path,
        output_path,
        size=25,
    )

    assert client.requests == [
        {
            "index": "video_frames_v1",
            "query": {
                "bool": {
                    "should": [
                        {
                            "match": {
                                "asr_text": {
                                    "query": "riding through the city",
                                    "boost": 2.0,
                                }
                            }
                        },
                        {
                            "match": {
                                "ocr_text": {
                                    "query": "MOTORBIKE",
                                    "boost": 0.25,
                                }
                            }
                        },
                    ],
                    "minimum_should_match": 1,
                    "filter": [
                        {"range": {"objects.car": {"gte": 1}}},
                        {"range": {"objects.person": {"gte": 2}}},
                    ],
                }
            },
            "size": 25,
            "source_includes": ["video_id", "shot_id", "frame_id"],
            "track_total_hits": False,
            "allow_partial_search_results": False,
        }
    ]
    assert results == [
        {
            "frame_id": 689,
            "shot_id": 10,
            "video_id": "l21-v001",
            "rank": 1,
            "score": 8.72,
        },
        {
            "frame_id": 210,
            "shot_id": 4,
            "video_id": "l21-v002",
            "rank": 2,
            "score": 5.31,
        },
    ]
    assert output_path.read_text(encoding="utf-8") == (
        '{"results":[{"frame_id":689,"shot_id":10,"video_id":"l21-v001",'
        '"rank":1,"score":8.72},{"frame_id":210,"shot_id":4,'
        '"video_id":"l21-v002","rank":2,"score":5.31}]}\n'
    )


def test_search_to_jsonl_accepts_official_response_and_object_only_query(
    tmp_path: Path,
) -> None:
    query_path = write_query(
        tmp_path / "query.json",
        {
            "query_text": "FAISS only text",
            "object": {"person": 1, "car": 0},
        },
    )
    body = search_response()
    first_hit = body["hits"]["hits"][0]  # type: ignore[index]
    first_hit["_score"] = 0.0
    body["hits"]["hits"] = [first_hit]  # type: ignore[index]
    client = FakeClient(official_response(body))
    output_path = tmp_path / "results.jsonl"

    results = search_to_jsonl(
        client,
        "video_frames_v1",
        query_path,
        output_path,
    )

    assert client.requests[0]["query"] == {
        "bool": {
            "filter": [{"range": {"objects.person": {"gte": 1}}}]
        }
    }
    assert results[0]["score"] == 0.0
    assert json.loads(output_path.read_text(encoding="utf-8")) == {
        "results": results
    }


@pytest.mark.parametrize(
    "payload",
    [
        {"query_text": "faiss", "asr_text": "", "ocr_text": "", "object": None},
        {"query_text": "faiss"},
        {
            "query_text": "faiss",
            "asr_text": "   ",
            "ocr_text": "",
            "object": {},
        },
        {
            "query_text": "faiss",
            "asr_text": "",
            "ocr_text": "",
            "object": {"person": 0},
        },
    ],
)
def test_search_to_jsonl_skips_elasticsearch_when_criteria_are_empty(
    tmp_path: Path, payload: dict[str, object]
) -> None:
    query_path = write_query(tmp_path / "query.json", payload)
    output_path = tmp_path / "results.jsonl"
    client = FailingClient(AssertionError("Elasticsearch must not be called"))

    results = search_to_jsonl(
        client,
        "video_frames_v1",
        query_path,
        output_path,
    )

    assert results == []
    assert client.calls == 0
    assert output_path.read_text(encoding="utf-8") == '{"results":[]}\n'


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ([], "query JSON must contain an object"),
        ({"unknown": 1}, "unknown query fields: unknown"),
        ({"asr_text": "hello"}, "missing query field: query_text"),
        ({"query_text": None}, "query_text must be a string"),
        ({"query_text": "faiss", "asr_text": 3}, "asr_text must be a string"),
        ({"query_text": "faiss", "ocr_text": []}, "ocr_text must be a string"),
        (
            {"query_text": "faiss", "asr_weight": True},
            "asr_weight must be a positive finite number",
        ),
        (
            {"query_text": "faiss", "asr_weight": 0},
            "asr_weight must be a positive finite number",
        ),
        (
            {"query_text": "faiss", "ocr_weight": math.inf},
            "ocr_weight must be a positive finite number",
        ),
        (
            {"query_text": "faiss", "asr_weight": 10**400},
            "asr_weight must be a positive finite number",
        ),
        (
            {"query_text": "faiss", "object": []},
            "object must be a JSON object or null",
        ),
        (
            {"query_text": "faiss", "object": {"person": True}},
            "object count for 'person' must be a non-negative integer",
        ),
        (
            {"query_text": "faiss", "object": {"person": -1}},
            "object count for 'person' must be a non-negative integer",
        ),
        (
            {"query_text": "faiss", "object": {"bad.label": 1}},
            "object label 'bad.label' must not contain",
        ),
    ],
)
def test_search_to_jsonl_rejects_invalid_query_contract(
    tmp_path: Path, payload: object, message: str
) -> None:
    query_path = write_query(tmp_path / "query.json", payload)
    output_path = tmp_path / "results.jsonl"
    client = FailingClient(AssertionError("invalid input must not reach Elasticsearch"))

    with pytest.raises(ValueError, match=message):
        search_to_jsonl(
            client,
            "video_frames_v1",
            query_path,
            output_path,
        )

    assert client.calls == 0
    assert not output_path.exists()


@pytest.mark.parametrize("size", [0, -1, True, 1.5])
def test_search_to_jsonl_rejects_invalid_size(tmp_path: Path, size: object) -> None:
    query_path = write_query(
        tmp_path / "query.json",
        {"query_text": "faiss", "asr_text": "hello"},
    )
    with pytest.raises(ValueError, match="size must be a positive integer"):
        search_to_jsonl(
            FailingClient(AssertionError("invalid size must not search")),
            "video_frames_v1",
            query_path,
            tmp_path / "results.jsonl",
            size=size,  # type: ignore[arg-type]
        )


def test_search_to_jsonl_wraps_connection_failure_without_writing_output(
    tmp_path: Path,
) -> None:
    query_path = write_query(
        tmp_path / "query.json",
        {"query_text": "faiss", "asr_text": "hello"},
    )
    output_path = tmp_path / "results.jsonl"

    with pytest.raises(
        RuntimeError,
        match="Elasticsearch search failed for index 'video_frames_v1'.*offline",
    ):
        search_to_jsonl(
            FailingClient(ConnectionError("offline")),
            "video_frames_v1",
            query_path,
            output_path,
        )

    assert not output_path.exists()


@pytest.mark.parametrize(
    "response",
    [
        {},
        {"hits": []},
        {
            "took": 1,
            "timed_out": False,
            "_shards": {"total": 1, "successful": 1, "skipped": 0, "failed": 0},
            "hits": {"total": {"value": 0, "relation": "eq"}, "hits": {}},
        },
        {
            "took": 1,
            "timed_out": True,
            "_shards": {"total": 1, "successful": 1, "skipped": 0, "failed": 0},
            "hits": {"total": {"value": 0, "relation": "eq"}, "hits": []},
        },
        {
            "took": 1,
            "timed_out": False,
            "_shards": {"total": 2, "successful": 1, "skipped": 0, "failed": 1},
            "hits": {"total": {"value": 0, "relation": "eq"}, "hits": []},
        },
        {
            "took": 1,
            "timed_out": False,
            "_shards": {"total": 2, "successful": 2, "skipped": 3, "failed": 0},
            "hits": {"total": {"value": 0, "relation": "eq"}, "hits": []},
        },
        {
            "took": 1,
            "timed_out": False,
            "_shards": {"total": 1, "successful": 1, "skipped": 0, "failed": 0},
            "hits": {
                "total": {"value": 1, "relation": "eq"},
                "hits": [
                    {
                        "_score": None,
                        "_source": {
                            "video_id": "l21-v001",
                            "shot_id": 1,
                            "frame_id": 2,
                        },
                    }
                ],
            },
        },
        {
            "took": 1,
            "timed_out": False,
            "_shards": {"total": 1, "successful": 1, "skipped": 0, "failed": 0},
            "hits": {
                "total": {"value": 1, "relation": "eq"},
                "hits": [
                    {
                        "_score": 10**400,
                        "_source": {
                            "video_id": "l21-v001",
                            "shot_id": 1,
                            "frame_id": 2,
                        },
                    }
                ],
            },
        },
        {
            "took": 1,
            "timed_out": False,
            "_shards": {"total": 1, "successful": 1, "skipped": 0, "failed": 0},
            "hits": {
                "total": {"value": 1, "relation": "eq"},
                "hits": [
                    {
                        "_score": "high",
                        "_source": {
                            "video_id": "l21-v001",
                            "shot_id": 1,
                            "frame_id": 2,
                        },
                    }
                ]
            }
        },
        {
            "took": 1,
            "timed_out": False,
            "_shards": {"total": 1, "successful": 1, "skipped": 0, "failed": 0},
            "hits": {
                "total": {"value": 1, "relation": "eq"},
                "hits": [
                    {
                        "_score": 1.0,
                        "_source": {
                            "video_id": "l21-v001",
                            "shot_id": True,
                            "frame_id": 2,
                        },
                    }
                ]
            }
        },
    ],
)
def test_search_to_jsonl_rejects_malformed_search_response(
    tmp_path: Path, response: object
) -> None:
    query_path = write_query(
        tmp_path / "query.json",
        {"query_text": "faiss", "ocr_text": "sign"},
    )
    output_path = tmp_path / "results.jsonl"

    with pytest.raises(RuntimeError, match="invalid search response"):
        search_to_jsonl(
            FakeClient(response),
            "video_frames_v1",
            query_path,
            output_path,
        )

    assert not output_path.exists()


def test_search_to_jsonl_reports_query_file_failure(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.json"
    with pytest.raises(RuntimeError, match="cannot read query file"):
        search_to_jsonl(
            FailingClient(AssertionError("missing file must not search")),
            "video_frames_v1",
            missing_path,
            tmp_path / "results.jsonl",
        )


def test_search_to_jsonl_preserves_existing_output_when_replacement_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    query_path = write_query(
        tmp_path / "query.json",
        {"query_text": "faiss", "asr_text": "hello"},
    )
    output_path = tmp_path / "results.jsonl"
    output_path.write_text('{"previous":true}\n', encoding="utf-8")

    def fail_replace(source: object, destination: object) -> None:
        del source, destination
        raise OSError("disk unavailable")

    monkeypatch.setattr(search_module.os, "replace", fail_replace)

    with pytest.raises(RuntimeError, match="cannot write search results.*disk unavailable"):
        search_to_jsonl(
            FakeClient(search_response()),
            "video_frames_v1",
            query_path,
            output_path,
        )

    assert output_path.read_text(encoding="utf-8") == '{"previous":true}\n'
    assert not any(path.suffix == ".tmp" for path in tmp_path.iterdir())
