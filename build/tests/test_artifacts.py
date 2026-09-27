from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from elasticsearch_builder.artifacts import (
    iter_bulk_actions,
    iter_records,
    load_labels,
    preflight_jsonl,
)
from elasticsearch_builder.models import BuildResult, FrameRecord


def write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def test_frame_record_exposes_canonical_id_and_six_field_source() -> None:
    record = FrameRecord(
        video_id="l21-v001",
        shot_id=10,
        frame_id=689,
        ocr_text="OCR raw",
        asr_text="ASR raw",
        objects=(("car", 1), ("person", 2)),
    )

    assert record.keyframe_id == "l21-v001_s000010_f000000689"
    assert record.to_source() == {
        "video_id": "l21-v001",
        "shot_id": 10,
        "frame_id": 689,
        "ocr_text": "OCR raw",
        "asr_text": "ASR raw",
        "objects": {"car": 1, "person": 2},
    }
    assert BuildResult("frames_v1", 1).document_count == 1


def test_load_labels_preserves_authoritative_order(tmp_path: Path) -> None:
    path = write_json(tmp_path / "label.json", ["person", "car", "stop sign"])
    assert load_labels(path) == ("person", "car", "stop sign")


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ([], "non-empty JSON array"),
        ({"person": 1}, "non-empty JSON array"),
        (["person", "person"], "duplicate label 'person'"),
        (["person", ""], "label at index 1 must be nonblank"),
        (["person", 7], "label at index 1 must be a string"),
        (["person", " car "], "must not have surrounding whitespace"),
        (["person", "bad.label"], "must not contain '.'"),
    ],
)
def test_load_labels_rejects_invalid_contract(
    tmp_path: Path, payload: object, message: str
) -> None:
    path = write_json(tmp_path / "label.json", payload)
    with pytest.raises(ValueError, match=message.replace(".", r"\.")):
        load_labels(path)


def test_load_labels_reports_malformed_json(tmp_path: Path) -> None:
    path = tmp_path / "label.json"
    path.write_text("[", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON in label file"):
        load_labels(path)


def test_load_labels_reports_filesystem_failure(tmp_path: Path) -> None:
    path = tmp_path / "missing.json"
    with pytest.raises(RuntimeError, match="cannot read label file"):
        load_labels(path)


def write_jsonl(path: Path, rows: list[object]) -> Path:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def valid_row() -> dict[str, object]:
    return {
        "video_id": "l21-v001",
        "shot_id": "10",
        "frame_id": "689",
        "ocr_text": " OCR raw ",
        "asr_text": "ASR raw",
        "objects": {"person": 2, "car": 1, "stop sign": 0},
    }


def test_iter_records_normalizes_ids_but_preserves_text(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path / "frames.jsonl", [valid_row()])
    records = list(iter_records(path, ("person", "car", "stop sign")))

    assert records == [
        FrameRecord(
            video_id="l21-v001",
            shot_id=10,
            frame_id=689,
            ocr_text=" OCR raw ",
            asr_text="ASR raw",
            objects=(("car", 1), ("person", 2), ("stop sign", 0)),
        )
    ]


def test_preflight_and_actions_use_create_with_deterministic_id(tmp_path: Path) -> None:
    second = valid_row()
    second["frame_id"] = 690
    path = write_jsonl(tmp_path / "frames.jsonl", [valid_row(), second])
    labels = ("person", "car", "stop sign")

    assert preflight_jsonl(path, labels) == 2
    actions = list(iter_bulk_actions(path, labels, "frames_v1"))
    assert [action["_id"] for action in actions] == [
        "l21-v001_s000010_f000000689",
        "l21-v001_s000010_f000000690",
    ]
    assert actions[0] == {
        "_op_type": "create",
        "_index": "frames_v1",
        "_id": "l21-v001_s000010_f000000689",
        "_source": {
            "video_id": "l21-v001",
            "shot_id": 10,
            "frame_id": 689,
            "ocr_text": " OCR raw ",
            "asr_text": "ASR raw",
            "objects": {"car": 1, "person": 2, "stop sign": 0},
        },
    }


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda row: row.pop("asr_text"), "missing fields: asr_text"),
        (lambda row: row.__setitem__("extra", 1), "unknown fields: extra"),
        (lambda row: row.__setitem__("video_id", "  "), "video_id must be nonblank"),
        (lambda row: row.__setitem__("video_id", 1), "video_id must be a string"),
        (lambda row: row.__setitem__("shot_id", True), "shot_id must be"),
        (lambda row: row.__setitem__("shot_id", -1), "shot_id must be"),
        (lambda row: row.__setitem__("shot_id", 1.5), "shot_id must be"),
        (lambda row: row.__setitem__("frame_id", "1.5"), "frame_id must be"),
        (lambda row: row.__setitem__("ocr_text", None), "ocr_text must be a string"),
        (lambda row: row.__setitem__("asr_text", 3), "asr_text must be a string"),
        (lambda row: row.__setitem__("objects", []), "objects must be an object"),
        (lambda row: row.__setitem__("objects", {"unknown": 1}), "unknown object label 'unknown'"),
        (lambda row: row.__setitem__("objects", {"person": True}), "object count for 'person' must be a non-negative integer"),
        (lambda row: row.__setitem__("objects", {"person": -1}), "object count for 'person' must be a non-negative integer"),
        (lambda row: row.__setitem__("objects", {"person": 1.5}), "object count for 'person' must be a non-negative integer"),
    ],
)
def test_iter_records_rejects_invalid_fields(tmp_path: Path, mutate: object, message: str) -> None:
    row = valid_row()
    mutate(row)  # type: ignore[operator]
    path = write_jsonl(tmp_path / "frames.jsonl", [row])
    with pytest.raises(ValueError, match=message):
        list(iter_records(path, ("person", "car", "stop sign")))


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("", "must contain at least one record"),
        ("\n", "blank line 1"),
        ("[1, 2]\n", "line must contain a JSON object.*line 1"),
        ("{\n", "invalid JSON at line 1"),
    ],
)
def test_iter_records_rejects_invalid_jsonl(tmp_path: Path, content: str, message: str) -> None:
    path = tmp_path / "frames.jsonl"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        list(iter_records(path, ("person", "car", "stop sign")))


def test_preflight_rejects_duplicate_identity(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path / "frames.jsonl", [valid_row(), valid_row()])
    with pytest.raises(ValueError, match="duplicate frame identity.*line 2"):
        preflight_jsonl(path, ("person", "car", "stop sign"))


def test_iter_records_reports_filesystem_failure(tmp_path: Path) -> None:
    path = tmp_path / "missing.jsonl"
    with pytest.raises(RuntimeError, match="cannot read JSONL"):
        list(iter_records(path, ("person",)))


def test_iter_records_allows_empty_text_and_objects(tmp_path: Path) -> None:
    row = valid_row()
    row["ocr_text"] = ""
    row["asr_text"] = ""
    row["objects"] = {}
    path = write_jsonl(tmp_path / "frames.jsonl", [row])

    assert list(iter_records(path, ("person", "car", "stop sign"))) == [
        FrameRecord(
            video_id="l21-v001",
            shot_id=10,
            frame_id=689,
            ocr_text="",
            asr_text="",
            objects=(),
        )
    ]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("shot_id", 2**63 - 1),
        ("frame_id", 2**63 - 1),
        ("shot_id", "9223372036854775807"),
        ("frame_id", "0009223372036854775807"),
    ],
)
def test_iter_records_accepts_long_maximum(tmp_path: Path, field: str, value: object) -> None:
    row = valid_row()
    row[field] = value
    path = write_jsonl(tmp_path / "frames.jsonl", [row])

    records = list(iter_records(path, ("person", "car", "stop sign")))
    assert getattr(records[0], field) == 2**63 - 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("shot_id", 2**63),
        ("frame_id", 2**63),
        ("shot_id", "9223372036854775808"),
        ("frame_id", "9" * 10000),
    ],
)
def test_iter_records_rejects_identifier_over_long_maximum(
    tmp_path: Path, field: str, value: object
) -> None:
    row = valid_row()
    row[field] = value
    path = write_jsonl(tmp_path / "frames.jsonl", [row])

    with pytest.raises(ValueError, match=f"{field} must be"):
        list(iter_records(path, ("person", "car", "stop sign")))


@pytest.mark.parametrize("count", [2**31 - 1])
def test_iter_records_accepts_integer_maximum(tmp_path: Path, count: int) -> None:
    row = valid_row()
    row["objects"] = {"person": count}
    path = write_jsonl(tmp_path / "frames.jsonl", [row])

    assert list(iter_records(path, ("person", "car", "stop sign")))[0].objects == (
        ("person", count),
    )


@pytest.mark.parametrize("count", [2**31])
def test_iter_records_rejects_object_count_over_integer_maximum(
    tmp_path: Path, count: int
) -> None:
    row = valid_row()
    row["objects"] = {"person": count}
    path = write_jsonl(tmp_path / "frames.jsonl", [row])

    with pytest.raises(
        ValueError,
        match="object count for 'person' must be a non-negative integer",
    ):
        list(iter_records(path, ("person", "car", "stop sign")))


def test_iter_records_rejects_frame_identity_over_512_utf8_bytes(
    tmp_path: Path,
) -> None:
    row = valid_row()
    row["video_id"] = "é" * 256
    path = write_jsonl(tmp_path / "frames.jsonl", [row])

    with pytest.raises(ValueError, match="frame identity.*512 UTF-8 bytes"):
        list(iter_records(path, ("person", "car", "stop sign")))


def test_iter_records_accepts_frame_identity_at_512_utf8_bytes(tmp_path: Path) -> None:
    row = valid_row()
    row["video_id"] = "é" * 246 + "a"
    path = write_jsonl(tmp_path / "frames.jsonl", [row])

    record = list(iter_records(path, ("person", "car", "stop sign")))[0]
    assert len(record.keyframe_id.encode("utf-8")) == 512


@pytest.mark.parametrize(
    "content",
    [
        '{"video_id":"l21-v001","video_id":"other","shot_id":10,"frame_id":689,"ocr_text":"","asr_text":"","objects":{}}\n',
        '{"video_id":"l21-v001","shot_id":10,"frame_id":689,"ocr_text":"","asr_text":"","objects":{"person":1,"person":2}}\n',
    ],
)
def test_iter_records_rejects_duplicate_json_member_names(
    tmp_path: Path, content: str
) -> None:
    path = tmp_path / "frames.jsonl"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate JSON member"):
        list(iter_records(path, ("person", "car", "stop sign")))


def test_iter_records_reports_oversized_json_numeric_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    huge_digits = "9" * 10000
    path = tmp_path / "frames.jsonl"
    path.write_text(
        '{"video_id":"l21-v001","shot_id":'
        + huge_digits
        + ',"frame_id":689,"ocr_text":"","asr_text":"","objects":{}}\n',
        encoding="utf-8",
    )

    setter = getattr(sys, "set_int_max_str_digits", None)
    getter = getattr(sys, "get_int_max_str_digits", None)
    if setter is not None and getter is not None:
        previous_limit = getter()
        setter(640)
        try:
            with pytest.raises(ValueError, match="invalid JSON at line 1") as error:
                list(iter_records(path, ("person", "car", "stop sign")))
        finally:
            setter(previous_limit)
    else:
        real_loads = json.loads

        def loads_with_digit_limit(raw: str, *args: object, **kwargs: object) -> object:
            if huge_digits in raw:
                raise ValueError("Exceeds the limit for integer string conversion")
            return real_loads(raw, *args, **kwargs)

        monkeypatch.setattr(json, "loads", loads_with_digit_limit)
        with pytest.raises(ValueError, match="invalid JSON at line 1") as error:
            list(iter_records(path, ("person", "car", "stop sign")))

    assert f"{path}: invalid JSON at line 1" in str(error.value)
