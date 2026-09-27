from __future__ import annotations


def _text_field() -> dict[str, object]:
    return {
        "type": "text",
        "analyzer": "standard",
        "similarity": "hcmc_bm25",
    }


def build_index_definition(labels: tuple[str, ...]) -> dict[str, object]:
    object_properties = {label: {"type": "integer"} for label in labels}
    return {
        "settings": {
            "similarity": {
                "hcmc_bm25": {"type": "BM25", "k1": 1.2, "b": 0.75}
            }
        },
        "mappings": {
            "dynamic": "strict",
            "properties": {
                "video_id": {"type": "keyword"},
                "shot_id": {"type": "long"},
                "frame_id": {"type": "long"},
                "ocr_text": _text_field(),
                "asr_text": _text_field(),
                "objects": {
                    "type": "object",
                    "dynamic": "strict",
                    "properties": object_properties,
                },
            },
        },
    }
