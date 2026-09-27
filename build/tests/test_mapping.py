from elasticsearch_builder.mapping import build_index_definition


def test_build_index_definition_is_strict_and_assigns_bm25() -> None:
    definition = build_index_definition(("person", "car", "stop sign"))

    assert definition["settings"] == {
        "similarity": {
            "hcmc_bm25": {"type": "BM25", "k1": 1.2, "b": 0.75}
        }
    }
    mappings = definition["mappings"]
    assert mappings["dynamic"] == "strict"
    properties = mappings["properties"]
    assert properties["video_id"] == {"type": "keyword"}
    assert properties["shot_id"] == {"type": "long"}
    assert properties["frame_id"] == {"type": "long"}
    assert properties["ocr_text"] == {
        "type": "text",
        "analyzer": "standard",
        "similarity": "hcmc_bm25",
    }
    assert properties["asr_text"] == properties["ocr_text"]
    assert properties["objects"] == {
        "type": "object",
        "dynamic": "strict",
        "properties": {
            "person": {"type": "integer"},
            "car": {"type": "integer"},
            "stop sign": {"type": "integer"},
        },
    }


def test_build_index_definition_does_not_share_mutable_field_dicts() -> None:
    definition = build_index_definition(("person",))
    properties = definition["mappings"]["properties"]
    properties["ocr_text"]["boost"] = 2
    assert "boost" not in properties["asr_text"]
