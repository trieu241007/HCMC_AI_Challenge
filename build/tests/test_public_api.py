from pathlib import Path

import elasticsearch_builder
from elasticsearch_builder import BuildResult, build_index, search_to_jsonl


def test_public_api_exports_supported_surface() -> None:
    assert callable(build_index)
    assert callable(search_to_jsonl)
    assert BuildResult("frames_v1", 2).document_count == 2


def test_public_api_declares_exact_dependency_contract() -> None:
    requirements_path = Path(elasticsearch_builder.__file__).with_name(
        "requirements.txt"
    )
    assert requirements_path.read_text(encoding="utf-8") == "elasticsearch>=8.17,<9\n"
