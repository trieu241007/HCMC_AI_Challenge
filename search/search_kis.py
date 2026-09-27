"""Run single KIS or two-event Multi-KIS and create a headerless CSV.

Run directly or pass output basename via CLI argument:
    python search_kis.py [output_basename]

Legacy {query_text,...} input remains supported. With {events:[E1,E2]}, E2
is optional; a missing, null, or blank E2 query_text falls back to single KIS.
Multi-KIS outputs only E1 frames, promoting E1 when a later E2 frame exists
in the same video. No maximum time or frame gap is applied.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import os
import re
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import ModuleType


# ============================================================
# Configuration: edit only these values.
# ============================================================

name_query = "KIS"
name_output = 'query-1'


def sanitize_output_basename(name: str) -> str:
    """Validate and sanitize output file basename at trust boundary."""
    name = (name or "").strip()
    if not name:
        return "test"
    clean = Path(name).name
    clean = re.sub(r'[\\/:*?"<>|\r\n\t]', "_", clean)
    if clean.lower().endswith(".csv"):
        clean = clean[:-4]
    return clean or "test"


def resolve_output_path(name: str | None = None) -> Path:
    base = sanitize_output_basename(name) if name is not None else name_output
    return Path(rf"C:\linhtinh\HCMC AI\deloy\log\{base}.csv")


INPUT_PATH = Path(rf"C:\linhtinh\HCMC AI\deloy\query\{name_query}.json")
OUTPUT_PATH = resolve_output_path(name_output)

FAISS_SCRIPT_PATH = Path(
    r"C:\linhtinh\HCMC AI\deloy\search\search_faiss.py"
)
ELASTIC_SCRIPT_PATH = Path(
    r"C:\linhtinh\HCMC AI\deloy\search\search_elastic.py"
)

ELASTICSEARCH_URL = "http://localhost:9200"
ELASTICSEARCH_INDEX = "video_frames_batch1"

RETRIEVAL_TOP_K = 200
MULTI_RETRIEVAL_TOP_K = 200
OUTPUT_TOP_K = 100

RRF_K = 20
FAISS_WEIGHT = 0.6
ELASTIC_WEIGHT = 0.4

PRINT_TOP_K = 30


SearchResult = dict[str, object]
FaissRetriever = Callable[[str, Path, Mapping[str, object], int], list[SearchResult]]
ElasticRetriever = Callable[[Path, Path], list[SearchResult]]
EventRetriever = Callable[
    [Mapping[str, object], Path, str, int],
    list[SearchResult],
]


def configure_console_utf8() -> None:
    """Prevent Vietnamese logging from failing on legacy Windows consoles."""

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="backslashreplace")


def load_python_file(path: Path, module_name: str) -> ModuleType:
    """Load one existing Python script without invoking its main function."""

    if not path.is_file():
        raise FileNotFoundError(f"Python script does not exist: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load Python script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_input_document(path: Path) -> dict[str, object]:
    """Load one UTF-8 JSON object without assuming single or Multi-KIS."""

    if not path.is_file():
        raise FileNotFoundError(f"Input JSON does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Cannot read valid UTF-8 JSON from '{path}': {error}"
        ) from error
    if not isinstance(payload, dict):
        raise ValueError("KIS input must contain one JSON object")
    return payload


def load_query(path: Path) -> dict[str, object]:
    """Load and validate one legacy single-KIS JSON object."""

    payload = load_input_document(path)
    query_text = payload.get("query_text")
    if not isinstance(query_text, str) or not query_text.strip():
        raise ValueError("KIS input field 'query_text' must be a non-empty string")
    return payload

def normalize_events(document: object) -> list[dict[str, object]]:
    """Return one or two valid event payloads with an optional E2 fallback."""

    if not isinstance(document, Mapping):
        raise ValueError("KIS input must contain one JSON object")
    if "events" not in document:
        query_text = document.get("query_text")
        if not isinstance(query_text, str) or not query_text.strip():
            raise ValueError("KIS input field 'query_text' must be non-empty")
        return [dict(document)]

    raw_events = document.get("events")
    if not isinstance(raw_events, list) or not 1 <= len(raw_events) <= 2:
        raise ValueError("Multi-KIS 'events' must contain one or two events")
    event1 = raw_events[0]
    if not isinstance(event1, Mapping):
        raise ValueError("Multi-KIS Event 1 must be a JSON object")
    event1_query = event1.get("query_text")
    if not isinstance(event1_query, str) or not event1_query.strip():
        raise ValueError("Multi-KIS Event 1 query_text must be non-empty")
    normalized = [dict(event1)]

    if len(raw_events) == 1 or raw_events[1] is None:
        return normalized
    event2 = raw_events[1]
    if not isinstance(event2, Mapping):
        raise ValueError("Multi-KIS Event 2 must be a JSON object or null")
    event2_query = event2.get("query_text")
    if event2_query is None:
        return normalized
    if not isinstance(event2_query, str):
        raise ValueError("Multi-KIS Event 2 query_text must be text or null")
    if not event2_query.strip():
        return normalized
    normalized.append(dict(event2))
    return normalized


def e2_rerank_enabled(document: Mapping[str, object]) -> bool:
    """Return the optional 0/1 E2-rank tie-break setting."""

    value = document.get("e2_rerank", 0)
    if isinstance(value, bool) or not isinstance(value, int) or value not in (0, 1):
        raise ValueError("Multi-KIS field 'e2_rerank' must be integer 0 or 1")
    return value == 1


def extract_results(document: object, source: str) -> list[SearchResult]:
    """Accept the current {results:[...]} output contract."""

    if not isinstance(document, Mapping):
        raise RuntimeError(f"{source} did not return a JSON object")
    results = document.get("results")
    if not isinstance(results, list):
        raise RuntimeError(f"{source} output does not contain a results list")
    if not all(isinstance(result, dict) for result in results):
        raise RuntimeError(f"{source} results must contain JSON objects")
    return results


def query_objects(payload: Mapping[str, object]) -> dict[str, int]:
    """Return validated positive object-count requirements."""

    value = payload.get("object", {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("KIS input field 'object' must be a JSON object")
    result: dict[str, int] = {}
    for label, count in value.items():
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"Invalid KIS object label: {label!r}")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(
                f"KIS object count for '{label}' must be a non-negative integer"
            )
        if count > 0:
            result[label.strip()] = count
    return result


def needs_elasticsearch(
    payload: Mapping[str, object],
    objects: Mapping[str, object],
) -> bool:
    """Skip Elastic when object counts are the only auxiliary signal."""

    if not objects:
        return True
    return any(
        isinstance(payload.get(field), str) and str(payload[field]).strip()
        for field in ("asr_text", "ocr_text")
    )


def run_faiss_search(
    query_text: str,
    output_path: Path,
    object_query: Mapping[str, object],
    retrieval_top_k: int = RETRIEVAL_TOP_K,
) -> list[SearchResult]:
    """Send query_text and object requirements to PE-Core FAISS."""

    if retrieval_top_k <= 0:
        raise ValueError("retrieval_top_k must be positive")
    module = load_python_file(FAISS_SCRIPT_PATH, "textual_kis_search_faiss")
    device = module.select_device()
    document = module.run_pipeline(
        vietnamese_query=query_text,
        index_path=module.INDEX_PATH,
        metadata_path=module.METADATA_PATH,
        query_npy_path=module.QUERY_NPY_PATH,
        query_metadata_path=module.QUERY_METADATA_PATH,
        output_path=output_path,
        top_k=retrieval_top_k,
        nprobe=module.NPROBE,
        print_top_k=min(PRINT_TOP_K, retrieval_top_k),
        device=device,
        object_query=object_query,
        object_data_path=module.OBJECT_DATA_PATH,
    )
    return extract_results(document, "FAISS")

def run_elastic_search(query_path: Path, output_path: Path) -> list[SearchResult]:
    """Send the complete input JSON path to the existing Elastic script."""

    module = load_python_file(
        ELASTIC_SCRIPT_PATH,
        "textual_kis_search_elastic",
    )
    client = module.Elasticsearch(ELASTICSEARCH_URL)
    try:
        results = module.search_to_jsonl(
            client=client,
            index_name=ELASTICSEARCH_INDEX,
            query_path=query_path,
            output_path=output_path,
            size=1000,
        )
    finally:
        client.close()
    if not isinstance(results, list) or not all(
        isinstance(result, dict) for result in results
    ):
        raise RuntimeError("Elasticsearch did not return a results list")
    return results


def retrieve_event_results(
    event: Mapping[str, object],
    input_path: Path,
    event_label: str,
    retrieval_top_k: int,
    *,
    faiss_retriever: FaissRetriever = run_faiss_search,
    elastic_retriever: ElasticRetriever = run_elastic_search,
) -> list[SearchResult]:
    """Run the existing KIS retrieval flow and return fused event results."""

    query_text = event.get("query_text")
    if not isinstance(query_text, str) or not query_text.strip():
        raise ValueError(f"{event_label} query_text must be non-empty")
    objects = query_objects(event)
    faiss_output_path = input_path.with_name(
        f"{input_path.stem}.{event_label}.faiss.json"
    )
    elastic_output_path = input_path.with_name(
        f"{input_path.stem}.{event_label}.elastic.json"
    )

    temporary_event_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            suffix=".json",
            delete=False,
        ) as stream:
            temporary_event_path = Path(stream.name)
            json.dump(dict(event), stream, ensure_ascii=False)

        print(f"{event_label} FAISS query_text: {query_text.strip()}")
        faiss_results = faiss_retriever(
            query_text.strip(),
            faiss_output_path,
            objects,
            retrieval_top_k,
        )
        print(f"{event_label} FAISS results: {len(faiss_results)}")

        if needs_elasticsearch(event, objects):
            elastic_results = elastic_retriever(
                temporary_event_path,
                elastic_output_path,
            )
            print(f"{event_label} Elasticsearch results: {len(elastic_results)}")
        else:
            elastic_results = []
            print(
                f"{event_label} Elasticsearch skipped: "
                "object is the only auxiliary signal"
            )
        return weighted_rrf(faiss_results, elastic_results)
    finally:
        if temporary_event_path is not None:
            temporary_event_path.unlink(missing_ok=True)

def canonical_video_id(value: object) -> str:
    """Make FAISS l28-v010 and Elastic L28_V010 refer to one video."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError("Search result video_id must be a non-empty string")
    video_id = value.strip()
    if video_id.lower().endswith(".mp4"):
        video_id = video_id[:-4]
    match = re.fullmatch(r"[lL](\d+)[-_][vV](\d+)", video_id)
    if match is not None:
        return f"L{match.group(1)}_V{match.group(2)}"
    return video_id


def rerank_event1_by_future_event(
    event1_results: Sequence[Mapping[str, object]],
    event2_results: Sequence[Mapping[str, object]],
    *,
    use_e2_rank_tiebreaker: bool = False,
) -> list[SearchResult]:
    """Promote verified E1 frames and optionally break E1-score ties by E2 rank."""

    future_by_video: dict[str, list[tuple[int, int]]] = {}
    for event2_rank, result in enumerate(event2_results, start=1):
        video_key = canonical_video_id(result.get("video_id")).casefold()
        frame_id = validate_frame_id(result.get("frame_id"), "Event 2")
        future_by_video.setdefault(video_key, []).append((frame_id, event2_rank))

    annotated: list[SearchResult] = []
    for event1_rank, source in enumerate(event1_results, start=1):
        result = dict(source)
        video_key = canonical_video_id(result.get("video_id")).casefold()
        frame_id = validate_frame_id(result.get("frame_id"), "Event 1")
        support = min(
            (
                candidate
                for candidate in future_by_video.get(video_key, [])
                if candidate[0] > frame_id
            ),
            key=(
                (lambda candidate: (candidate[1], candidate[0]))
                if use_e2_rank_tiebreaker
                else (lambda candidate: (candidate[0], candidate[1]))
            ),
            default=None,
        )
        result["original_e1_rank"] = event1_rank
        result["verified"] = support is not None
        result["support_e2_frame"] = None if support is None else support[0]
        result["support_e2_rank"] = None if support is None else support[1]
        annotated.append(result)

    def sort_key(result: Mapping[str, object]) -> tuple[object, ...]:
        verified_group = not bool(result["verified"])
        original_rank = int(result["original_e1_rank"])
        if not use_e2_rank_tiebreaker:
            return (verified_group, original_rank)

        fusion_score = result.get("fusion_score")
        if (
            isinstance(fusion_score, bool)
            or not isinstance(fusion_score, (int, float))
            or not math.isfinite(float(fusion_score))
        ):
            raise ValueError(
                "Event 1 fusion_score must be finite when e2_rerank is enabled"
            )
        support_rank = result.get("support_e2_rank")
        return (
            verified_group,
            -float(fusion_score),
            support_rank if isinstance(support_rank, int) else math.inf,
            original_rank,
        )

    annotated.sort(key=sort_key)
    for final_rank, result in enumerate(annotated, start=1):
        result["final_rank"] = final_rank
    return annotated

def validate_rank(value: object, source: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{source} result rank must be a positive integer")
    return value


def validate_frame_id(value: object, source: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{source} result frame_id must be a non-negative integer")
    return value


def best_source_ranks(
    results: Sequence[Mapping[str, object]],
    source: str,
) -> dict[tuple[str, int], tuple[int, str, int | None]]:
    """Keep the best rank for each unique (video, frame) in one source."""

    best: dict[tuple[str, int], tuple[int, str, int | None]] = {}
    for result in results:
        video_id = canonical_video_id(result.get("video_id"))
        frame_id = validate_frame_id(result.get("frame_id"), source)
        rank = validate_rank(result.get("rank"), source)
        shot_id = result.get("shot_id")
        if isinstance(shot_id, bool) or (shot_id is not None and (not isinstance(shot_id, int) or shot_id < 0)):
            shot_id = None
        key = (video_id.casefold(), frame_id)
        previous = best.get(key)
        if previous is None or rank < previous[0]:
            best[key] = (rank, video_id, shot_id)
    return best


def validate_fusion_configuration() -> None:
    if isinstance(RRF_K, bool) or not isinstance(RRF_K, int) or RRF_K < 0:
        raise ValueError("RRF_K must be a non-negative integer")
    for value, name in (
        (FAISS_WEIGHT, "FAISS_WEIGHT"),
        (ELASTIC_WEIGHT, "ELASTIC_WEIGHT"),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value < 0
        ):
            raise ValueError(f"{name} must be a non-negative finite number")
    if FAISS_WEIGHT + ELASTIC_WEIGHT <= 0:
        raise ValueError("At least one fusion weight must be positive")


def weighted_rrf(
    faiss_results: Sequence[Mapping[str, object]],
    elastic_results: Sequence[Mapping[str, object]],
) -> list[SearchResult]:
    """Fuse heterogeneous result lists using weighted reciprocal ranks."""

    validate_fusion_configuration()
    faiss_ranks = best_source_ranks(faiss_results, "FAISS")
    elastic_ranks = best_source_ranks(elastic_results, "Elasticsearch")

    candidates: dict[tuple[str, int], SearchResult] = {}
    for ranks, source_name, weight in (
        (faiss_ranks, "faiss_rank", float(FAISS_WEIGHT)),
        (elastic_ranks, "elastic_rank", float(ELASTIC_WEIGHT)),
    ):
        for key, (rank, video_id, shot_id) in ranks.items():
            candidate = candidates.setdefault(
                key,
                {
                    "video_id": video_id,
                    "frame_id": key[1],
                    "shot_id": shot_id,
                    "fusion_score": 0.0,
                    "faiss_rank": None,
                    "elastic_rank": None,
                },
            )
            if candidate.get("shot_id") is None and shot_id is not None:
                candidate["shot_id"] = shot_id
            candidate[source_name] = rank
            candidate["fusion_score"] = float(candidate["fusion_score"]) + (
                weight / (RRF_K + rank)
            )

    def sort_key(result: Mapping[str, object]) -> tuple[object, ...]:
        source_ranks = [
            rank
            for rank in (result["faiss_rank"], result["elastic_rank"])
            if isinstance(rank, int)
        ]
        best_rank = min(source_ranks) if source_ranks else math.inf
        return (
            -float(result["fusion_score"]),
            best_rank,
            str(result["video_id"]),
            int(result["frame_id"]),
        )

    return sorted(candidates.values(), key=sort_key)


def write_kis_csv_atomic(
    output_path: Path,
    results: Sequence[Mapping[str, object]],
) -> None:
    """Write headerless UTF-8 KIS CSV (strictly 2 columns: video_id, frame_id) without risking a partial output."""

    if isinstance(OUTPUT_TOP_K, bool) or not isinstance(OUTPUT_TOP_K, int):
        raise ValueError("OUTPUT_TOP_K must be an integer")
    if OUTPUT_TOP_K <= 0 or OUTPUT_TOP_K > 100:
        raise ValueError("OUTPUT_TOP_K must be between 1 and 100")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            writer = csv.writer(stream, lineterminator="\n")
            for result in results[:OUTPUT_TOP_K]:
                writer.writerow([result["video_id"], result["frame_id"]])
        os.replace(temporary_path, output_path)
    except (OSError, UnicodeError, csv.Error) as error:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise RuntimeError(f"Cannot write KIS CSV '{output_path}': {error}") from error


def write_kis_meta_jsonl_atomic(
    meta_path: Path,
    results: Sequence[Mapping[str, object]],
) -> None:
    """Write internal per-rank metadata JSONL ({rank, video_id, frame_id, shot_id})."""

    meta_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=meta_path.parent,
            prefix=f".{meta_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            for rank, result in enumerate(results[:OUTPUT_TOP_K], start=1):
                rec = {
                    "rank": rank,
                    "video_id": result["video_id"],
                    "frame_id": int(result["frame_id"]),
                    "shot_id": result.get("shot_id"),
                }
                stream.write(json.dumps(rec, ensure_ascii=False) + "\n")
        os.replace(temporary_path, meta_path)
    except Exception as error:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise RuntimeError(f"Cannot write KIS metadata JSONL '{meta_path}': {error}") from error


def print_fused_results(results: Sequence[Mapping[str, object]]) -> None:
    shown = min(PRINT_TOP_K, len(results), OUTPUT_TOP_K)
    print(f"Top {shown} fused results")
    for final_rank, result in enumerate(results[:shown], start=1):
        print(
            f"{final_rank:3d}. "
            f"video={result['video_id']} "
            f"frame={result['frame_id']} "
            f"rrf={float(result['fusion_score']):.8f} "
            f"faiss_rank={result['faiss_rank']} "
            f"elastic_rank={result['elastic_rank']}"
        )


def print_multi_results(results: Sequence[Mapping[str, object]]) -> None:
    """Print E1 results and the later E2 frame used as verification."""

    shown = min(PRINT_TOP_K, len(results), OUTPUT_TOP_K)
    print(f"Top {shown} Multi-KIS E1 results")
    for result in results[:shown]:
        marker = "verified" if result["verified"] else "unverified"
        print(
            f"{int(result['final_rank']):3d}. "
            f"video={result['video_id']} "
            f"frame={result['frame_id']} "
            f"e1_rank={result['original_e1_rank']} "
            f"status={marker} "
            f"e2_frame={result['support_e2_frame']}"
        )

def run_pipeline(
    input_path: Path,
    *,
    output_path: Path | None = None,
    faiss_retriever: FaissRetriever = run_faiss_search,
    elastic_retriever: ElasticRetriever = run_elastic_search,
    event_retriever: EventRetriever | None = None,
) -> Path:
    """Run one normal KIS or verify E1 with a later E2 in the same video."""

    configure_console_utf8()
    input_path = Path(input_path).resolve()
    document = load_input_document(input_path)
    events = normalize_events(document)
    final_output_path = (
        input_path.with_suffix(".csv")
        if output_path is None
        else Path(output_path).resolve()
    )
    if final_output_path.suffix.lower() != ".csv":
        raise ValueError("OUTPUT_PATH must use the .csv extension")

    if event_retriever is None:
        def active_event_retriever(
            event: Mapping[str, object],
            source_path: Path,
            event_label: str,
            retrieval_top_k: int,
        ) -> list[SearchResult]:
            return retrieve_event_results(
                event,
                source_path,
                event_label,
                retrieval_top_k,
                faiss_retriever=faiss_retriever,
                elastic_retriever=elastic_retriever,
            )
    else:
        active_event_retriever = event_retriever

    multi_mode = len(events) == 2
    use_e2_rank_tiebreaker = (
        e2_rerank_enabled(document) if multi_mode else False
    )
    retrieval_top_k = (
        MULTI_RETRIEVAL_TOP_K if multi_mode else RETRIEVAL_TOP_K
    )
    print(f"Input: {input_path}")
    print("Mode: two-event Multi-KIS" if multi_mode else "Mode: single KIS")
    if multi_mode:
        state = "enabled" if use_e2_rank_tiebreaker else "disabled"
        print(f"E2 rank tie-breaker: {state}")

    event1_results = active_event_retriever(
        events[0],
        input_path,
        "event1",
        retrieval_top_k,
    )
    if multi_mode:
        event2_results = active_event_retriever(
            events[1],
            input_path,
            "event2",
            retrieval_top_k,
        )
        final_results = rerank_event1_by_future_event(
            event1_results,
            event2_results,
            use_e2_rank_tiebreaker=use_e2_rank_tiebreaker,
        )
        verified_count = sum(
            bool(result["verified"]) for result in final_results
        )
        print(
            f"Verified E1 results: {verified_count}/{len(final_results)}"
        )
        print_multi_results(final_results)
    else:
        final_results = event1_results
        print_fused_results(final_results)

    write_kis_csv_atomic(final_output_path, final_results)
    meta_output_path = final_output_path.with_suffix(".meta.jsonl")
    write_kis_meta_jsonl_atomic(meta_output_path, final_results)
    print(f"Saved {min(len(final_results), OUTPUT_TOP_K)} rows: {final_output_path}")
    return final_output_path

def main() -> None:
    output = resolve_output_path(sys.argv[1]) if len(sys.argv) > 1 else OUTPUT_PATH
    run_pipeline(INPUT_PATH, output_path=output)


if __name__ == "__main__":
    main()
