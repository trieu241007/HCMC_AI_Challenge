"""Single-file N-event TRAKE pipeline. Edit Configuration; there is no CLI.

Track A shortlists videos with PE-Core/FAISS plus optional OCR/ASR/object BM25.
Track B independently ranks the best KIS frames for every event in each video.
Only indexed keyframes are available; unindexed raw frames cannot be refined.
"""

from __future__ import annotations

import csv
import gc
import json
import math
import os
import re
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import NamedTuple

import numpy as np


# ========================= Configuration =========================
QUERY_NAME = "TRAKE"
name_output = 'query-p1-13-kis'
INPUT_PATH = Path(r"C:\linhtinh\HCMC AI\deloy\query") / f"{QUERY_NAME}.json"
OUTPUT_CSV_PATH = Path(r"C:\linhtinh\HCMC AI\deloy\submission") / f"{name_output}.csv"
DEBUG_JSON_PATH = Path(r"C:\linhtinh\HCMC AI\deloy\query") / f"{QUERY_NAME}.debug.json"
INDEX_PATH = Path(r"C:\linhtinh\HCMC AI\deloy\data\index.faiss")
EMBEDDINGS_PATH = Path(r"C:\linhtinh\HCMC AI\deloy\data\embeddings.npy")
METADATA_PATH = Path(r"C:\linhtinh\HCMC AI\deloy\data\keyframes.jsonl")
ELASTICSEARCH_URL = "http://localhost:9200"
ELASTICSEARCH_INDEX = "video_frames_batch1"
TRANSLATION_MODEL_ID = "vinai/vinai-translate-vi2en-v2"
PE_MODEL_CONFIG = "PE-Core-L14-336"
PE_CHECKPOINT_PATH: Path | None = None

EXPECTED_EMBEDDING_DIM = 1024
RETRIEVAL_TOP_K = 1000
ELASTIC_TOP_K = 1000
NPROBE = 128
VIDEO_SHORTLIST = 20
OUTPUT_VIDEO_COUNT = 10
KIS_RANKS_PER_VIDEO = 10
RRF_K = 60
PE_WEIGHT_WITH_TEXT = 0.8
TEXT_WEIGHT = 0.2
CONTEXT_WEIGHT = 0.2
MIN_EVENT_WEIGHT = 0.3
OUTPUT_TOP_K = 100
PRINT_TOP_K = 20

ROOT_FIELDS = frozenset({"query_id", "context_text", "events"})
EVENT_FIELDS = frozenset({"event_id", "query_text", "asr_text", "ocr_text", "asr_weight", "ocr_weight", "object"})
FrameKey = tuple[str, int]
Result = dict[str, object]


class AssetBundle(NamedTuple):
    index: object
    embeddings: np.ndarray
    metadata: list[dict[str, object]]
    video_rows: dict[str, np.ndarray]


def configure_console_utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        fn = getattr(stream, "reconfigure", None)
        if callable(fn):
            fn(encoding="utf-8", errors="backslashreplace")


def _positive(value: object, field: str, default: float) -> float:
    value = default if value is None else value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a positive finite number")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{field} must be a positive finite number")
    return value


def _objects(value: object, source: str) -> dict[str, int]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{source} object must be a JSON object")
    result: dict[str, int] = {}
    for label, count in value.items():
        if not isinstance(label, str) or not label.strip() or label != label.strip() or "." in label:
            raise ValueError(f"{source} object label is invalid: {label!r}")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"{source} object count for '{label}' must be a non-negative integer")
        if count:
            result[label] = count
    return result


def load_trake_query(path: Path) -> dict[str, object]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"TRAKE input JSON does not exist: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read valid UTF-8 JSON from '{path}': {error}") from error
    if not isinstance(raw, dict):
        raise ValueError("TRAKE input must be a JSON object")
    unknown = sorted(set(raw) - ROOT_FIELDS)
    if unknown:
        raise ValueError(f"Unknown TRAKE root fields: {', '.join(unknown)}")
    query_id = raw.get("query_id")
    if query_id is not None and (not isinstance(query_id, str) or not query_id.strip()):
        raise ValueError("query_id must be a non-empty string")
    context = raw.get("context_text", "")
    if not isinstance(context, str):
        raise ValueError("context_text must be a string")
    raw_events = raw.get("events")
    if not isinstance(raw_events, list) or len(raw_events) < 2:
        raise ValueError("TRAKE input must contain at least two events")
    events: list[dict[str, object]] = []
    for i, event in enumerate(raw_events, 1):
        source = f"event {i}"
        if not isinstance(event, dict):
            raise ValueError(f"{source} must be a JSON object")
        unknown = sorted(set(event) - EVENT_FIELDS)
        if unknown:
            raise ValueError(f"Unknown {source} fields: {', '.join(unknown)}")
        event_id = event.get("event_id", f"E{i}")
        query = event.get("query_text")
        asr = event.get("asr_text", "")
        ocr = event.get("ocr_text", "")
        if not isinstance(event_id, str) or not event_id.strip():
            raise ValueError(f"{source} event_id must be non-empty")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"{source} query_text must be non-empty")
        if not isinstance(asr, str) or not isinstance(ocr, str):
            raise ValueError(f"{source} ASR/OCR text must be strings")
        events.append({
            "event_id": event_id.strip(), "query_text": query.strip(),
            "asr_text": asr.strip(), "ocr_text": ocr.strip(),
            "asr_weight": _positive(event.get("asr_weight"), f"{source} asr_weight", 1.0),
            "ocr_weight": _positive(event.get("ocr_weight"), f"{source} ocr_weight", 0.4),
            "object": _objects(event.get("object"), source),
        })
    return {"query_id": query_id.strip() if isinstance(query_id, str) else None,
            "context_text": context.strip(), "events": events}


def effective_queries(payload: Mapping[str, object]) -> tuple[str | None, list[str]]:
    context = str(payload.get("context_text", "")).strip()
    events = payload.get("events")
    if not isinstance(events, list):
        raise ValueError("payload events must be a list")
    result = [str(event["query_text"]).strip() for event in events]
    return context or None, result


def canonical_video_id(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("video_id must be a non-empty string")
    value = value.strip()
    if value.casefold().endswith(".mp4"):
        value = value[:-4]
    match = re.fullmatch(r"[lL](\d+)[-_][vV](\d+)", value)
    return f"L{match.group(1)}_V{match.group(2)}" if match else value


def normalize_rows(array: np.ndarray) -> np.ndarray:
    matrix = np.asarray(array)
    if matrix.ndim != 2 or 0 in matrix.shape:
        raise ValueError(f"Expected a non-empty 2D matrix, got {matrix.shape}")
    if not np.issubdtype(matrix.dtype, np.number) or np.issubdtype(matrix.dtype, np.complexfloating):
        raise ValueError("Embedding matrix must contain real numbers")
    matrix = np.ascontiguousarray(matrix, dtype=np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError("Embedding matrix contains NaN or infinity")
    norms = np.linalg.norm(matrix.astype(np.float64), axis=1)
    if np.any(norms <= 1e-12) or not np.isfinite(norms).all():
        raise ValueError("Embedding matrix contains a zero-length row")
    matrix /= norms.astype(np.float32)[:, None]
    return np.ascontiguousarray(matrix, dtype=np.float32)


def select_device() -> object:
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _dtype(device: object) -> str:
    return str(getattr(device, "type", device))


def release_memory(device: object) -> None:
    gc.collect()
    if _dtype(device) == "cuda":
        import torch
        torch.cuda.empty_cache()


def translate_batch(texts: Sequence[str], device: object) -> list[str]:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    texts = [x.strip() for x in texts]
    if not texts or any(not x for x in texts):
        raise ValueError("Translation texts must be non-empty")
    tokenizer = AutoTokenizer.from_pretrained(TRANSLATION_MODEL_ID, src_lang="vi_VN")
    model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATION_MODEL_ID).eval().to(device)
    try:
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
        token_id = getattr(tokenizer, "lang_code_to_id", {}).get("en_XX")
        if token_id is None:
            token_id = tokenizer.convert_tokens_to_ids("en_XX")
        if token_id is None or token_id < 0:
            raise RuntimeError("VinAI tokenizer has no en_XX token")
        with torch.inference_mode():
            ids = model.generate(**inputs, decoder_start_token_id=token_id,
                                 num_beams=5, early_stopping=True)
        output = [x.strip() for x in tokenizer.batch_decode(ids, skip_special_tokens=True)]
        if len(output) != len(texts) or any(not x for x in output):
            raise RuntimeError("VinAI returned missing translations")
        return output
    finally:
        del model, tokenizer
        release_memory(device)


def encode_pe_batch(texts: Sequence[str], device: object) -> np.ndarray:
    import torch
    try:
        import core.vision_encoder.pe as pe
        import core.vision_encoder.transforms as pe_transforms
    except ModuleNotFoundError as error:
        raise RuntimeError("PE-Core is unavailable") from error
    checkpoint = None if PE_CHECKPOINT_PATH is None else str(PE_CHECKPOINT_PATH)
    model = pe.CLIP.from_config(PE_MODEL_CONFIG, pretrained=True,
                                checkpoint_path=checkpoint).eval().to(device)
    try:
        tokenizer = pe_transforms.get_text_tokenizer(model.context_length)
        tokens = tokenizer([x.strip() for x in texts]).to(device)
        autocast = torch.autocast("cuda", dtype=torch.float16) if _dtype(device) == "cuda" else nullcontext()
        with torch.inference_mode(), autocast:
            features = model.encode_text(tokens, normalize=False)
        return normalize_rows(features.detach().float().cpu().numpy())
    finally:
        del model
        release_memory(device)


def _nonnegative_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def load_assets(index_path: Path, embeddings_path: Path, metadata_path: Path) -> AssetBundle:
    import faiss
    for path, name in ((index_path, "FAISS index"), (embeddings_path, "embeddings"), (metadata_path, "metadata")):
        if not Path(path).is_file():
            raise FileNotFoundError(f"{name} does not exist: {path}")
    index = faiss.read_index(str(index_path))
    embeddings = np.load(embeddings_path, mmap_mode="r", allow_pickle=False)
    if embeddings.ndim != 2 or embeddings.shape[1] != EXPECTED_EMBEDDING_DIM:
        raise ValueError(f"Expected embedding shape (*,{EXPECTED_EMBEDDING_DIM}), got {embeddings.shape}")
    if index.metric_type != faiss.METRIC_INNER_PRODUCT or index.d != embeddings.shape[1]:
        raise ValueError("FAISS must be an inner-product index matching embedding dimension")
    metadata: list[dict[str, object]] = []
    grouped: dict[str, list[int]] = {}
    with Path(metadata_path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            row = len(metadata)
            if not isinstance(record, dict) or record.get("row") != row:
                raise ValueError(f"Metadata line {line_number} must have row={row}")
            video = canonical_video_id(record.get("video_id"))
            frame = _nonnegative_int(record.get("frame_id"), "metadata frame_id")
            shot = _nonnegative_int(record.get("shot_id"), "metadata shot_id")
            record = {**record, "video_id": video, "frame_id": frame, "shot_id": shot}
            metadata.append(record)
            grouped.setdefault(video, []).append(row)
    if index.ntotal != len(metadata) or embeddings.shape[0] != len(metadata):
        raise ValueError(f"Row mismatch: FAISS={index.ntotal}, embeddings={embeddings.shape[0]}, metadata={len(metadata)}")
    return AssetBundle(index, embeddings, metadata,
                       {k: np.asarray(v, dtype=np.int64) for k, v in grouped.items()})


def faiss_search_many(index: object, queries: np.ndarray,
                      metadata: Sequence[Mapping[str, object]], top_k: int) -> list[list[Result]]:
    queries = normalize_rows(queries)
    if queries.shape[1] != index.d or top_k <= 0:
        raise ValueError("Invalid FAISS query dimension or top_k")
    if hasattr(index, "nprobe"):
        index.nprobe = min(NPROBE, int(getattr(index, "nlist", NPROBE)))
    scores, ids = index.search(queries, min(top_k, index.ntotal))
    output: list[list[Result]] = []
    for query_scores, query_ids in zip(scores, ids):
        rows: list[Result] = []
        for rank, (score, raw_id) in enumerate(zip(query_scores, query_ids), 1):
            row = int(raw_id)
            if row < 0:
                continue
            record = metadata[row]
            rows.append({"video_id": canonical_video_id(record["video_id"]),
                         "frame_id": int(record["frame_id"]), "shot_id": int(record["shot_id"]),
                         "rank": rank, "score": float(score)})
        output.append(rows)
    return output


def event_needs_elastic(event: Mapping[str, object]) -> bool:
    objects = event.get("object", {})
    return bool(str(event.get("asr_text", "")).strip() or str(event.get("ocr_text", "")).strip()
                or (isinstance(objects, Mapping) and any(int(x) > 0 for x in objects.values())))


def search_elastic_event(client: object, event: Mapping[str, object],
                         index_name: str, size: int) -> list[Result]:
    if not event_needs_elastic(event):
        return []
    should: list[dict[str, object]] = []
    for field, weight in (("asr_text", "asr_weight"), ("ocr_text", "ocr_weight")):
        text = str(event.get(field, "")).strip()
        if text:
            should.append({"match": {field: {"query": text, "boost": float(event[weight])}}})
    boolean: dict[str, object] = {}
    if should:
        boolean.update({"should": should, "minimum_should_match": 1})
    objects = event.get("object", {})
    if isinstance(objects, Mapping) and objects:
        boolean["filter"] = [{"range": {f"objects.{k}": {"gte": int(v)}}}
                             for k, v in sorted(objects.items()) if int(v) > 0]
    try:
        response = client.search(index=index_name, query={"bool": boolean}, size=size,
                                 source_includes=["video_id", "shot_id", "frame_id"],
                                 track_total_hits=False, allow_partial_search_results=False)
    except Exception as error:
        raise RuntimeError(f"Elasticsearch search failed: {error}") from error
    body = getattr(response, "body", response)
    hits = body.get("hits", {}).get("hits") if isinstance(body, Mapping) else None
    if not isinstance(hits, list):
        raise RuntimeError("Invalid Elasticsearch response")
    output: list[Result] = []
    for rank, hit in enumerate(hits, 1):
        source = hit.get("_source", {})
        output.append({"video_id": canonical_video_id(source.get("video_id")),
                       "frame_id": _nonnegative_int(source.get("frame_id"), "Elastic frame_id"),
                       "shot_id": _nonnegative_int(source.get("shot_id"), "Elastic shot_id"),
                       "rank": rank, "score": float(hit.get("_score", 0.0))})
    return output


def _best_ranks(results: Sequence[Mapping[str, object]], source: str) -> dict[FrameKey, int]:
    best: dict[FrameKey, int] = {}
    for item in results:
        key = (canonical_video_id(item.get("video_id")), _nonnegative_int(item.get("frame_id"), f"{source} frame"))
        rank = item.get("rank")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
            raise ValueError(f"{source} rank must be positive")
        best[key] = min(rank, best.get(key, rank))
    return best


def weighted_rrf(pe_results: Sequence[Mapping[str, object]], text_results: Sequence[Mapping[str, object]],
                 pe_weight: float, text_weight: float, rrf_k: int) -> dict[FrameKey, float]:
    if any(not math.isfinite(float(x)) or x < 0 for x in (pe_weight, text_weight)) or pe_weight + text_weight <= 0:
        raise ValueError("RRF weights must be non-negative and have positive sum")
    if isinstance(rrf_k, bool) or not isinstance(rrf_k, int) or rrf_k < 0:
        raise ValueError("rrf_k must be non-negative")
    pe, text = _best_ranks(pe_results, "PE"), _best_ranks(text_results, "BM25")
    maximum = (pe_weight + text_weight) / (rrf_k + 1)
    fused: dict[FrameKey, float] = {}
    for key in pe.keys() | text.keys():
        raw = (pe_weight / (rrf_k + pe[key]) if key in pe else 0.0)
        raw += text_weight / (rrf_k + text[key]) if key in text else 0.0
        fused[key] = raw / maximum
    return fused


def _video_best(scores: Mapping[FrameKey, float]) -> dict[str, float]:
    output: dict[str, float] = {}
    for (video, _), score in scores.items():
        output[video] = max(float(score), output.get(video, 0.0))
    return output


def rank_candidate_videos(context_scores: Mapping[FrameKey, float],
                          event_frame_scores: Sequence[Mapping[FrameKey, float]],
                          lambda_context: float, beta: float, limit: int) -> list[Result]:
    if not event_frame_scores or not (0 <= lambda_context <= 1 and 0 <= beta <= 1) or limit <= 0:
        raise ValueError("Invalid video-ranking configuration")
    context = _video_best(context_scores)
    events = [_video_best(x) for x in event_frame_scores]
    videos = set(context)
    for event in events:
        videos.update(event)
    wc = lambda_context if context else 0.0
    output: list[Result] = []
    for video in videos:
        evidence = [x.get(video, 0.0) for x in events]
        event_score = (1 - beta) * sum(evidence) / len(evidence) + beta * min(evidence)
        score = wc * context.get(video, 0.0) + (1 - wc) * event_score
        output.append({"video_id": video, "video_score": float(score),
                       "context_score": float(context.get(video, 0.0)), "event_evidence": evidence})
    output.sort(key=lambda x: (-float(x["video_score"]), str(x["video_id"])))
    return output[:limit]


def rank_video_event_frames(video: str, assets: AssetBundle,
                            event_embeddings: np.ndarray, top_k: int) -> list[Result]:
    """Rank each event independently against every indexed frame in one video."""

    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    queries = normalize_rows(event_embeddings)
    rows = assets.video_rows.get(canonical_video_id(video))
    if rows is None or len(rows) < top_k:
        return []
    frames = np.asarray([assets.metadata[int(row)]["frame_id"] for row in rows], np.int64)
    order = np.argsort(frames, kind="stable")
    frames, rows = frames[order], rows[order]
    frames, first = np.unique(frames, return_index=True)
    rows = rows[first]
    if len(frames) < top_k:
        return []
    images = normalize_rows(np.asarray(assets.embeddings[rows]))
    cosine = np.asarray(queries @ images.T, dtype=np.float32)
    rankings: list[Result] = []
    for event_scores in cosine:
        best = np.lexsort((frames, -event_scores))[:top_k]
        rankings.append({
            "frame_ids": [int(frames[index]) for index in best],
            "scores": [float(event_scores[index]) for index in best],
        })
    return rankings


def rows_from_event_rankings(candidate: Mapping[str, object],
                             event_rankings: Sequence[Mapping[str, object]],
                             video_rank: int, ranks_per_video: int) -> list[Result]:
    """Transpose N event rankings into one output row per shared KIS rank."""

    if isinstance(video_rank, bool) or not isinstance(video_rank, int) or video_rank <= 0:
        raise ValueError("video_rank must be a positive integer")
    if isinstance(ranks_per_video, bool) or not isinstance(ranks_per_video, int) or ranks_per_video <= 0:
        raise ValueError("ranks_per_video must be a positive integer")
    if len(event_rankings) < 2:
        raise ValueError("At least two event rankings are required")
    normalized: list[tuple[list[object], list[object]]] = []
    for event_number, ranking in enumerate(event_rankings, 1):
        frames, scores = ranking.get("frame_ids"), ranking.get("scores")
        if not isinstance(frames, list) or not isinstance(scores, list):
            raise ValueError(f"Event {event_number} ranking must contain frame_ids and scores")
        if len(frames) < ranks_per_video or len(scores) < ranks_per_video:
            raise ValueError(f"Event {event_number} has fewer than {ranks_per_video} KIS results")
        normalized.append((frames, scores))

    video = canonical_video_id(candidate.get("video_id"))
    rows: list[Result] = []
    for rank_index in range(ranks_per_video):
        frames = [_nonnegative_int(items[0][rank_index], "KIS frame") for items in normalized]
        scores = [float(items[1][rank_index]) for items in normalized]
        if not all(math.isfinite(score) for score in scores):
            raise ValueError("KIS event scores must be finite")
        rows.append({
            "video_id": video,
            "video_rank": video_rank,
            "kis_rank": rank_index + 1,
            "frame_ids": frames,
            "event_scores": scores,
            "video_score": float(candidate["video_score"]),
            "context_score": float(candidate["context_score"]),
            "event_evidence": list(candidate["event_evidence"]),
        })
    return rows


def _validate_output(paths: Sequence[Mapping[str, object]], n: int) -> list[Result]:
    output: list[Result] = []
    for rank, item in enumerate(paths[:OUTPUT_TOP_K], 1):
        frames = [_nonnegative_int(x, f"row {rank} frame") for x in item["frame_ids"]]
        if len(frames) != n:
            raise ValueError(f"Output row {rank} must contain exactly {n} frame IDs")
        output.append({**item, "video_id": canonical_video_id(item["video_id"]), "frame_ids": frames})
    if not output:
        raise ValueError("No complete TRAKE output rows")
    return output


def _json_default(value: object) -> object:
    if isinstance(value, np.integer): return int(value)
    if isinstance(value, np.floating): return float(value)
    if isinstance(value, np.ndarray): return value.tolist()
    raise TypeError(type(value).__name__)


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temp = Path(stream.name)
            stream.write(content)
        os.replace(temp, path)
    except Exception:
        if temp is not None:
            temp.unlink(missing_ok=True)
        raise


def write_debug_json(path: Path, paths: Sequence[Mapping[str, object]]) -> None:
    content = json.dumps({"results": list(paths[:OUTPUT_TOP_K])}, ensure_ascii=False,
                         allow_nan=False, separators=(",", ":"), default=_json_default) + "\n"
    _atomic_text(Path(path), content)


def write_submission_csv(path: Path, paths: Sequence[Mapping[str, object]], event_count: int) -> None:
    validated = _validate_output(paths, event_count)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temp = Path(stream.name)
            writer = csv.writer(stream, lineterminator="\n")
            for item in validated:
                writer.writerow([item["video_id"], *item["frame_ids"]])
        os.replace(temp, path)
    except Exception:
        if temp is not None:
            temp.unlink(missing_ok=True)
        raise


def run_pipeline(input_path: Path, output_csv_path: Path, debug_json_path: Path) -> tuple[Path, Path]:
    configure_console_utf8()
    started = time.perf_counter()
    print("Mode: indexed keyframes only; raw-frame refinement is unavailable")
    payload = load_trake_query(input_path)
    events = payload["events"]
    context, event_queries = effective_queries(payload)
    assets = load_assets(INDEX_PATH, EMBEDDINGS_PATH, METADATA_PATH)
    print(f"Assets: {assets.index.ntotal:,} x {assets.index.d}; videos={len(assets.video_rows):,}")
    texts = ([context] if context else []) + event_queries
    device = select_device()
    translations = translate_batch(texts, device)
    for vi, en in zip(texts, translations): print(f"VI: {vi}\nEN: {en}")
    embeddings = encode_pe_batch(translations, device)
    if embeddings.shape[1] != EXPECTED_EMBEDDING_DIM:
        raise ValueError(f"PE dimension {embeddings.shape[1]} != {EXPECTED_EMBEDDING_DIM}")
    faiss_lists = faiss_search_many(assets.index, embeddings, assets.metadata, RETRIEVAL_TOP_K)
    offset = 1 if context else 0
    context_scores = weighted_rrf(faiss_lists[0], [], 1, 0, RRF_K) if context else {}
    elastic_lists: list[list[Result]] = [[] for _ in events]
    needs_text = [event_needs_elastic(event) for event in events]
    if any(needs_text):
        from elasticsearch import Elasticsearch
        client = Elasticsearch(ELASTICSEARCH_URL)
        try:
            for i, event in enumerate(events):
                if needs_text[i]:
                    elastic_lists[i] = search_elastic_event(client, event, ELASTICSEARCH_INDEX, ELASTIC_TOP_K)
        finally:
            client.close()
    event_frame_scores = []
    for i in range(len(events)):
        wp, wt = (PE_WEIGHT_WITH_TEXT, TEXT_WEIGHT) if needs_text[i] else (1.0, 0.0)
        event_frame_scores.append(weighted_rrf(faiss_lists[offset+i], elastic_lists[i], wp, wt, RRF_K))
    candidates = rank_candidate_videos(context_scores, event_frame_scores,
                                       CONTEXT_WEIGHT, MIN_EVENT_WEIGHT, VIDEO_SHORTLIST)
    output: list[Result] = []
    accepted_videos = 0
    event_embeddings = embeddings[offset:]
    for candidate_rank, candidate in enumerate(candidates, 1):
        video = str(candidate["video_id"])
        rankings = rank_video_event_frames(
            video, assets, event_embeddings, KIS_RANKS_PER_VIDEO
        )
        if len(rankings) != len(events):
            print(
                f"Warning: skipping candidate {candidate_rank} {video}; "
                f"fewer than {KIS_RANKS_PER_VIDEO} unique indexed frames"
            )
            continue
        accepted_videos += 1
        output.extend(rows_from_event_rankings(
            candidate, rankings, candidate_rank, KIS_RANKS_PER_VIDEO
        ))
        if accepted_videos == OUTPUT_VIDEO_COUNT:
            break
    if accepted_videos < OUTPUT_VIDEO_COUNT:
        raise RuntimeError(
            f"Only {accepted_videos} candidate videos have at least "
            f"{KIS_RANKS_PER_VIDEO} frames; need {OUTPUT_VIDEO_COUNT}"
        )
    for rank, item in enumerate(output, 1):
        item["rank"] = rank
    output = _validate_output(output, len(events))
    write_debug_json(debug_json_path, output)
    write_submission_csv(output_csv_path, output, len(events))
    for item in output[:PRINT_TOP_K]:
        print(
            f"{item['rank']:3d}. video_rank={item['video_rank']} "
            f"kis_rank={item['kis_rank']} {item['video_id']},"
            f"{','.join(map(str,item['frame_ids']))}"
        )
    print(f"Saved JSON: {debug_json_path}\nSaved CSV: {output_csv_path}\nElapsed: {time.perf_counter()-started:.2f}s")
    return Path(debug_json_path), Path(output_csv_path)


def main() -> None:
    run_pipeline(INPUT_PATH, OUTPUT_CSV_PATH, DEBUG_JSON_PATH)


if __name__ == "__main__":
    main()
