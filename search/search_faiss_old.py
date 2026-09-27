"""Translate one Vietnamese query, encode it with PE-Core, and search FAISS.

Edit the constants in the Configuration section, then run this file directly.
There is intentionally no command-line interface.

Local dependencies used by the original notebook:
    pip install einops==0.8.1 timm==1.0.15 ftfy regex huggingface_hub
    pip install --no-deps git+https://github.com/facebookresearch/perception_models.git@3e352cca660658d4b5c90f42a7808b11469e4c66
"""

from __future__ import annotations

import gc
import json
import re
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Callable, ContextManager

import faiss
import numpy as np


# ============================================================
# Configuration: edit these values, then run this file.
# ============================================================

VIETNAMESE_QUERY = "Ba người đang đi bộ xuống một con dốc trong cơn mưa, có 2 người cầm dù, trong đó người cầm dù đi sau lại mặc một chiếc áo mưa có in hình con gấu ở sau lưng."
OBJECT_QUERY: dict[str, int] = {}

TRANSLATION_MODEL_ID = "vinai/vinai-translate-vi2en-v2"
PE_MODEL_CONFIG = "PE-Core-L14-336"
PE_CHECKPOINT_PATH: Path | None = None

INDEX_PATH = Path(r"C:\linhtinh\HCMC AI\deloy\data\index.faiss")
METADATA_PATH = Path(r"C:\linhtinh\HCMC AI\deloy\data\keyframes.jsonl")
OBJECT_DATA_PATH = Path(r"C:\linhtinh\HCMC AI\deloy\data\elastic_data.jsonl")
QUERY_NPY_PATH = Path(r"C:\linhtinh\HCMC AI\deloy\data\query.npy")
QUERY_METADATA_PATH = Path(r"C:\linhtinh\HCMC AI\deloy\data\query.jsonl")
OUTPUT_PATH = Path(
    r"C:\linhtinh\HCMC AI\deloy\query\faiss_results.json"
)

TOP_K = 200
NPROBE = 128
PRINT_TOP_K = 20
NORMALIZATION_EPSILON = np.float32(1e-9)


def select_device() -> object:
    """Return a CUDA device when available, otherwise CPU."""

    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def device_type(device: object) -> str:
    value = getattr(device, "type", device)
    return str(value)


def autocast_context(device: object) -> ContextManager[object]:
    """Use FP16 computation on CUDA and FP32 computation on CPU."""

    import torch

    if device_type(device) == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def release_cuda_memory(device: object) -> None:
    """Collect unreachable models and return cached CUDA blocks."""

    gc.collect()
    if device_type(device) == "cuda":
        import torch

        torch.cuda.empty_cache()


def translate_vi_to_en(vietnamese_query: str, device: object) -> str:
    """Translate one Vietnamese sequence using VinAI's official procedure."""

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    query = vietnamese_query.strip()
    if not query:
        raise ValueError("VIETNAMESE_QUERY must not be empty")

    tokenizer = AutoTokenizer.from_pretrained(
        TRANSLATION_MODEL_ID,
        src_lang="vi_VN",
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        TRANSLATION_MODEL_ID
    ).eval().to(device)

    try:
        inputs = tokenizer(query, return_tensors="pt").to(device)
        language_codes = getattr(tokenizer, "lang_code_to_id", {})
        decoder_start_token_id = language_codes.get("en_XX")
        if decoder_start_token_id is None:
            decoder_start_token_id = tokenizer.convert_tokens_to_ids("en_XX")
        if decoder_start_token_id is None or decoder_start_token_id < 0:
            raise RuntimeError("VinAI tokenizer does not expose the en_XX token")

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                decoder_start_token_id=decoder_start_token_id,
                num_return_sequences=1,
                num_beams=5,
                early_stopping=True,
            )
        translations = tokenizer.batch_decode(
            output_ids,
            skip_special_tokens=True,
        )
        if len(translations) != 1 or not translations[0].strip():
            raise RuntimeError("VinAI returned an empty translation")
        return translations[0].strip()
    finally:
        del model
        del tokenizer
        release_cuda_memory(device)


def encode_pe_text(english_query: str, device: object) -> np.ndarray:
    """Encode one English query with PE-Core and return NumPy FP32."""

    import torch

    try:
        import core.vision_encoder.pe as pe
        import core.vision_encoder.transforms as pe_transforms
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "PE-Core is unavailable. Install the pinned perception_models "
            "package shown in this file's header."
        ) from error

    checkpoint = (
        None if PE_CHECKPOINT_PATH is None else str(PE_CHECKPOINT_PATH)
    )
    model = pe.CLIP.from_config(
        PE_MODEL_CONFIG,
        pretrained=True,
        checkpoint_path=checkpoint,
    ).eval().to(device)

    try:
        tokenizer = pe_transforms.get_text_tokenizer(model.context_length)
        tokens = tokenizer([english_query]).to(device)
        with torch.inference_mode(), autocast_context(device):
            features = model.encode_text(tokens, normalize=False)
        if not isinstance(features, torch.Tensor) or features.ndim != 2:
            raise TypeError("PE-Core did not return a two-dimensional tensor")
        return np.ascontiguousarray(
            features.detach().float().cpu().numpy(),
            dtype=np.float32,
        )
    finally:
        del model
        release_cuda_memory(device)


def normalize_query(array: np.ndarray) -> np.ndarray:
    """Validate and L2-normalize exactly one FP32 query vector."""

    query = np.asarray(array)
    if query.ndim == 1:
        query = query.reshape(1, -1)
    if query.ndim != 2 or query.shape[0] != 1 or query.shape[1] == 0:
        raise ValueError(
            "Expected exactly one non-empty query row, got "
            f"shape {query.shape}"
        )
    if not np.issubdtype(query.dtype, np.number) or np.issubdtype(
        query.dtype,
        np.complexfloating,
    ):
        raise ValueError("Query embedding must contain real numeric values")

    query = np.ascontiguousarray(query, dtype=np.float32)
    if not np.isfinite(query).all():
        raise ValueError("Query embedding contains NaN or infinity")
    norm = float(np.linalg.norm(query.astype(np.float64)))
    if not np.isfinite(norm) or norm <= float(NORMALIZATION_EPSILON):
        raise ValueError("Query embedding has zero length")
    query /= np.float32(norm)
    return np.ascontiguousarray(query, dtype=np.float32)


def load_metadata(path: Path) -> list[dict[str, object]]:
    """Load JSONL metadata whose row fields match physical line order."""

    records: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(
                    f"Metadata line {line_number} is not a JSON object"
                )
            expected_row = len(records)
            if record.get("row") != expected_row:
                raise ValueError(
                    f"Metadata line {line_number} must have row={expected_row}"
                )
            records.append(record)
    return records


def canonical_video_id(value: object) -> str:
    """Normalize keyframe and Elastic video IDs to the same lookup key."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError("video_id must be a non-empty string")
    video_id = value.strip()
    if video_id.lower().endswith(".mp4"):
        video_id = video_id[:-4]
    match = re.fullmatch(r"[lL](\d+)[-_][vV](\d+)", video_id)
    if match is not None:
        return f"L{match.group(1)}_V{match.group(2)}".casefold()
    return video_id.casefold()


def normalize_object_query(value: object) -> dict[str, int]:
    """Validate requested object counts and discard zero-count conditions."""

    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("object must be a JSON object mapping labels to counts")
    normalized: dict[str, int] = {}
    for label, count in value.items():
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"Invalid object label: {label!r}")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(
                f"Object count for '{label}' must be a non-negative integer"
            )
        if count > 0:
            normalized[label.strip()] = count
    return normalized


def rerank_by_objects(
    results: Sequence[Mapping[str, object]],
    object_query: object,
    object_data_path: Path,
) -> list[dict[str, object]]:
    """Rerank FAISS candidates by satisfied object-count conditions."""

    requested = normalize_object_query(object_query)
    copied = [dict(result) for result in results]
    if not requested:
        return copied
    if not object_data_path.is_file():
        raise FileNotFoundError(f"Object data does not exist: {object_data_path}")

    candidate_keys: set[tuple[str, int]] = set()
    for result in copied:
        frame_id = result.get("frame_id")
        if isinstance(frame_id, bool) or not isinstance(frame_id, int) or frame_id < 0:
            raise ValueError("FAISS result frame_id must be a non-negative integer")
        candidate_keys.add((canonical_video_id(result.get("video_id")), frame_id))

    object_lookup: dict[tuple[str, int], Mapping[str, object]] = {}
    with object_data_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, Mapping):
                raise ValueError(
                    f"Object data line {line_number} is not a JSON object"
                )
            frame_id = record.get("frame_id")
            if isinstance(frame_id, bool) or not isinstance(frame_id, int):
                continue
            try:
                key = (canonical_video_id(record.get("video_id")), frame_id)
            except ValueError:
                continue
            if key not in candidate_keys:
                continue
            objects = record.get("objects", {})
            if not isinstance(objects, Mapping):
                raise ValueError(
                    f"Object data line {line_number} field 'objects' is invalid"
                )
            object_lookup[key] = objects
            if len(object_lookup) == len(candidate_keys):
                break

    ranked: list[tuple[int, int, dict[str, object]]] = []
    for original_position, result in enumerate(copied):
        key = (
            canonical_video_id(result.get("video_id")),
            int(result["frame_id"]),
        )
        detected = object_lookup.get(key, {})
        bonus = 0
        for label, required_count in requested.items():
            count = detected.get(label, 0)
            if (
                not isinstance(count, bool)
                and isinstance(count, (int, float))
                and count >= required_count
            ):
                bonus += 1
        ranked.append((bonus, original_position, result))

    ranked.sort(key=lambda item: (-item[0], item[1]))
    reranked: list[dict[str, object]] = []
    for new_rank, (bonus, _, result) in enumerate(ranked, start=1):
        result["rank"] = new_rank
        result["object_bonus"] = bonus
        reranked.append(result)
    return reranked


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, document: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )


def search_index(
    index: faiss.Index,
    metadata_path: Path,
    query: np.ndarray,
    *,
    top_k: int,
    nprobe: int,
) -> dict[str, object]:
    """Search one normalized query and attach keyframe metadata."""

    if top_k <= 0:
        raise ValueError("TOP_K must be positive")
    if nprobe <= 0:
        raise ValueError("NPROBE must be positive")

    if index.metric_type != faiss.METRIC_INNER_PRODUCT:
        raise ValueError(
            "Cosine search requires a FAISS inner-product index"
        )
    if query.shape != (1, index.d):
        raise ValueError(
            f"PE query dimension {query.shape[1]} does not match FAISS "
            f"index dimension {index.d}"
        )

    metadata = load_metadata(metadata_path)
    if index.ntotal != len(metadata):
        raise ValueError(
            f"FAISS index has {index.ntotal} vectors but metadata has "
            f"{len(metadata)} rows"
        )
    if hasattr(index, "nprobe"):
        nlist = int(getattr(index, "nlist", nprobe))
        index.nprobe = min(nprobe, nlist)

    scores, ids = index.search(query, min(top_k, index.ntotal))
    results: list[dict[str, object]] = []
    for rank, (score, faiss_id) in enumerate(
        zip(scores[0], ids[0]),
        start=1,
    ):
        row = int(faiss_id)
        if row < 0:
            continue
        if row >= len(metadata):
            raise ValueError(f"FAISS returned invalid metadata row {row}")
        results.append(
            {
                **metadata[row],
                "rank": rank,
                "score": float(score),
            }
        )
    return results

def print_results(results: list[dict[str, object]], limit: int) -> None:
    if limit <= 0:
        raise ValueError("PRINT_TOP_K must be positive")
    shown = min(limit, len(results))
    print(f"Top {shown} results")
    for result in results[:shown]:
        print(
            f"{int(result['rank']):3d}. "
            f"score={float(result['score']):.6f} "
            f"video={result.get('video_id')} "
            f"frame={result.get('frame_id')} "
            f"shot={result.get('shot_id')}"
        )


def run_pipeline(
    *,
    vietnamese_query: str,
    index_path: Path,
    metadata_path: Path,
    query_npy_path: Path,
    query_metadata_path: Path,
    output_path: Path,
    top_k: int,
    nprobe: int,
    print_top_k: int,
    device: object,
    object_query: object = None,
    object_data_path: Path = OBJECT_DATA_PATH,
    translate: Callable[[str, object], str] = translate_vi_to_en,
    encode: Callable[[str, object], np.ndarray] = encode_pe_text,
) -> list[dict[str, object]]:
    """Run the approved single-query translation and retrieval flow."""

    query = vietnamese_query.strip()
    if not query:
        raise ValueError("VIETNAMESE_QUERY must not be empty")
    for required_path, label in (
        (index_path, "FAISS index"),
        (metadata_path, "metadata"),
    ):
        if not required_path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {required_path}")

    print(f"Device: {device}")
    print(f"Vietnamese query: {query}")
    english_query = translate(query, device).strip()
    if not english_query:
        raise RuntimeError("Translation is empty")
    print(f"English translation: {english_query}")

    raw_query = encode(english_query, device)
    normalized_query = normalize_query(raw_query)

    index = faiss.read_index(str(index_path))
    if normalized_query.shape[1] != index.d:
        raise ValueError(
            f"PE query dimension {normalized_query.shape[1]} does not match "
            f"FAISS index dimension {index.d}"
        )
    print(
        f"Validated query dimension {index.d} against "
        f"{index.ntotal:,} indexed vectors"
    )

    query_npy_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(query_npy_path, normalized_query, allow_pickle=False)
    query_metadata = {
        "query_row": 0,
        "vietnamese_query": query,
        "english_query": english_query,
        "translation_model": TRANSLATION_MODEL_ID,
        "embedding_model": PE_MODEL_CONFIG,
    }
    write_jsonl(query_metadata_path, [query_metadata])

    results = search_index(
        index,
        metadata_path,
        normalized_query,
        top_k=top_k,
        nprobe=nprobe,
    )
    requested_objects = normalize_object_query(object_query)
    results = rerank_by_objects(results, requested_objects, object_data_path)
    if requested_objects:
        print(
            f"Object rerank: {len(requested_objects)} conditions over "
            f"{len(results)} FAISS candidates"
        )
    output_document: dict[str, object] = {
        "results": [
            {
                "frame_id": result["frame_id"],
                "shot_id": result["shot_id"],
                "video_id": result["video_id"],
                "rank": result["rank"],
                "score": result["score"],
            }
            for result in results
        ]
    }
    write_json(output_path, output_document)
    print_results(results, print_top_k)
    print(f"Saved normalized query: {query_npy_path}")
    print(f"Saved query metadata: {query_metadata_path}")
    print(f"Saved {len(results)} search results: {output_path}")
    return output_document

def main() -> None:
    device = select_device()
    run_pipeline(
        vietnamese_query=VIETNAMESE_QUERY,
        index_path=INDEX_PATH,
        metadata_path=METADATA_PATH,
        query_npy_path=QUERY_NPY_PATH,
        query_metadata_path=QUERY_METADATA_PATH,
        output_path=OUTPUT_PATH,
        top_k=TOP_K,
        nprobe=NPROBE,
        print_top_k=PRINT_TOP_K,
        device=device,
        object_query=OBJECT_QUERY,
        object_data_path=OBJECT_DATA_PATH,
    )


if __name__ == "__main__":
    main()
