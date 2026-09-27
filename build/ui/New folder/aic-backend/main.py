import os
import sys
import subprocess
import json
import tempfile
import asyncio
import time
import uuid
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator

# In-process RAM cache for sidecar JSON with per-path async lock, stat-before/stat-after validation and LRU 32 capacity.
from collections import OrderedDict

sidecar_ram_cache: OrderedDict[str, tuple[int, int, dict]] = OrderedDict()
sidecar_locks: dict[str, asyncio.Lock] = {}
SIDECAR_LRU_MAX = 32

def _read_and_parse_sidecar(sidecar_path: Path, cache_key: str) -> dict | None:
    if not sidecar_path.is_file():
        return None
    try:
        stat_before = sidecar_path.stat()
        cached = sidecar_ram_cache.get(cache_key)
        if cached is not None:
            mtime_ns, size, payload = cached
            if mtime_ns == stat_before.st_mtime_ns and size == stat_before.st_size:
                sidecar_ram_cache.move_to_end(cache_key)
                return payload

        for attempt in range(2):
            try:
                stat_b = sidecar_path.stat()
                content = sidecar_path.read_text(encoding="utf-8")
                payload = json.loads(content)
                stat_a = sidecar_path.stat()
                if stat_b.st_mtime_ns == stat_a.st_mtime_ns and stat_b.st_size == stat_a.st_size and isinstance(payload, dict):
                    sidecar_ram_cache[cache_key] = (stat_a.st_mtime_ns, stat_a.st_size, payload)
                    sidecar_ram_cache.move_to_end(cache_key)
                    while len(sidecar_ram_cache) > SIDECAR_LRU_MAX:
                        sidecar_ram_cache.popitem(last=False)
                    return payload
            except (OSError, json.JSONDecodeError):
                if attempt == 1:
                    raise
                time.sleep(0.01)
    except Exception as err:
        print(f"Error reading/caching sidecar {sidecar_path}: {err}")
    return None

def _is_valid_neighbor_item(x: object) -> bool:
    if not isinstance(x, dict):
        return False
    v = x.get("video_id")
    f = x.get("frame_id")
    if not isinstance(v, str) or not v.strip():
        return False
    if isinstance(f, bool) or not isinstance(f, int) or f < 0:
        return False
    return True

async def get_cached_sidecar_payload(sidecar_path: Path) -> dict | None:
    cache_key = str(sidecar_path)
    lock = sidecar_locks.setdefault(cache_key, asyncio.Lock())
    async with lock:
        return await asyncio.to_thread(_read_and_parse_sidecar, sidecar_path, cache_key)

# Lock đồng bộ in-process cho thao tác ghi file query/KIS.json và chạy subprocess
kis_lock = asyncio.Lock()

# Khởi tạo ứng dụng FastAPI
app = FastAPI(title="AIC 2026 - Video Retrieval API")

# Cấu trúc EventInput nhận từ UI
class EventInput(BaseModel):
    query_text: str = ""
    ocr_text: str = ""
    asr_text: str = ""
    object: str | dict = {}

    @field_validator("query_text", "ocr_text", "asr_text", mode="before")
    @classmethod
    def string_must_be_str(cls, v):
        if v is None:
            return ""
        return str(v)

    @field_validator("object", mode="before")
    @classmethod
    def parse_and_validate_object(cls, v):
        if isinstance(v, str):
            v_str = v.strip()
            if not v_str:
                return {}
            try:
                v = json.loads(v_str)
            except Exception:
                raise ValueError("object string must be valid JSON object")
        if not isinstance(v, dict):
            raise ValueError("object must be a JSON object mapping label to non-negative integer")
        
        parsed_dict = {}
        for key, val in v.items():
            try:
                int_val = int(val)
            except (ValueError, TypeError):
                raise ValueError(f"object count for '{key}' must be an integer")
            if int_val < 0:
                raise ValueError(f"object count for '{key}' must be non-negative")
            parsed_dict[str(key)] = int_val
        return parsed_dict

class SubmitItem(BaseModel):
    video_id: str
    frame_id: str | int

class SubmitRequest(BaseModel):
    name_output: str = "test"
    items: list[SubmitItem] = []

class BatchItem(BaseModel):
    video_id: str
    frame_id: str | int

class BatchImageRequest(BaseModel):
    items: list[BatchItem] = []

    @field_validator("items")
    @classmethod
    def validate_batch_items(cls, v):
        if len(v) > 200:
            raise ValueError("batch items count must not exceed 200")
        return v

# Cấu trúc dữ liệu nhận từ giao diện khi bấm nút "Truy xuất"
class SearchRequest(BaseModel):
    type: str
    query: str
    question: str | None = None
    name_output: str | None = None
    events: list[EventInput] | None = None

    @field_validator("events")
    @classmethod
    def validate_events(cls, v):
        if v is not None:
            if not (1 <= len(v) <= 2):
                raise ValueError("events must contain 1 or 2 items")
            if not v[0].query_text.strip():
                raise ValueError("Event 1 query_text must not be blank")
        return v

# ==========================================
# 1. API Tìm kiếm (Tạm thời trả về dữ liệu giả lập)
# ==========================================
@app.post("/api/search")
async def search_video(req: SearchRequest):
    req_id = str(uuid.uuid4())[:8]
    start_time = time.perf_counter()
    cache_hits = 0
    cache_misses = 0
    downloads = 0

    if req.name_output:
        repo_root = Path(__file__).resolve().parents[4]
        search_kis_path = repo_root / "search" / "search_kis.py"
        if not search_kis_path.exists():
            raise HTTPException(status_code=500, detail=f"Runner script not found: {search_kis_path}")
        
        # Import sanitize helper if available or perform server-side basename validation
        try:
            sys.path.insert(0, str(repo_root / "search"))
            import search_kis
            sanitized_name = search_kis.sanitize_output_basename(req.name_output)
        except Exception:
            sanitized_name = Path(req.name_output).name
            if not sanitized_name:
                raise HTTPException(status_code=400, detail="Invalid name_output parameter")

        # Ghi file query/KIS.json nếu req.events được truyền lên
        if req.events:
            event_dicts = []
            for ev in req.events:
                event_dicts.append({
                    "query_text": ev.query_text,
                    "ocr_text": ev.ocr_text,
                    "asr_text": ev.asr_text,
                    "object": ev.object
                })
            kis_payload = {
                "events": event_dicts,
                "e2_rerank": 1
            }
            query_dir = repo_root / "query"
            query_dir.mkdir(parents=True, exist_ok=True)
            kis_json_path = query_dir / "KIS.json"

            async with kis_lock:
                # Atomic write KIS.json
                temp_file = None
                try:
                    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=query_dir, delete=False, suffix=".tmp") as tf:
                        temp_file = Path(tf.name)
                        json.dump(kis_payload, tf, ensure_ascii=False, indent=2)
                    os.replace(temp_file, kis_json_path)
                except Exception as err:
                    if temp_file and temp_file.exists():
                        temp_file.unlink(missing_ok=True)
                    raise HTTPException(status_code=500, detail=f"Failed writing query/KIS.json: {err}")

                log_dir = repo_root / "log"
                log_dir.mkdir(parents=True, exist_ok=True)
                log_csv = log_dir / f"{sanitized_name}.csv"
                try:
                    res = subprocess.run(
                        [sys.executable, str(search_kis_path), sanitized_name],
                        check=True,
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace"
                    )
                    if res.stdout:
                        print(f"Runner stdout:\n{res.stdout}")
                    if res.stderr:
                        print(f"Runner stderr:\n{res.stderr}")
                except subprocess.CalledProcessError as err:
                    error_msg = err.stderr.strip() if err.stderr else str(err)
                    print(f"Runner execution failed error_msg: {error_msg}")
                    raise HTTPException(status_code=500, detail=f"Search runner execution failed: {error_msg}")
                except Exception as err:
                    print(f"Runner start failed err: {err}")
                    raise HTTPException(status_code=500, detail=f"Failed to start search runner: {err}")
        else:
            log_dir = repo_root / "log"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_csv = log_dir / f"{sanitized_name}.csv"
            try:
                res = subprocess.run(
                    [sys.executable, str(search_kis_path), sanitized_name],
                    check=True,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace"
                )
                if res.stdout:
                    print(f"Runner stdout:\n{res.stdout}")
                if res.stderr:
                    print(f"Runner stderr:\n{res.stderr}")
            except subprocess.CalledProcessError as err:
                error_msg = err.stderr.strip() if err.stderr else str(err)
                print(f"Runner execution failed error_msg: {error_msg}")
                raise HTTPException(status_code=500, detail=f"Search runner execution failed: {error_msg}")
            except Exception as err:
                print(f"Runner start failed err: {err}")
                raise HTTPException(status_code=500, detail=f"Failed to start search runner: {err}")
    
    # Đọc kết quả từ CSV tại log/<sanitized_name>.csv
    results_list = []
    log_csv = repo_root / "log" / f"{sanitized_name}.csv"

    if log_csv.exists():
        try:
            sys.path.insert(0, str(repo_root / "search"))
            import view_kis_results
            raw_rows = view_kis_results.load_kis_csv(log_csv)

            # Build atomic neighbors sidecar JSON for all search target rows in 1 scan
            sidecar_path = repo_root / "log" / f"{sanitized_name}.neighbors.json"
            index_path = view_kis_results.KEYFRAME_INDEX_PATH
            if index_path.exists():
                try:
                    await asyncio.to_thread(
                        view_kis_results.build_search_neighbors_sidecar,
                        raw_rows,
                        sidecar_path,
                        index_path,
                        10
                    )
                    # Invalidate RAM cache entry for sidecar on rebuild
                    sidecar_ram_cache.pop(str(sidecar_path), None)
                except Exception as sidecar_err:
                    print(f"Neighbors sidecar build warning: {sidecar_err}")

            # Read exact per-rank metadata JSONL ({rank, video_id, frame_id, shot_id}) if created by search runner
            meta_jsonl_path = repo_root / "log" / f"{sanitized_name}.meta.jsonl"
            rank_meta_map: dict[int, tuple[str, int, int]] = {}
            if meta_jsonl_path.is_file():
                try:
                    with meta_jsonl_path.open("r", encoding="utf-8") as stream:
                        for line in stream:
                            if line.strip():
                                rec = json.loads(line)
                                if isinstance(rec, dict) and "rank" in rec and "video_id" in rec and "frame_id" in rec and "shot_id" in rec:
                                    r_val = int(rec["rank"])
                                    v_val = str(rec["video_id"]).strip()
                                    f_val = int(rec["frame_id"])
                                    s_val = rec["shot_id"]
                                    if isinstance(s_val, int) and s_val >= 0 and not isinstance(s_val, bool):
                                        rank_meta_map[r_val] = (v_val, f_val, s_val)
                except Exception as meta_err:
                    print(f"Metadata JSONL read warning: {meta_err}")

            for row in raw_rows:
                v_id = str(row["video_id"]).strip()
                f_id = int(row["frame_id"])
                f_id_str = str(f_id)
                rank = int(row.get("rank", 0))
                s_id = row.get("shot_id")
                
                # Priority 1: explicit shot_id on row
                # Priority 2: exact rank + video_id + frame_id validation join from internal metadata JSONL
                if s_id is None and rank in rank_meta_map:
                    mv_id, mf_id, ms_id = rank_meta_map[rank]
                    if mv_id == v_id and mf_id == f_id:
                        s_id = ms_id

                has_valid_shot = isinstance(s_id, int) and s_id >= 0 and not isinstance(s_id, bool)
                query_str = f"?shot_id={s_id}" if has_valid_shot else ""
                
                thumb_url = None
                if has_valid_shot:
                    try:
                        v_prefix = view_kis_results.keyframe_video_prefix(v_id)
                        webp_path = repo_root / "keyframes" / f"{v_prefix}_s{s_id:06d}_f{f_id:09d}.webp"
                        if webp_path.is_file():
                            thumb_url = f"/api/thumbnails/{v_id}/{f_id_str}{query_str}"
                    except Exception:
                        pass

                results_list.append({
                    "video_id": v_id,
                    "frame_id": f_id_str,
                    "matching_score": 1.0,
                    "image_url": f"/api/images/{v_id}/{f_id_str}{query_str}",
                    "thumbnail_url": thumb_url,
                    "predicted_answer": None
                })
        except Exception as read_err:
            print(f"Error loading CSV results from {log_csv}: {read_err}")

    if not results_list and log_csv.exists():
        raise HTTPException(status_code=500, detail=f"No results found in CSV: {log_csv}")

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    print(f"[TIMING] [req:{req_id}] search_video completed in {elapsed_ms:.2f}ms | total_results={len(results_list)}")

    return {
        "list_search_results": results_list,
        "neighbor_cache": sanitized_name
    }

# ==========================================
# API Get Neighbor Keyframes
# ==========================================
@app.get("/api/neighbor-frames/{video_id}/{frame_id}")
async def get_neighbor_frames(video_id: str, frame_id: int, radius: int = 10, neighbor_cache: str | None = None):
    req_id = str(uuid.uuid4())[:8]
    start_time = time.perf_counter()
    if radius < 1 or radius > 10:
        raise HTTPException(status_code=400, detail="radius must be between 1 and 10")

    repo_root = Path(__file__).resolve().parents[4]
    try:
        sys.path.insert(0, str(repo_root / "search"))
        import view_kis_results
        
        index_path = view_kis_results.KEYFRAME_INDEX_PATH
        cache_root = view_kis_results.KEYFRAME_CACHE_ROOT.resolve()
        neighbors: list[dict[str, object]] = []

        # 1. Try reading directly from sidecar JSON if neighbor_cache token is passed
        if neighbor_cache:
            import search_kis
            sanitized_cache = search_kis.sanitize_output_basename(neighbor_cache)
            sidecar_path = repo_root / "log" / f"{sanitized_cache}.neighbors.json"
            payload = await get_cached_sidecar_payload(sidecar_path)
            if payload is not None:
                canonical_key = f"{video_id}:{frame_id}"
                if canonical_key in payload:
                    raw_items = payload[canonical_key]
                    if isinstance(raw_items, list) and all(_is_valid_neighbor_item(x) for x in raw_items):
                        neighbors = raw_items

        # 2. Fallback to scanning index if sidecar miss
        if not neighbors:
            if not index_path.exists():
                raise HTTPException(status_code=500, detail=f"Keyframe index not found: {index_path}")
            neighbors = await asyncio.to_thread(
                view_kis_results.load_neighbor_keyframes,
                video_id,
                frame_id,
                index_path,
                radius
            )

        if not neighbors:
            return {"list_search_results": []}

        # 3. Process neighbors for local WebP thumbnails only (no Kaggle download/resolution)
        results_list = []
        for item in neighbors:
            v_id = str(item["video_id"])
            f_id = str(item["frame_id"])
            s_id = item.get("shot_id")
            has_valid_shot = isinstance(s_id, int) and s_id >= 0 and not isinstance(s_id, bool)
            query_str = f"?shot_id={s_id}" if has_valid_shot else ""
            
            # Check local WebP existence
            thumb_url = None
            if has_valid_shot:
                try:
                    v_prefix = view_kis_results.keyframe_video_prefix(v_id)
                    webp_path = repo_root / "keyframes" / f"{v_prefix}_s{s_id:06d}_f{int(f_id):09d}.webp"
                    if webp_path.is_file():
                        thumb_url = f"/api/thumbnails/{v_id}/{f_id}{query_str}"
                except Exception:
                    pass

            results_list.append({
                "video_id": v_id,
                "frame_id": f_id,
                "matching_score": 1.0,
                "image_url": f"/api/images/{v_id}/{f_id}{query_str}",
                "thumbnail_url": thumb_url,
                "predicted_answer": None
            })

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        print(f"[TIMING] [req:{req_id}] get_neighbor_frames completed in {elapsed_ms:.2f}ms | total_neighbors={len(results_list)}")

        return {"list_search_results": results_list}
    except HTTPException:
        raise
    except Exception as err:
        print(f"Error fetching neighbor frames for {video_id}/{frame_id}: {err}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch neighbor frames: {err}")

# ==========================================
# API Batch Resolve Images & Pre-generate Thumbnails (Deprecation stub)
# ==========================================
@app.post("/api/batch-images")
async def batch_images():
    return {"items": []}

# ==========================================
# API Submit CSV
# ==========================================
@app.post("/api/submit")
async def submit_csv(req: SubmitRequest):
    repo_root = Path(__file__).resolve().parents[4]
    try:
        sys.path.insert(0, str(repo_root / "search"))
        import search_kis
        sanitized_name = search_kis.sanitize_output_basename(req.name_output)
    except Exception:
        sanitized_name = Path(req.name_output).name
        if not sanitized_name:
            raise HTTPException(status_code=400, detail="Invalid name_output parameter")

    sub_dir = repo_root / "submission"
    sub_dir.mkdir(parents=True, exist_ok=True)
    csv_path = sub_dir / f"{sanitized_name}.csv"

    # Save CSV atomically
    try:
        sys.path.insert(0, str(repo_root / "search"))
        import search_kis
        items_to_write = [
            {"video_id": search_kis.canonical_video_id(item.video_id), "frame_id": item.frame_id}
            for item in req.items
        ]
        search_kis.write_kis_csv_atomic(csv_path, items_to_write)
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Failed to write submission CSV: {err}")

    return {"status": "success", "submission_path": str(csv_path), "count": len(items_to_write)}

def get_or_generate_thumbnail(full_image_path: Path, video_id: str, frame_id: str) -> Path | None:
    """Atomic Pillow thumbnail generation/cache (320x180 max)."""

    repo_root = Path(__file__).resolve().parents[4]
    thumb_dir = repo_root / "keyframe_cache" / "thumbnails" / video_id
    thumb_dir.mkdir(parents=True, exist_ok=True)
    thumb_path = thumb_dir / f"{frame_id}.jpg"

    if thumb_path.is_file() and thumb_path.stat().st_size > 0:
        return thumb_path

    try:
        from PIL import Image
        with Image.open(full_image_path) as img:
            img = img.convert("RGB")
            img.thumbnail((320, 180), Image.Resampling.LANCZOS)
            temp_thumb = None
            try:
                with tempfile.NamedTemporaryFile(mode="wb", dir=thumb_dir, delete=False, suffix=".tmp") as tf:
                    temp_thumb = Path(tf.name)
                    img.save(tf, format="JPEG", quality=80)
                os.replace(temp_thumb, thumb_path)
            except Exception as save_err:
                if temp_thumb and temp_thumb.exists():
                    temp_thumb.unlink(missing_ok=True)
                print(f"Error saving thumbnail atomically: {save_err}")
                return None
        return thumb_path
    except Exception as err:
        print(f"Error generating thumbnail for {video_id}/{frame_id}: {err}")
        return None

# ==========================================
# API Trả về hình ảnh thumbnail
# ==========================================
@app.get("/api/thumbnails/{video_id}/{frame_id}")
async def get_thumbnail(video_id: str, frame_id: str, shot_id: int | None = None):
    repo_root = Path(__file__).resolve().parents[4]
    cache_headers = {"Cache-Control": "public, max-age=86400"}

    try:
        sys.path.insert(0, str(repo_root / "search"))
        import view_kis_results
        view_kis_results.keyframe_video_prefix(video_id)
        frame_int = int(frame_id)
        if frame_int < 0:
            raise ValueError("frame_id negative")
        if shot_id is not None and shot_id < 0:
            raise ValueError("shot_id negative")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid parameters")

    # 1. Check local keyframes/*.webp if full identity (video_id, shot_id, frame_id) is provided
    if shot_id is not None:
        v_prefix = view_kis_results.keyframe_video_prefix(video_id)
        webp_path = repo_root / "keyframes" / f"{v_prefix}_s{shot_id:06d}_f{int(frame_id):09d}.webp"
        if webp_path.is_file():
            return FileResponse(str(webp_path), media_type="image/webp", headers=cache_headers)
        raise HTTPException(status_code=404, detail=f"WebP thumbnail not found: {video_id}/{frame_id}/{shot_id}")

    # 2. When shot_id is absent/invalid for Grid thumbnail requests, reject 404 before Kaggle download
    raise HTTPException(status_code=404, detail=f"Thumbnail requires valid shot_id: {video_id}/{frame_id}")

# ==========================================
# 2. API Trả về hình ảnh tĩnh (Hiển thị Thumbnail)
# ==========================================
@app.get("/api/images/{video_id}/{frame_id}")
async def get_image(video_id: str, frame_id: str, shot_id: int | None = None):
    req_id = str(uuid.uuid4())[:8]
    start_time = time.perf_counter()
    repo_root = Path(__file__).resolve().parents[4]
    cache_headers = {"Cache-Control": "public, max-age=86400"}

    try:
        sys.path.insert(0, str(repo_root / "search"))
        import view_kis_results
        view_kis_results.keyframe_video_prefix(video_id)
        frame_int = int(frame_id)
        if frame_int < 0:
            raise ValueError("frame_id negative")
        if shot_id is not None and shot_id < 0:
            raise ValueError("shot_id negative")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid parameters")

    # 1. Kiểm tra local Keyframes
    local_path = repo_root / "data_aic" / "Keyframes" / video_id / f"{frame_id}.jpg"
    if local_path.exists():
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        print(f"[TIMING] [req:{req_id}] get_image (local_hit) in {elapsed_ms:.2f}ms")
        return FileResponse(str(local_path), headers=cache_headers)

    # 2. Kiểm tra keyframe cache theo logic view_kis_results.py (Kaggle download / cache)
    try:
        sys.path.insert(0, str(repo_root / "search"))
        import view_kis_results
        cache_root = view_kis_results.KEYFRAME_CACHE_ROOT.resolve()
        index_path = view_kis_results.KEYFRAME_INDEX_PATH

        # 2a. Try exact cache lookup if shot_id can be inferred or constructed
        exact_cached = view_kis_results.resolve_exact_cache_path({"video_id": video_id, "frame_id": int(frame_id), "shot_id": shot_id}, cache_root)
        if exact_cached and exact_cached.is_file():
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            print(f"[TIMING] [req:{req_id}] get_image (exact_cache_hit) in {elapsed_ms:.2f}ms")
            return FileResponse(str(exact_cached), headers=cache_headers)

        # Chỉ khi hoàn toàn miss mới gọi resolve_results (tải Kaggle)
        if index_path.exists():
            print(f"[TIMING] [req:{req_id}] get_image cache miss -> triggering Kaggle download")
            req_item = {"rank": 1, "video_id": video_id, "frame_id": int(frame_id)}
            if shot_id is not None:
                req_item["shot_id"] = shot_id
            resolved = await asyncio.to_thread(
                view_kis_results.resolve_results,
                [req_item],
                keyframe_cache_root=cache_root,
                keyframe_index_path=index_path,
                max_workers=100
            )
            if resolved and "image_path" in resolved[0]:
                img_p = Path(resolved[0]["image_path"])
                if img_p.exists():
                    elapsed_ms = (time.perf_counter() - start_time) * 1000
                    print(f"[TIMING] [req:{req_id}] get_image (download_hit) in {elapsed_ms:.2f}ms")
                    return FileResponse(str(img_p), headers=cache_headers)
    except Exception as err:
        print(f"[TIMING] [req:{req_id}] get_image error: {err}")

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    print(f"[TIMING] [req:{req_id}] get_image (miss_404) in {elapsed_ms:.2f}ms")
    raise HTTPException(status_code=404, detail=f"Image not found: {video_id}/{frame_id}")

# ==========================================
# 3. API Trả về Video hỗ trợ Stream (Cho phép tua thời gian #t=...)
# ==========================================
@app.get("/api/videos/{video_id}")
def get_video(request: Request, video_id: str):
    video_path = f"./data_aic/Videos/{video_id}.mp4"
    
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Không tìm thấy video")

    file_size = os.path.getsize(video_path)
    
    # Lấy header 'range' do trình duyệt gửi lên để biết người dùng đang muốn xem từ giây nào
    range_header = request.headers.get("range")

    # Nếu trình duyệt không yêu cầu tua, trả về file như bình thường
    if not range_header:
        return FileResponse(video_path, media_type="video/mp4")

    # Tính toán số byte cần cắt ra để gửi
    start, end = range_header.replace("bytes=", "").split("-")
    start = int(start)
    end = int(end) if end else file_size - 1
    
    chunk_size = end - start + 1

    # Hàm đọc nhỏ file từ ổ cứng để tối ưu RAM
    def stream_video():
        with open(video_path, "rb") as video:
            video.seek(start)
            yield video.read(chunk_size)

    # Cấu hình các Header bắt buộc để trình duyệt hiểu đây là luồng stream 206
    headers = {
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(chunk_size),
    }

    return StreamingResponse(
        stream_video(), 
        status_code=206, 
        headers=headers, 
        media_type="video/mp4"
    )

# ==========================================
# 4. Mount Static UI Files & Keyframe Cache Mounts
# ==========================================
keyframe_thumb_dir = Path(__file__).resolve().parents[4] / "keyframe_cache" / "thumbnails"
if keyframe_thumb_dir.exists():
    app.mount("/keyframe-thumbnails", StaticFiles(directory=str(keyframe_thumb_dir)), name="keyframe-thumbnails")

keyframe_cache_dir = Path(__file__).resolve().parents[4] / "keyframe_cache" / "keyframes"
if keyframe_cache_dir.exists():
    app.mount("/cache-keyframes", StaticFiles(directory=str(keyframe_cache_dir)), name="cache-keyframes")

ui_dir = Path(__file__).resolve().parents[4] / "ui"
if ui_dir.exists():
    app.mount("/", StaticFiles(directory=str(ui_dir), html=True), name="ui")