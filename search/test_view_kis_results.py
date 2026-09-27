from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import view_kis_results as viewer_module
from view_kis_results import (
    KAGGLE_DATASET,
    download_kaggle_file,
    keyframe_video_prefix,
    resolve_results,
)


class FakeDecodedImage:
    """Small Pillow boundary double; Pillow itself owns JPEG decoding tests."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.format = "JPEG"

    def __enter__(self) -> "FakeDecodedImage":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def load(self) -> None:
        data = self.path.read_bytes()
        if (
            not data.startswith(b"\xff\xd8")
            or not data.endswith(b"\xff\xd9")
            or b"corrupt" in data
            or b"interrupted" in data
        ):
            raise OSError("cannot decode JPEG")


class ResolveKaggleResultsTests(unittest.TestCase):
    def setUp(self) -> None:
        fake_pil = types.ModuleType("PIL")
        fake_image_module = types.ModuleType("PIL.Image")
        fake_image_module.open = FakeDecodedImage  # type: ignore[attr-defined]
        fake_pil.Image = fake_image_module  # type: ignore[attr-defined]
        self.pillow_patcher = patch.dict(
            sys.modules,
            {
                "PIL": fake_pil,
                "PIL.Image": fake_image_module,
            },
        )
        self.pillow_patcher.start()

        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.index_path = self.root / "keyframes.jsonl"
        self.metadata = {
            "video_id": "L21_V001",
            "frame_id": 62,
            "shot_id": 3,
        }
        unrelated_metadata = {
            "video_id": "L99_V999",
            "frame_id": 1,
            "shot_id": 1,
        }
        self.index_path.write_text(
            "\n".join(
                (
                    json.dumps(unrelated_metadata),
                    json.dumps(self.metadata),
                )
            )
            + "\n",
            encoding="utf-8",
        )
        self.results = [
            {
                "rank": 1,
                "video_id": "L21_V001",
                "frame_id": 62,
            }
        ]

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()
        self.pillow_patcher.stop()

    def test_downloads_only_the_matching_kaggle_image_into_the_cache(self) -> None:
        cache_root = self.root / "cache"
        calls: list[tuple[str, Path]] = []

        def fake_downloader(dataset_path: str, output_dir: Path) -> Path:
            calls.append((dataset_path, output_dir))
            downloaded_path = output_dir / Path(dataset_path)
            downloaded_path.parent.mkdir(parents=True, exist_ok=True)
            downloaded_path.write_bytes(b"\xff\xd8jpeg-data\xff\xd9")
            return downloaded_path

        try:
            resolved = resolve_results(
                self.results,
                cache_root,
                self.index_path,
                downloader=fake_downloader,
            )
        except TypeError as error:
            self.fail(f"resolve_results does not support Kaggle downloads: {error}")

        expected_path = (
            cache_root
            / "keyframes"
            / "l21-v001_s000003_f000000062.jpg"
        )
        self.assertEqual(
            calls,
            [
                (
                    "keyframes/l21-v001_s000003_f000000062.jpg",
                    cache_root,
                )
            ],
        )
        self.assertEqual(resolved[0]["image_path"], expected_path)
        self.assertTrue(expected_path.is_file())

    def test_reuses_a_cached_image_without_downloading_it_again(self) -> None:
        cache_root = self.root / "cache"
        cached_path = (
            cache_root
            / "keyframes"
            / "l21-v001_s000003_f000000062.jpg"
        )
        cached_path.parent.mkdir(parents=True)
        cached_path.write_bytes(b"\xff\xd8cached-jpeg-data\xff\xd9")

        def unexpected_downloader(_dataset_path: str, _output_dir: Path) -> Path:
            self.fail("The downloader must not run for an existing cache file")

        try:
            resolved = resolve_results(
                self.results,
                cache_root,
                self.index_path,
                downloader=unexpected_downloader,
            )
        except TypeError as error:
            self.fail(f"resolve_results does not support cache reuse: {error}")

        self.assertEqual(resolved[0]["image_path"], cached_path)

    def test_downloads_a_result_batch_concurrently_and_preserves_csv_order(
        self,
    ) -> None:
        cache_root = self.root / "batch-cache"
        metadata_rows = [
            {"video_id": "L21_V001", "frame_id": 10, "shot_id": 1},
            {"video_id": "L21_V002", "frame_id": 20, "shot_id": 2},
            {"video_id": "L21_V003", "frame_id": 30, "shot_id": 3},
            {"video_id": "L21_V004", "frame_id": 40, "shot_id": 4},
        ]
        batch_results = [
            {
                "rank": rank,
                "video_id": metadata["video_id"],
                "frame_id": metadata["frame_id"],
            }
            for rank, metadata in enumerate(metadata_rows, start=1)
        ]
        self.index_path.write_text(
            "\n".join(json.dumps(row) for row in reversed(metadata_rows)) + "\n",
            encoding="utf-8",
        )

        lock = threading.Lock()
        active = 0
        maximum_active = 0

        def fake_downloader(dataset_path: str, output_dir: Path) -> Path:
            nonlocal active, maximum_active
            with lock:
                active += 1
                maximum_active = max(maximum_active, active)
            try:
                time.sleep(0.05)
                downloaded_path = output_dir / Path(dataset_path)
                downloaded_path.parent.mkdir(parents=True, exist_ok=True)
                downloaded_path.write_bytes(b"\xff\xd8batch-jpeg\xff\xd9")
                return downloaded_path
            finally:
                with lock:
                    active -= 1

        try:
            resolved = resolve_results(
                batch_results,
                cache_root,
                self.index_path,
                downloader=fake_downloader,
                max_workers=4,
            )
        except TypeError as error:
            self.fail(f"resolve_results does not support batch workers: {error}")

        self.assertGreaterEqual(maximum_active, 2)
        self.assertEqual(
            [Path(result["image_path"]).name for result in resolved],
            [
                "l21-v001_s000001_f000000010.jpg",
                "l21-v002_s000002_f000000020.jpg",
                "l21-v003_s000003_f000000030.jpg",
                "l21-v004_s000004_f000000040.jpg",
            ],
        )

    def test_cancels_pending_batch_downloads_after_the_first_failure(self) -> None:
        cache_root = self.root / "failure-cache"
        metadata_rows = [
            {
                "video_id": f"L21_V{video_number:03d}",
                "frame_id": video_number * 10,
                "shot_id": video_number,
            }
            for video_number in range(1, 7)
        ]
        batch_results = [
            {
                "rank": rank,
                "video_id": metadata["video_id"],
                "frame_id": metadata["frame_id"],
            }
            for rank, metadata in enumerate(metadata_rows, start=1)
        ]
        self.index_path.write_text(
            "\n".join(json.dumps(row) for row in metadata_rows) + "\n",
            encoding="utf-8",
        )

        release_downloads = threading.Event()
        lock = threading.Lock()
        started: list[str] = []

        def failing_downloader(dataset_path: str, output_dir: Path) -> Path:
            with lock:
                started.append(dataset_path)
            if "l21-v001_" in dataset_path:
                raise RuntimeError("simulated network failure")
            release_downloads.wait(timeout=1)
            downloaded_path = output_dir / Path(dataset_path)
            downloaded_path.parent.mkdir(parents=True, exist_ok=True)
            downloaded_path.write_bytes(b"\xff\xd8batch-jpeg\xff\xd9")
            return downloaded_path

        release_timer = threading.Timer(0.2, release_downloads.set)
        release_timer.start()
        try:
            with self.assertRaisesRegex(
                RuntimeError,
                "keyframes/l21-v001_s000001_f000000010.jpg",
            ):
                resolve_results(
                    batch_results,
                    cache_root,
                    self.index_path,
                    downloader=failing_downloader,
                    max_workers=2,
                )
        finally:
            release_downloads.set()
            release_timer.cancel()
            release_timer.join()

        self.assertLess(len(started), len(metadata_rows))

    def test_redownloads_an_incomplete_cached_jpeg(self) -> None:
        cache_root = self.root / "cache"
        cached_path = (
            cache_root
            / "keyframes"
            / "l21-v001_s000003_f000000062.jpg"
        )
        cached_path.parent.mkdir(parents=True)
        cached_path.write_bytes(b"\xff\xd8interrupted-download")
        calls: list[str] = []

        def fake_downloader(dataset_path: str, output_dir: Path) -> Path:
            calls.append(dataset_path)
            downloaded_path = output_dir / Path(dataset_path)
            downloaded_path.write_bytes(b"\xff\xd8complete-jpeg\xff\xd9")
            return downloaded_path

        resolved = resolve_results(
            self.results,
            cache_root,
            self.index_path,
            downloader=fake_downloader,
        )

        self.assertEqual(
            calls,
            ["keyframes/l21-v001_s000003_f000000062.jpg"],
        )
        self.assertEqual(resolved[0]["image_path"], cached_path)
        self.assertEqual(cached_path.read_bytes(), b"\xff\xd8complete-jpeg\xff\xd9")

    def test_redownloads_a_corrupt_jpeg_with_valid_boundary_markers(self) -> None:
        cache_root = self.root / "cache"
        cached_path = (
            cache_root
            / "keyframes"
            / "l21-v001_s000003_f000000062.jpg"
        )
        cached_path.parent.mkdir(parents=True)
        cached_path.write_bytes(b"\xff\xd8corrupt-jpeg\xff\xd9")
        calls: list[str] = []

        def fake_downloader(dataset_path: str, output_dir: Path) -> Path:
            calls.append(dataset_path)
            downloaded_path = output_dir / Path(dataset_path)
            downloaded_path.write_bytes(b"\xff\xd8complete-jpeg\xff\xd9")
            return downloaded_path

        resolved = resolve_results(
            self.results,
            cache_root,
            self.index_path,
            downloader=fake_downloader,
        )

        self.assertEqual(
            calls,
            ["keyframes/l21-v001_s000003_f000000062.jpg"],
        )
        self.assertEqual(resolved[0]["image_path"], cached_path)
        self.assertEqual(cached_path.read_bytes(), b"\xff\xd8complete-jpeg\xff\xd9")

    def test_rejects_video_ids_that_could_escape_the_cache(self) -> None:
        with self.assertRaisesRegex(ValueError, "supported L<digits>_V<digits>"):
            keyframe_video_prefix(r"C:\outside")

    def test_kaggle_adapter_requests_one_exact_dataset_file(self) -> None:
        cache_root = self.root / "cache"
        cache_root.mkdir()
        dataset_path = "keyframes/l21-v001_s000003_f000000062.jpg"
        expected_path = cache_root / Path(dataset_path)
        calls: list[tuple[str, dict[str, str]]] = []

        def fake_dataset_download(handle: str, **kwargs: str) -> str:
            calls.append((handle, kwargs))
            expected_path.parent.mkdir(parents=True, exist_ok=True)
            expected_path.write_bytes(b"\xff\xd8jpeg-data\xff\xd9")
            return str(expected_path)

        fake_kagglehub = types.SimpleNamespace(
            dataset_download=fake_dataset_download,
        )
        with patch.dict(sys.modules, {"kagglehub": fake_kagglehub}):
            downloaded_path = download_kaggle_file(dataset_path, cache_root)

        self.assertEqual(downloaded_path, expected_path)
        self.assertEqual(
            calls,
            [
                (
                    KAGGLE_DATASET,
                    {
                        "path": dataset_path,
                        "output_dir": str(cache_root),
                    },
                )
            ],
        )

    def test_kaggle_adapter_explains_when_output_dir_is_unsupported(self) -> None:
        cache_root = self.root / "cache"
        cache_root.mkdir()

        def old_dataset_download(_handle: str, **_kwargs: str) -> str:
            raise TypeError("unexpected keyword argument 'output_dir'")

        old_kagglehub = types.SimpleNamespace(
            dataset_download=old_dataset_download,
        )
        with patch.dict(sys.modules, {"kagglehub": old_kagglehub}):
            with self.assertRaisesRegex(
                RuntimeError,
                "Upgrade kagglehub",
            ):
                download_kaggle_file(
                    "keyframes/l21-v001_s000003_f000000062.jpg",
                    cache_root,
                )

    def test_formats_total_time_and_speed_for_100_images(self) -> None:
        self.assertTrue(
            hasattr(viewer_module, "format_load_summary"),
            "The viewer must provide a terminal load-time summary",
        )
        summary = viewer_module.format_load_summary(100, 25.0)

        self.assertEqual(
            summary,
            (
                "Image loading finished: 100 images in 25.00 seconds\n"
                "Average: 0.250 seconds/image | Speed: 4.00 images/second"
            ),
        )

    def test_load_kis_csv_supports_shot_id(self) -> None:
        csv_file = self.root / "test_shot.csv"
        csv_file.write_text("L21_V001,100,10\nL21_V001,100,20\n", encoding="utf-8")
        rows = viewer_module.load_kis_csv(csv_file)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["shot_id"], 10)
        self.assertEqual(rows[1]["shot_id"], 20)

    def test_submission_csv_strictly_two_columns(self) -> None:
        import search_kis
        sub_path = self.root / "sub_test.csv"
        results = [{"video_id": "L21_V001", "frame_id": 100, "shot_id": 10}]
        search_kis.write_kis_csv_atomic(sub_path, results)
        content = sub_path.read_text(encoding="utf-8").strip()
        self.assertEqual(content, "L21_V001,100")
        lines = content.splitlines()
        self.assertEqual(len(lines[0].split(",")), 2)

    def test_internal_meta_jsonl_preserves_shot_id(self) -> None:
        import search_kis
        meta_path = self.root / "meta_test.meta.jsonl"
        results = [
            {"video_id": "L21_V001", "frame_id": 100, "shot_id": 10},
            {"video_id": "L21_V001", "frame_id": 100, "shot_id": 20},
        ]
        search_kis.write_kis_meta_jsonl_atomic(meta_path, results)
        lines = [json.loads(l) for l in meta_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0]["rank"], 1)
        self.assertEqual(lines[0]["shot_id"], 10)
        self.assertEqual(lines[1]["rank"], 2)
        self.assertEqual(lines[1]["shot_id"], 20)

    def test_duplicate_shots_exact_selection_and_sidecar_neighbor(self) -> None:
        cache_root = self.root / "shot-cache"
        index_path = self.root / "shot-keyframes.jsonl"
        metadata_rows = [
            {"video_id": "L21_V001", "frame_id": 100, "shot_id": 1},
            {"video_id": "L21_V001", "frame_id": 100, "shot_id": 2},
            {"video_id": "L21_V001", "frame_id": 105, "shot_id": 2},
        ]
        index_path.write_text(
            "\n".join(json.dumps(row) for row in metadata_rows) + "\n",
            encoding="utf-8",
        )

        def fake_downloader(dataset_path: str, output_dir: Path) -> Path:
            downloaded_path = output_dir / Path(dataset_path)
            downloaded_path.parent.mkdir(parents=True, exist_ok=True)
            downloaded_path.write_bytes(b"\xff\xd8fake-jpeg\xff\xd9")
            return downloaded_path

        # 1. Test duplicate shot exact selection
        results = [{"rank": 1, "video_id": "L21_V001", "frame_id": 100, "shot_id": 2}]
        resolved = resolve_results(
            results, cache_root, index_path, downloader=fake_downloader
        )
        self.assertEqual(len(resolved), 1)
        self.assertIn("s000002", str(resolved[0]["image_path"]))

        # 2. Test sidecar building
        sidecar_path = self.root / "test.neighbors.json"
        raw_rows = [{"rank": 1, "video_id": "L21_V001", "frame_id": 100}]
        viewer_module.build_search_neighbors_sidecar(
            raw_rows, sidecar_path, index_path, radius=5
        )
        self.assertTrue(sidecar_path.is_file())
        sidecar_data = json.loads(sidecar_path.read_text(encoding="utf-8"))
        self.assertIn("L21_V001:100", sidecar_data)

        # 3. Test neighbor loading
        neighbors = viewer_module.load_neighbor_keyframes(
            "L21_V001", 100, index_path, radius=5
        )
        self.assertTrue(len(neighbors) >= 2)
        self.assertEqual(neighbors[0]["video_id"], "L21_V001")


if __name__ == "__main__":
    unittest.main()
