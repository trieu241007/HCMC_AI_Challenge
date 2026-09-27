import unittest
from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parent
sys.path.insert(0, str(repo_root / "search"))

import search_kis

class TestShotIdPreservation(unittest.TestCase):
    def test_weighted_rrf_preserves_shot_id(self):
        faiss_results = [
            {"video_id": "L01_V001", "frame_id": 100, "rank": 1, "shot_id": 42},
            {"video_id": "L01_V001", "frame_id": 200, "rank": 2, "shot_id": 43},
        ]
        elastic_results = [
            {"video_id": "L01_V001", "frame_id": 100, "rank": 1},  # missing shot_id in elastic
            {"video_id": "L01_V001", "frame_id": 300, "rank": 2, "shot_id": 44},
        ]

        fused = search_kis.weighted_rrf(faiss_results, elastic_results)
        self.assertEqual(len(fused), 3)

        # Check for L01_V001 frame 100 -> shot_id must be 42
        item_100 = next(x for x in fused if x["frame_id"] == 100)
        self.assertEqual(item_100["shot_id"], 42)

        # Check for L01_V001 frame 200 -> shot_id must be 43
        item_200 = next(x for x in fused if x["frame_id"] == 200)
        self.assertEqual(item_200["shot_id"], 43)

        # Check for L01_V001 frame 300 -> shot_id must be 44
        item_300 = next(x for x in fused if x["frame_id"] == 300)
        self.assertEqual(item_300["shot_id"], 44)

    def test_meta_jsonl_writes_shot_id(self):
        import tempfile, json
        fused = [
            {"video_id": "L01_V001", "frame_id": 100, "shot_id": 42},
            {"video_id": "L01_V001", "frame_id": 200, "shot_id": None},
        ]
        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".jsonl") as tf:
            meta_path = Path(tf.name)
        try:
            search_kis.write_kis_meta_jsonl_atomic(meta_path, fused)
            with meta_path.open("r", encoding="utf-8") as f:
                lines = [json.loads(line) for line in f if line.strip()]
            self.assertEqual(len(lines), 2)
            self.assertEqual(lines[0]["shot_id"], 42)
            self.assertIsNone(lines[1]["shot_id"])
        finally:
            meta_path.unlink(missing_ok=True)

if __name__ == "__main__":
    unittest.main()
