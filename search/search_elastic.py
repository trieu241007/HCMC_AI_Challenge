import sys
from pathlib import Path

from elasticsearch import Elasticsearch


QUERY_DIR = Path(__file__).resolve().parent
BUILD_DIR = QUERY_DIR.parent / "build"

# Cho phép import package trong thư mục deloy/build
sys.path.insert(0, str(BUILD_DIR))

from elasticsearch_builder import search_to_jsonl


def main() -> None:
    client = Elasticsearch("http://localhost:9200")

    try:
        results = search_to_jsonl(
            client=client,
            index_name="video_frames_batch1",
            query_path=QUERY_DIR / "KIS.json",
            output_path=QUERY_DIR / "elastic_results.jsonl",
            size=200,
        )

        print(f"Đã tìm thấy {len(results)} kết quả")
        print(f"Đã lưu tại: {QUERY_DIR / 'search_results.jsonl'}")
    finally:
        client.close()


if __name__ == "__main__":
    main()