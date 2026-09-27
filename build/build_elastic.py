from pathlib import Path
from elasticsearch import Elasticsearch
from elasticsearch_builder import build_index

client = Elasticsearch(
    "http://localhost:9200",
    request_timeout=120,
)

result = build_index(
    client=client,
    jsonl_path=Path(r"C:\linhtinh\HCMC AI\deloy\data\elastic_data.jsonl"),
    label_path=Path(r"C:\linhtinh\HCMC AI\deloy\data\label.json"),
    index_name="video_frames_batch1",
    batch_size=1000,
)

print(result)