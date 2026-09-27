from __future__ import annotations

import argparse
from pathlib import Path

import faiss
import numpy as np

NLIST = 4096               # Roughly 4 * sqrt(415,000) to 16 * sqrt(415,000)
TRAIN_SIZE = 200_000       # Needs to be at least 39 * NLIST (39 * 4096 ≈ 160k)
ADD_BATCH_SIZE = 100_000   # Increased for better memory/speed throughput
SEED = 42


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a cosine IVF_FLAT index")
    parser.add_argument("input", type=Path, help="Input embeddings.npy")
    parser.add_argument("output", type=Path, help="Output embeddings.faiss")
    args = parser.parse_args()

    vectors = np.load(args.input, mmap_mode="r", allow_pickle=False)
    if vectors.ndim != 2 or vectors.shape[0] == 0:
        raise ValueError(f"Expected a non-empty 2D matrix, got {vectors.shape}")

    row_count, dimension = vectors.shape
    train_count = min(TRAIN_SIZE, row_count)
    nlist = min(NLIST, max(1, train_count // 39))

    rng = np.random.default_rng(SEED)
    train_rows = rng.choice(row_count, size=train_count, replace=False)
    training = np.ascontiguousarray(vectors[train_rows], dtype=np.float32)

    quantizer = faiss.IndexFlatIP(dimension)
    index = faiss.IndexIVFFlat(
        quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT
    )
    index.train(training)

    for start in range(0, row_count, ADD_BATCH_SIZE):
        end = min(start + ADD_BATCH_SIZE, row_count)
        batch = np.ascontiguousarray(vectors[start:end], dtype=np.float32)
        index.add(batch)

    if index.ntotal != row_count:
        raise RuntimeError(f"Expected {row_count} vectors, indexed {index.ntotal}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(args.output))
    print(
        f"Saved {index.ntotal:,} vectors, dimension={dimension}, "
        f"nlist={nlist}: {args.output}"
    )


if __name__ == "__main__":
    main()
