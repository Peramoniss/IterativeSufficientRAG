import json
import os
import struct
import numpy as np
import faiss
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# ---------------- CONFIG ----------------
JSONL_PATH = "documents/wiki-18_utf-8.jsonl"
TEXT_FIELD = "contents"       # <-- CHECK against your actual jsonl keys
MODEL_NAME = "intfloat/e5-base-v2"
D = 768
NLIST = 65_536
QTYPE = faiss.ScalarQuantizer.QT_8bit
TRAIN_SAMPLE_SIZE = 4_000_000   # ~60 vectors/centroid at nlist=65536
ENCODE_BATCH_SIZE = 256         # lower this if you hit GPU OOM
OUTPUT_INDEX_PATH = "wiki2018_sq8_embedded.index"
OFFSETS_PATH = "wiki2018_line_offsets.npy"   # byte offset of each line
TRAIN_SAMPLE_PATH = "train_sample.npy"       # cached training vectors
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# -----------------------------------------


def build_line_offsets(path):
    """Scan the file once and record the byte offset of the start of every line."""
    offsets = []
    with open(path, "rb") as f:
        offset = 0
        for line in tqdm(f, desc="Indexing line offsets"):
            offsets.append(offset)
            offset += len(line)
    return np.array(offsets, dtype=np.int64)


def read_jsonl_texts(path):
    """Yields formatted passage text for each line, in file order."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            text = obj.get(TEXT_FIELD, "") or ""
            yield "passage: " + text


def encode_batch(model, texts):
    emb = model.encode(
        texts,
        batch_size=len(texts),
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return emb.astype("float32")

def read_line_at(f, offsets, line_id):
    f.seek(offsets[line_id])
    line = f.readline()
    obj = json.loads(line)
    text = obj.get(TEXT_FIELD, "") or ""
    return "passage: " + text

def build_length_sorted_order(path):
    """
    One pass over the file: record an approximate token length per line
    (word count is a fine, cheap proxy for real tokenization here).
    Returns an array of line indices sorted by length ascending.
    """
    lengths = []
    with open(path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Scanning lengths for bucketing"):
            obj = json.loads(line)
            text = obj.get(TEXT_FIELD, "") or ""
            lengths.append(len(text.split()))
    lengths = np.array(lengths, dtype=np.int32)
    order = np.argsort(lengths, kind="stable")  # ascending: short -> long
    return order  # order[k] = the line_id that goes in sorted position k

def main():
    print("Counting lines...")
    n_total = 0
    lengths = []
    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i <= 50000:  # sample, don't need the whole file
                obj = json.loads(line)
                lengths.append(len(obj.get(TEXT_FIELD, "")))
            n_total +=1

    print(np.percentile(lengths, [50, 95, 99, 100]))
    print(f"n_total = {n_total}")

    # ---- Check / Build Line Offsets ----
    if os.path.exists(OFFSETS_PATH):
        print(f"Found existing offsets file at '{OFFSETS_PATH}'. Loading...")
        offsets = np.load(OFFSETS_PATH)
    else:
        print("Building line-offset index (for O(1) text lookup by id later)...")
        offsets = build_line_offsets(JSONL_PATH)
        np.save(OFFSETS_PATH, offsets)

    assert len(offsets) == n_total, f"offsets len {len(offsets)} != n_total {n_total}"

    print(f"Loading model {MODEL_NAME} on {DEVICE}...")
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    model.max_seq_length = 256
    model.half()
    # ---- Check / Generate Training Sample (Pass 1) ----
    if os.path.exists(TRAIN_SAMPLE_PATH):
        print(f"Found existing training sample at '{TRAIN_SAMPLE_PATH}'. Loading and skipping Pass 1...")
        train_embeddings = np.load(TRAIN_SAMPLE_PATH)
    else:
        rng = np.random.default_rng(SEED)
        n_sample = min(TRAIN_SAMPLE_SIZE, n_total)
        train_ids = rng.choice(n_total, size=n_sample, replace=False)
        train_mask = np.zeros(n_total, dtype=bool)
        train_mask[train_ids] = True

        with open(JSONL_PATH, "r", encoding="utf-8") as f:
            sample_lengths = np.empty(n_sample, dtype=np.int32)
            for k, lid in enumerate(tqdm(train_ids, desc="Measuring sample lengths")):
                f.seek(offsets[lid])
                obj = json.loads(f.readline())
                sample_lengths[k] = len(obj.get(TEXT_FIELD, "").split())
        sort_order = np.argsort(sample_lengths, kind="stable")
        train_ids_sorted = train_ids[sort_order]

        print(f"Pass 1/2: encoding {n_sample} randomly sampled docs for IVF training...")
        train_embeddings = np.empty((n_sample, D), dtype="float32")
        with open(JSONL_PATH, "r", encoding="utf-8") as f, tqdm(total=n_sample, desc="Encoding train sample") as pbar:
            for start in range(0, n_sample, ENCODE_BATCH_SIZE):
                batch_ids = train_ids_sorted[start:start + ENCODE_BATCH_SIZE]
                batch_texts = [read_line_at(f, offsets, lid) for lid in batch_ids]
                emb = encode_batch(model, batch_texts)
                # write back in ORIGINAL (unsorted) position so downstream order doesn't matter for training anyway
                train_embeddings[start:start + len(batch_texts)] = emb
                pbar.update(len(batch_texts))

        print("Saving training sample to disk as a safety checkpoint...")
        np.save(TRAIN_SAMPLE_PATH, train_embeddings)

    # ---- Train the Index ----
    print("Training FAISS index...")
    quantizer = faiss.IndexFlatIP(D)
    index = faiss.IndexIVFScalarQuantizer(quantizer, D, NLIST, QTYPE, faiss.METRIC_INNER_PRODUCT)
    index.train(train_embeddings)
    assert index.is_trained
    del train_embeddings  # Free up memory

    # ---- Pass 2: encode everything, add to index with id = line number ----
    print("Pass 2/2: encoding full corpus and adding to index...")
    sorted_order = build_length_sorted_order(JSONL_PATH)  # array of line_ids, short -> long
    assert len(sorted_order) == n_total

    with open(JSONL_PATH, "r", encoding="utf-8") as f, tqdm(total=n_total, desc="Indexing") as pbar:
        for start in range(0, n_total, ENCODE_BATCH_SIZE):
            batch_line_ids = sorted_order[start:start + ENCODE_BATCH_SIZE]
            batch_texts = [read_line_at(f, offsets, lid) for lid in batch_line_ids]
            emb = encode_batch(model, batch_texts)
            ids = np.array(batch_line_ids, dtype="int64")  # TRUE line numbers, not sequential!
            index.add_with_ids(emb, ids)
            pbar.update(len(batch_texts))

    assert index.ntotal == n_total, f"index.ntotal={index.ntotal} != n_total={n_total} -- misalignment!"
    print("Writing index to disk...")
    faiss.write_index(index, OUTPUT_INDEX_PATH)
    print("Done. Index id space == JSONL line number, guaranteed.")


if __name__ == "__main__":
    main()