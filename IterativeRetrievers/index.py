import faiss
from sentence_transformers import SentenceTransformer
import json
import numpy as np

# Configuration
model = SentenceTransformer("intfloat/e5-base-v2")
batch_size = 5

complete_embeddings = []
encoding_documents = []
print("Encoding...")
with open("wiki-18_utf-8.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        line_data = json.loads(line)
        encoding_documents.append(f"passage: {line_data['contents']}")

        if len(encoding_documents) >= batch_size:
            embeddings = model.encode(encoding_documents, normalize_embeddings=True, show_progress_bar=True)
            complete_embeddings.append(embeddings)
            encoding_documents = []

    if encoding_documents:
        embeddings = model.encode(encoding_documents, normalize_embeddings=True, show_progress_bar=True)
        complete_embeddings.append(embeddings)

complete_embeddings = np.vstack(complete_embeddings).astype("float32")

# 1. Create the IVFSQ8 index
# Use faiss.METRIC_L2 for Euclidean distance, or faiss.METRIC_INNER_PRODUCT for Cosine similarity
print("Creating index")
dimension = complete_embeddings.shape[1]  # 768 for e5-base-v2
index = faiss.IndexFlatL2(dimension)

# 3. Add vectors
index.add(complete_embeddings)
faiss.write_index(index, "wiki_e5.index")