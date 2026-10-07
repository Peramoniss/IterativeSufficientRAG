from typing import override
from abc import ABC, abstractmethod
import requests
from transformers import AutoModel, AutoTokenizer
import numpy as np
import faiss
import json
from tqdm import tqdm
import torch

class Retriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, k: int = 5, return_corpus_doc: bool = True, corpus_offset: int = 0):
        pass

class DenseRetriever(Retriever):
    def __init__(self, model_name:str, base_index: faiss.Index, corpus_path: str, model=None, tokenizer=None, corpus_in_memory:bool = False):
        self.model = model if model is not None else AutoModel.from_pretrained(model_name) # Pode passar o modelo já carregado para não ter que fazer o load múltiplas vezes
        self.tokenizer = tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained(model_name) # Pode passar o modelo já carregado para não ter que fazer o load múltiplas vezes
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.index = base_index
        self.corpus_in_memory = corpus_in_memory
        self.corpus_path = corpus_path
        if corpus_in_memory:
            curr_id = 0
            with open(corpus_path, "r", encoding="utf-8") as f:
                lines = [json.loads(line) for line in f if line.strip()]
                self.corpus = {}
                for line in lines:
                    self.corpus[curr_id] = line 
                    curr_id += 1
                del lines

    def _mean_pooling(self, model_output, attention_mask):
        # Replica exatamente o Mean Pooling padrão do Sentence Transformers.
        token_embeddings = model_output[0]  # Primeiro elemento contém last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        return sum_embeddings / sum_mask

    def encode(self, text_documents:list[str], batch_size : int | None = None, feedback: bool = False):
        if batch_size == None: # Without a batch size, encode all at once
            inputs = self.tokenizer(text_documents, padding=True, truncation=True, return_tensors="pt").to(self.device) # pyrefly: ignore[not-callable]
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = self._mean_pooling(outputs, inputs['attention_mask'])
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1) # Normalização L2
            
            document_embeddings = embeddings.cpu().numpy().astype("float32")
        else:
            temp_embeddings = []
            iterator = tqdm(range(0, len(text_documents), batch_size)) if feedback else range(0, len(text_documents), batch_size)
            for i in iterator:
                inputs = self.tokenizer(text_documents[i:i+batch_size], padding=True, truncation=True, return_tensors="pt").to(self.device) # pyrefly: ignore[not-callable]
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    embeddings = self._mean_pooling(outputs, inputs['attention_mask'])
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1) # Normalização L2
                temp_embeddings.append(embeddings.cpu().numpy().astype("float32"))

            document_embeddings = np.concat(temp_embeddings, axis=0)

        document_embeddings = np.array(document_embeddings).astype("float32")
        return document_embeddings

    def load_to_index(self, text_documents: list[str], batch_size : int | None = None, feedback: bool = False):
        document_embeddings = self.encode(text_documents, batch_size, feedback)
        if feedback:
            print("Documents encoded. Loading index in memory...") # TODO: implement simple mmap for memory efficient indexing
        self.index.add(document_embeddings)

    def save_index(self, index_save_path:str):
        faiss.write_index(self.index, index_save_path)

    def read_index(self, index_path: str):
        self.index = faiss.read_index(index_path)

    @override
    def retrieve(self, query: str, k: int = 5, return_corpus_doc: bool = True, corpus_offset:int=0):
        query = f"query: {query}"
        query_embedding = self.encode([query])
        scores, indexes = self.index.search(query_embedding, k)
        if return_corpus_doc:
            returning_docs = []
            if self.corpus_in_memory:
                for index in indexes[0]:
                    doc = self.corpus.get(index, None)
                    if doc:
                        returning_docs.append(doc)
            else:
                with open(self.corpus_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line_data = json.loads(line)
                        if int(line_data["id"])+corpus_offset in indexes[0]: # Maybe check for type
                            returning_docs.append(line_data)
                            if len(returning_docs) >= len(indexes[0]):
                                break # Found all the documents needed

            return scores, indexes, returning_docs
        return scores, indexes

    def batch_retrieve(self, queries: list[str], k: int = 5, return_corpus_doc: bool = True):
        raise NotImplementedError("Segura aí meu bom uma hora eu implemento isso")
        # print("\nSearch Results:")
        # for i, doc_id in enumerate(indexes[0]):
        #     print(f"Rank {i+1} (Distance: {distances[0][i]:.4f}):")
        #     print(f"  {documents[doc_id]}")

class RemoteRetriever(Retriever):
    def __init__(self, host:str, max_attempts:int = 5):
        self.host = host
        self.max_attempts = max_attempts
        self.headers = {
            "ngrok-skip-browser-warning": "69420"
        }

    def retrieve(self, query: str, k: int = 5, return_corpus_doc: bool = True, corpus_offset: int = 0):
        payload: dict = {
            "query": query,
            "password": "kTOIe4wQQn 8wQ1XH2jS1 MjkZKOMab6 Rp3eMVT34D 1DitCzvPEm",
            "k": k,
            "return_corpus_doc": return_corpus_doc,
            "corpus_offset": corpus_offset
        }

        status = 0
        attempts = 0 
        while status != 200 and attempts < self.max_attempts:
            response = requests.post(f"{self.host}/retrieve", json=payload, headers=self.headers)
            status = response.status_code
        
        if attempts >= self.max_attempts:
            raise ConnectionError("Maximum attempts reached")

        response_dict = response.json()
        if return_corpus_doc:
            return response_dict["scores"], response_dict["indexes"], response_dict["documents"]
        else:
            return response_dict["scores"], response_dict["indexes"]

from pyserini.search.lucene import LuceneSearcher
import numpy as np
import json


class BM25Retriever(Retriever):
    """
    Sparse retriever backed by a pre-built Pyserini/Lucene BM25 index.
    Mirrors DenseRetriever's return shape: (scores, indexes, returning_docs).

    Unlike DenseRetriever, there's no separate corpus file to read at
    retrieve-time -- the Lucene index itself stores the raw document JSON,
    so `searcher.doc(hit.docid).raw()` gives back the same {"id", "contents"}
    structure your DenseRetriever's corpus_path JSONL is expected to have.
    """

    def __init__(self, index_path: str, k1: float = None, b: float = None):
        """
        index_path: path to a Pyserini Lucene index directory (as built by
                    `python -m pyserini.index.lucene ...`).
        k1, b: optional BM25 hyperparameters. If either is provided, both
               must be (Pyserini's set_bm25 requires both), overriding the
               index's default BM25 settings.
        """
        self.index_path = index_path
        self.searcher = LuceneSearcher(index_path)
        if k1 is not None or b is not None:
            if k1 is None or b is not None and k1 is None:
                raise ValueError("Both k1 and b must be provided together to override BM25 defaults.")
            self.searcher.set_bm25(k1, b)

    def _parse_hit(self, hit):
        """
        Pulls the stored raw JSON for a hit and normalizes its id the same
        way DenseRetriever does (int(id) + corpus_offset), so downstream
        code (metrics, dedup, etc.) can treat both retrievers' indexes
        identically.
        """
        doc = self.searcher.doc(hit.docid)
        if doc is None:
            return None, None
        data = json.loads(doc.raw())
        doc_id = int(data["id"])
        return doc_id, data

    @override
    def retrieve(self, query: str, k: int = 5, return_corpus_doc: bool = True):
        hits = self.searcher.search(query, k=k)

        indexes = []
        returning_docs = [] if return_corpus_doc else None
        scores = []

        for hit in hits:
            doc_id, data = self._parse_hit(hit)
            if doc_id is None:
                # Stored doc missing/unreadable for this hit -- skip it rather
                # than silently misaligning scores/indexes/returning_docs.
                continue
            indexes.append(doc_id)
            scores.append(hit.score)
            if return_corpus_doc:
                returning_docs.append(data)

        scores = np.array([scores], dtype="float32")
        indexes = np.array([indexes], dtype="int64")

        if return_corpus_doc:
            return scores, indexes, returning_docs
        return scores, indexes

    def batch_retrieve(self, queries: list[str], k: int = 5, return_corpus_doc: bool = True):
        """
        Unlike DenseRetriever's stub, Pyserini has native batch search
        support (multi-threaded under the hood), so this is implemented
        rather than raising NotImplementedError.
        """
        qids = [str(i) for i in range(len(queries))]
        batch_hits = self.searcher.batch_search(queries, qids, k=k)

        all_scores, all_indexes, all_docs = [], [], []
        for qid in qids:
            hits = batch_hits[qid]
            indexes, scores, docs = [], [], []
            for hit in hits:
                doc_id, data = self._parse_hit(hit, corpus_offset=0)
                if doc_id is None:
                    continue
                indexes.append(doc_id)
                scores.append(hit.score)
                if return_corpus_doc:
                    docs.append(data)
            all_scores.append(scores)
            all_indexes.append(indexes)
            if return_corpus_doc:
                all_docs.append(docs)

        scores_arr = np.array(all_scores, dtype="float32")
        indexes_arr = np.array(all_indexes, dtype="int64")

        if return_corpus_doc:
            return scores_arr, indexes_arr, all_docs
        return scores_arr, indexes_arr