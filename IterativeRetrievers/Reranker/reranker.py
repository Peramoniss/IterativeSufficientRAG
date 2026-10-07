from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
from tqdm import tqdm
import torch
from abc import ABC, abstractmethod
import requests

class Reranker(ABC):
    @abstractmethod
    def rerank(self, query: str, documents: list[str], k: int = None, batch_size: int = None, display: bool = False):
        pass

class LocalReranker(Reranker):
    def __init__(self, model_name: str, model=None, tokenizer=None, max_length: int = 1024):
        self.model = model if model is not None else AutoModelForSequenceClassification.from_pretrained(model_name) # Pode passar o modelo já carregado para não ter que fazer o load múltiplas vezes
        self.tokenizer = tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained(model_name) # Pode passar o tokenizer já carregado para não ter que fazer o load múltiplas vezes
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.max_length = max_length

    def _score_batch(self, query: str, documents: list[str]) -> np.ndarray:
        pairs = [[query, doc["text"]] for doc in documents]
        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

            # Modelos de reranking geralmente têm 1 logit (regressão) ou 2 (classificação binária)
            if logits.shape[-1] == 1:
                scores = logits.squeeze(-1)
            else:
                scores = torch.nn.functional.softmax(logits, dim=-1)[:, 1]

        return scores.cpu().numpy().astype("float32")

    def score(self, query: str, documents: list[str], batch_size: int = None, display: bool = False) -> np.ndarray:
        if not documents:
            return np.array([], dtype="float32")

        if batch_size is None: # Sem batch size, pontua tudo de uma vez
            return self._score_batch(query, documents)

        temp_scores = []
        iterator = tqdm(range(0, len(documents), batch_size)) if display else range(0, len(documents), batch_size)
        for i in iterator:
            batch = documents[i:i + batch_size]
            temp_scores.append(self._score_batch(query, batch))

        return np.concatenate(temp_scores, axis=0)

    def rerank(self, query: str, documents: list[str], k: int = None, batch_size: int = None, display: bool = False):
        """
        Reordena uma lista de documentos com base na relevância à query.

        Args:
            query: a query de busca.
            documents: lista de documentos. Pode ser lista de strings
            text_key: chave para extrair o texto quando documents é uma lista de dicts.
            k: número de documentos a retornar após o reranking (None = todos).
            batch_size: tamanho do batch para pontuação.
            display: exibe barra de progresso (tqdm) se True.

        Returns:
            reranked_docs: lista de documentos reordenados (mesmo formato de entrada).
            scores: scores correspondentes, já ordenados (maior -> menor).
        """
        scores = self.score(query, documents, batch_size=batch_size, display=display)

        order = np.argsort(-scores) # Ordem decrescente de relevância
        if k is not None:
            order = order[:k]

        reranked_docs = [documents[i] for i in order]
        reranked_scores = scores[order]

        return reranked_docs, reranked_scores

class RemoteReranker(Reranker):
    def __init__(self, host:str, max_attempts:int = 5):
        self.host = host
        self.max_attempts = max_attempts
        self.headers = {
            "ngrok-skip-browser-warning": "69420"
        }

    def rerank(self, query: str, documents: list[str], k: int = None, batch_size: int = None):
        payload: dict = {
            "query": query,
            "password": "kTOIe4wQQn 8wQ1XH2jS1 MjkZKOMab6 Rp3eMVT34D 1DitCzvPEm",
            "documents": documents,
            "k": k,
            "batch_size": batch_size
        }

        status = 0
        attempts = 0 
        while status != 200 and attempts < self.max_attempts:
            response = requests.post(f"{self.host}/rerank", json=payload, headers=self.headers)
            status = response.status_code
        
        if attempts >= self.max_attempts:
            raise ConnectionError("Maximum attempts reached")

        response_dict = response.json()
        return response_dict["reranked_docs"], response_dict["reranked_scores"]