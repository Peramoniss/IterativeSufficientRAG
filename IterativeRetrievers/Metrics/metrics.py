from abc import ABC, abstractmethod
from typing import Union
from Inference.prompts import PromptTemplate
from Inference.requestManagers import RequestManager
import torch
from collections import Counter
import numpy as np
import ast
from rouge_score import rouge_scorer
import re

class Metric(ABC):
    @abstractmethod
    def compute(self, *args, **kwargs):
        """Compute and return the metric's score."""
        raise NotImplementedError

    @staticmethod
    def _as_list(references: Union[str, list[str]]) -> list[str]:
        """Normalize a single-string or iterator-of-strings input to a list."""
        if isinstance(references, str):
            try:
                parsed = ast.literal_eval(references)
                references = parsed if isinstance(parsed, list) else [references]
            except:
                references = [references]
        elif not isinstance(references, list):
            references = [str(references)]

        refs = list(references)
        refs = set(refs)
        cleaned_refs = []
        for ref in refs:
            ref_str = str(ref)
            # 1. Strip content inside parentheses, e.g., "Little Fish (film)" -> "Little Fish "
            ref_str = re.sub(r"\(.*?\)", "", ref_str)
            # 2. Strip punctuation (punctuation, quotes, etc.)
            ref_str = re.sub(r"[^\w\s]", "", ref_str)
            # 3. Lowercase and collapse multiple internal spaces into a single space
            ref_str = " ".join(ref_str.lower().split())

            cleaned_refs.append(ref_str)

        return cleaned_refs

class RetrievalMetric(Metric, ABC):
    """
    Narrows Metric for RAG scoring: compares the retrieved document ids or titles with reference documents.
    """

    @abstractmethod
    def compute(self, retrieved_docs: Union[list[str], list[int]], reference_docs: Union[list[str], list[int]]) -> float:
        """
        Score `retrieved_docs` against `reference_docs`.
        """
        raise NotImplementedError

class PrecisionAtK(RetrievalMetric):
    """Fraction of the top-k retrieved docs that are relevant."""
    def __init__(self, k: int):
        self.k = k
 
    @property
    def name(self) -> str:
        return f"precision@{self.k}"
 
    def compute(self, retrieved_docs: Union[list[str], list[int]], reference_docs: Union[list[str], list[int]]) -> float:
        top_k = self._as_list(retrieved_docs)[: self.k] # First k docs
        if not top_k:
            return 0.0
        relevant = self._as_list(reference_docs)
        hits = sum(1 for doc in top_k if doc in relevant) # How many of the top k docs are within the relevant docs
        return hits / len(top_k) # What's the percentage of retrieved documents that are relevant within the first k
 
 
class RecallAtK(RetrievalMetric):
    """Fraction of all relevant docs that appear in the top-k retrieved."""
    def __init__(self, k: int):
        self.k = k
 
    @property
    def name(self) -> str:
        return f"recall@{self.k}"
 
    def compute(self, retrieved_docs:list, reference_docs:list) -> float:
        relevant = self._as_list(reference_docs)
        if not relevant:
            return 0.0
        top_k = self._as_list(retrieved_docs)[: self.k] # First k docs
        hits = sum(1 for doc in top_k if doc in relevant)
        return hits / len(relevant) # What's the percentage of relevant docs retrieved in the first k
 
 
class HitRateAtK(RetrievalMetric):
    """1.0 if at least one relevant doc is in the top-k, else 0.0."""
    def __init__(self, k: int):
        self.k = k
 
    @property
    def name(self) -> str:
        return f"hit_rate@{self.k}"
 
    def compute(self, retrieved_docs, reference_docs) -> float:
        relevant = self._as_list(reference_docs)
        top_k = self._as_list(retrieved_docs)[: self.k]
        return 1.0 if any(doc.lower().strip() in relevant for doc in top_k) else 0.0 
 
class AveragePrecision(RetrievalMetric):
    """
    Average precision for a single query (precision averaged over the
    ranks of each relevant hit, normalized by total relevant docs).
    Average this over a dataset to get MAP.
    """
 
    def __init__(self, k: int = None):
        self.k = k  # None = consider the full ranked list
 
    @property
    def name(self) -> str:
        return f"average_precision@{self.k}" if self.k else "average_precision"
 
    def compute(self, retrieved_docs, reference_docs) -> float:
        relevant = self._as_list(reference_docs)
        if not relevant:
            return 0.0
 
        docs = retrieved_docs[: self.k] if self.k else retrieved_docs
        docs = self._as_list(docs)
        hits = 0
        precision_sum = 0.0
        for rank, doc in enumerate(docs, start=1):
            if doc.lower().strip() in relevant:
                hits += 1
                precision_sum += hits / rank
 
        return precision_sum / len(relevant)

class PrecisionRecallCurve(RetrievalMetric):
    """
    Precision/recall computed at every rank of the retrieved list.
    `compute()` returns the AUC-PR float; use `curve()` to get the raw points for plotting.
    """
 
    @property
    def name(self) -> str:
        return "pr_auc"
 
    def curve(self, retrieved_docs: list[int], reference_docs: list[int]):
        """Returns (precisions, recalls), one pair per rank position."""
        relevant = self._as_list(reference_docs)
        precisions, recalls = [], []
 
        if not relevant:
            return precisions, recalls
 
        hits = 0
        retrieved_docs = self._as_list(retrieved_docs)
        for rank, doc in enumerate(retrieved_docs, start=1):
            if doc.lower().strip() in relevant:
                hits += 1
            precisions.append(hits / rank)
            recalls.append(hits / len(relevant))
 
        return precisions, recalls
 
    def compute(self, retrieved_docs, reference_docs) -> float:
        precisions, recalls = self.curve(retrieved_docs, reference_docs)
        if not recalls:
            return 0.0
 
        # Trapezoidal integration of precision over recall.
        auc = 0.0
        prev_recall, prev_precision = 0.0, precisions[0]
        for p, r in zip(precisions, recalls):
            auc += (r - prev_recall) * (p + prev_precision) / 2.0
            prev_recall, prev_precision = r, p
        return auc

class LanguageMetric(Metric, ABC):
    """
    Narrows Metric for QA-style scoring: every language metric compares
    a single model answer against one or more reference/gold answers.
    """

    @abstractmethod
    def compute(self, prediction: str, references: Union[str, list[str]]) -> float:
        """
        Score `prediction` against `references`.

        references: either a single gold answer, or a list of
        acceptable gold answers (e.g. SQuAD-style multiple references).
        Implementations typically score against each and aggregate
        (e.g. take the max), using `_as_list` below to normalize input.
        """
        raise NotImplementedError

class LanguageF1(LanguageMetric):
    """Token-level F1 between a prediction and one or more references."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.name = "f1"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def compute(self, prediction: str, references: Union[str, list[str]], return_tuple = False):
        refs = self._as_list(references)
        scores = []
        f1s = []
        pred_ids = self.tokenizer(self._as_list(prediction), truncation=True, return_tensors="pt")["input_ids"][0].tolist()
        for ref in refs:
            encoded = self.tokenizer([ref], truncation=True, return_tensors="pt")
            ref_ids = encoded["input_ids"][0].tolist()
            f1, precision, recall = self._token_f1(pred_ids, ref_ids)
            if return_tuple:
                scores.append({"f1": f1, "precision": precision, "recall": recall})
            
            f1s.append(f1)

        return scores[np.argmax(f1s)] if return_tuple else max(f1s)
    
    def _token_f1(self, pred_tokens: list[str], ref_tokens: list[str]):
        """Bag-of-tokens F1 (order-independent), SQuAD-style."""
        if not pred_tokens or not ref_tokens:
            return 0.0, 0.0, 0.0
 
        pred_counts = Counter(pred_tokens) # Counter({'tok_a': 4, 'tok_b': 7, 'tok_c': 2})
        ref_counts = Counter(ref_tokens)

        common_tokens = 0
        for token, count in pred_counts.items():
            if token in ref_counts:
                min_common = min(count, ref_counts[token])
                common_tokens += min_common # Token 'token' appears 'min_common' times in both answers
 
        if common_tokens == 0: # Avoid division by 0
            return 0.0, 0.0, 0.0
 
        precision = common_tokens / len(pred_tokens)
        recall = common_tokens / len(ref_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        return f1, precision, recall

class LanguageExactMatch(LanguageMetric):
    """Exact-match scoring, kept as a second example of the pattern."""

    def __init__(self):
        self.name = "exact_match"

    def compute(self, prediction: str, references: Union[str, list[str]]):
        refs = self._as_list(references)       
        prediction = self._as_list(prediction)[0]
        if prediction in refs:
            return 1.0

        return 0.0

class LLMAsAJudge(LanguageMetric):
    """LLM Judges equivalence"""

    def __init__(self, inference_manager: RequestManager, model: str, temperature: float = 0.0):
        self.name = "llm_as_a_judge"
        self.inference_manager = inference_manager
        self.template = PromptTemplate(user_prompt="Reference answers: {references}\nCandidate answer: {candidate}", system_prompt="You are a judge that must verify if a candidate answer for a given question is semantically equivalent to any of the possible reference answers for that same question. Consider polysemy, synonyms, aliases, form (i.e. 19 is the same as nineteen), formatting and any other small differences like this that preserve the original semantic meaning as equivalent answer. Return a jsonl of a single line, with a single field: {\"judgement\": true} or {\"judgement\": false}") # Do not consider affirmations or reasoning over the question, though, such as \"To answer the provided question, I would need to know...\".
        self.model = model
        self.temperature = temperature

    def compute(self, prediction: str, references: Union[str, list[str]]):
        refs = self._as_list(references)
        messages = self.template.format_prompt({"references": refs, "candidate": prediction})
        answer, _, _ = self.inference_manager.query_model(messages=messages, model=self.model, temperature=self.temperature)
        try:
            if "true" in str(answer).lower().strip():
                return 1.0
        except Exception as e:
            print(f"LLM as a Judge returning 0 for error: {e}")
            return 0.0
            
        return 0.0

class LanguageRouge(LanguageMetric):
    """ROUGE scoring between a prediction and one or more references.

    Defaults to ROUGE-L; pass e.g. rouge_types=["rouge1", "rouge2", "rougeL"]
    for multiple ROUGE variants. When multiple types are requested, their
    scores are averaged into a single f1/precision/recall (see per_type in
    return_tuple mode for the individual breakdown).
    """

    def __init__(self, rouge_types: list[str] = None, use_stemmer: bool = True):
        self.rouge_types = rouge_types or ["rougeL"]
        self.scorer = rouge_scorer.RougeScorer(self.rouge_types, use_stemmer=use_stemmer)
        self.name = "rouge"

    def compute(self, prediction: str, references: Union[str, list[str]], return_tuple=False):
        refs = self._as_list(references)
        pred = self._as_list(prediction)[0]

        scores = []
        f1s = []
        for ref in refs:
            result = self.scorer.score(ref, pred)

            f1 = sum(result[rt].fmeasure for rt in self.rouge_types) / len(self.rouge_types)
            precision = sum(result[rt].precision for rt in self.rouge_types) / len(self.rouge_types)
            recall = sum(result[rt].recall for rt in self.rouge_types) / len(self.rouge_types)

            if return_tuple:
                scores.append({
                    "f1": f1,
                    "precision": precision,
                    "recall": recall,
                    "per_type": {rt: result[rt] for rt in self.rouge_types},
                })
            f1s.append(f1)

        return scores[np.argmax(f1s)] if return_tuple else max(f1s)