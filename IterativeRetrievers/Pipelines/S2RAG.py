import os
import re
import json
import time
import traceback
import requests
import pandas as pd
import torch
from tqdm import tqdm

from Inference.prompts import IRCoTPromptTemplate
from Inference.requestManagers import RequestManager
from Inference.ircot import RemoteSufficiency  # <- adjust to wherever your RemoteSufficiency lives
from SufficiencyModel.modelStructures import TextDataset, SemanticOnlyBERT
from Retriever.retriever import Retriever
from Metrics.metrics import LanguageMetric, LanguageExactMatch, Metric, RetrievalMetric


# ======================================================================================
# Helpers
# ======================================================================================
def parse_json_object(text):
    """Extracts the first JSON object from an LLM answer (tolerates ```json fences and preambles)."""
    if text is None:
        return None
    text = str(text).replace("```json", "").replace("```", "")
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


# ======================================================================================
# Sufficiency + Gap judge (decoupled into two prompts)
# ======================================================================================
class RemoteSufficiencyWithGaps(RemoteSufficiency):
    """
    Two-stage S2G-Judge:
      1) `__call__`       -> is the evidence sufficient?  (LLM prompt, remote BERT server or local BERT)
      2) `identify_gaps`  -> if not, what is missing?      (always an LLM prompt)

    `__call__` keeps the same contract as RemoteSufficiency: returns (logits [1,2], tokens_used)
    where index 1 == "sufficient". Therefore, the ONLY thing that changes between the
    "original" S2G-RAG and the "personal classifier" version is what backs stage 1.

    host:
        - RequestManager       -> stage 1 is done by the LLM (original S2G-RAG)
        - str starting w/ http -> stage 1 is done by your BERT server (needs `gap_host` for stage 2)
        - other str            -> stage 1 is done by a local BERT checkpoint (needs `gap_host` for stage 2)
    gap_host: RequestManager used for gap identification. Defaults to `host` when host is a RequestManager.
    """

    SUFFICIENCY_SYSTEM_PROMPT = (
        "You are a QA/RAG sufficiency judge.\nGiven a QUESTION and a CONTEXT (documents retrieved so far), decide whether the CONTEXT ALONE contains enough explicit information to reliably answer the QUESTION.\n"
        "Output format (strict):\n- Output exactly one JSON object and nothing else."
        "Schema: {\"sufficient\": true/false}"
    )

    GAP_SYSTEM_PROMPT = (
        "You are a QA/RAG information-gap analyst.\n"
        "Given a QUESTION and a CONTEXT (documents retrieved so far), in which the CONTEXT is NOT sufficient to answer the QUESTION, list the gap items that describe what information is still missing.\n"
        "Output format (strict):\n- Output exactly one JSON object and nothing else."
        "Schema: {\"gap items\": [\n\"category\": \"bridge entity | attribute | relation | evidence span | other\",\n\"target\": \"string\",\n\"slot\": \"string\", \n\"description\": \"string\"]}"
    )

    USER_TEMPLATE = "QUESTION\n{question}\n\nCONTEXT\n{context}\n\nNow output ONLY the JSON object:"

    def __init__(self, host, dataset: str, gap_host: RequestManager = None, sufficiency_model: str = "gpt-4o-mini",
                 gap_model: str = "gpt-4o-mini", max_attempts: int = 5, password: str = None, local_checkpoint: str = None):
        llm = isinstance(host, RequestManager)
        super().__init__(host, dataset, llm=llm, max_attempts=max_attempts)
        self.sufficiency_model = sufficiency_model
        self.gap_model = gap_model
        self.gap_host = gap_host if gap_host is not None else (host if llm else None)
        if self.gap_host is None:
            raise ValueError("Gap identification always needs an LLM: pass `gap_host` when the sufficiency classifier is a BERT server/local model.")

        self.local_model = None
        if not llm and not str(host).startswith("http"):  # local BERT checkpoint
            self.local_model = SemanticOnlyBERT("bert-base-uncased", hidden_channels=32, num_classes=2, heads=8, dropout_rate=0.2, freeze_bert_layers=6)
            self.local_model.eval()
            checkpoint = torch.load(host, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            self.local_model.load_state_dict(checkpoint["model_state_dict"])

    # ---------------------------------------------------------------- stage 1
    def __call__(self, query: str, docs: list[str], model: str = None):
        if self.llm:
            return self._llm_sufficiency(query, docs, model or self.sufficiency_model)
        if self.local_model is not None:
            with torch.no_grad():
                return self.local_model(TextDataset([(query, docs, 0)]))  # placeholder y
        return self._server_sufficiency(query, docs)

    def _llm_sufficiency(self, query, docs, model):
        context = "\n".join(docs) if docs else "(empty)"
        messages = [
            {"role": "system", "content": self.SUFFICIENCY_SYSTEM_PROMPT},
            {"role": "user", "content": self.USER_TEMPLATE.format(question=query, context=context)},
        ]
        attempts = 0
        while attempts < self.max_attempts:
            try:
                answer, input_tok, out_tok = self.host.query_model(messages, model)
                parsed = parse_json_object(answer)
                if parsed is None or "sufficient" not in parsed:
                    raise ValueError(f"Malformed sufficiency output: {answer}")
                sufficient = parsed["sufficient"] is True or str(parsed["sufficient"]).lower() == "true"
                logits = torch.tensor([[0.0, 1.0]]) if sufficient else torch.tensor([[1.0, 0.0]])
                return logits, input_tok + out_tok
            except Exception as e:
                print(f"Sufficiency LLM error: {e}")
                traceback.print_exc()
                attempts += 1
        raise ConnectionError("Maximum attempts reached")

    def _server_sufficiency(self, query, docs):
        payload = {
            "query": query, 
            "password": "kTOIe4wQQn 8wQ1XH2jS1 MjkZKOMab6 Rp3eMVT34D 1DitCzvPEm",
            "docs": docs
        }
        attempts = 0
        while attempts < self.max_attempts:
            try:
                response = requests.post(f"{self.host}/sufficiency/{self.dataset}", json=payload, headers=self.headers)
                if response.status_code == 200:
                    response_dict = response.json()
                    return torch.tensor(response_dict["logits"]), response_dict["suff_tokens_used"]
            except Exception as e:
                print(f"Sufficiency server error: {e}")
            attempts += 1  # (the original loop never incremented it)
        raise ConnectionError("Maximum attempts reached")

    # ---------------------------------------------------------------- stage 2
    def identify_gaps(self, query: str, docs: list[str], model: str = None):
        """Returns (list of gap dicts, tokens_used). Only called when stage 1 said 'insufficient'."""
        model = model or self.gap_model
        context = "\n".join(docs) if docs else "(empty)"
        messages = [
            {"role": "system", "content": self.GAP_SYSTEM_PROMPT},
            {"role": "user", "content": self.USER_TEMPLATE.format(question=query, context=context)},
        ]
        attempts = 0
        while attempts < self.max_attempts:
            try:
                answer, input_tok, out_tok = self.gap_host.query_model(messages, model)
                parsed = parse_json_object(answer)
                if parsed is None:
                    raise ValueError(f"Malformed gap output: {answer}")
                raw_gaps = parsed.get("gap_items", parsed.get("gap items", []))
                gaps = [g for g in raw_gaps if isinstance(g, dict)]
                return gaps, input_tok + out_tok
            except Exception as e:
                print(f"Gap LLM error: {e}")
                traceback.print_exc()
                attempts += 1
        return [], 0  # No usable gaps -> query falls back to the original question, as in the paper


# ======================================================================================
# S2G-RAG
# ======================================================================================
class S2GRAG:
    REASONER_INSTRUCTION = (
        "You are an intelligent assistant, adept at answering complex multi-hop questions using ONLY the Wikipedia evidence provided. "
        "Reason concisely step-by-step (follow the examples), and finish with \"So the answer is: <answer>.\""
    )

    REASONER_EXAMPLE = """
        Q: When did the director of the film Hypocrite die?
        A: The film Hypocrite was directed by Miguel Morayta. Miguel Morayta died on 19 June 2013. So the answer is: 19 June 2013.

        Q: Are both Kurram Garhi and Trojkrsti located in the same country?
        A: Kurram Garhi is located in the country of Pakistan. Trojkrsti is located in the country of Republic of Macedonia. Thus, they are not in the same country. So the answer is: no.

        Q: Who was born first out of Martin Hodge and Ivania Martinich?
        A: Martin Hodge was born on 4 February 1959. Ivania Martinich was born on 25 July 1995. Thus, Martin Hodge was born first. So the answer is: Martin Hodge.
        """

    EXTRACTOR_SYSTEM_PROMPT = (
        "You are a sentence-level evidence selector for a multi-hop RAG system.\n"
        "You will receive:\n1. an ORIGINAL QUESTION,\n2. MISSING FACTS describing what information is still missing,\n"
        "3. a numbered list of SENTENCES from retrieved documents.\n"
        "Your task is to select the sentence ids that maximize answerability for the ORIGINAL QUESTION.\n"
        "Selection policy:\n"
        "1. First prioritize sentences that fill the MISSING FACTS (bridge entities, attributes, relations, evidence spans needed for the next hop).\n"
        "2. Then prioritize sentences that directly support the final answer to the ORIGINAL QUESTION.\n"
        "3. Prefer self-contained and explicit sentences (key entity, relation, attribute, date, number or answer-bearing fact).\n"
        "4. If a selected sentence depends on nearby context to be understandable, include the minimal additional sentence(s) needed.\n"
        "5. Do not infer, rewrite, paraphrase, or generate evidence text. Only return ids from the provided list.\n"
        "6. If no sentence is useful, return an empty list.\n"
        "Output format (strict): return exactly one JSON object and nothing else:\n"
        "{\"evidence_global_ids\": [1, 5, 7]}\n"
        "Constraints:\n- \"evidence global ids\" must be a JSON array of integers.\n"
        "- Select at most K sentences, where K is given in the user message\n"
        "- Only use ids that appear in the numbered sentence list\n- Do not repeat ids."
    )

    def __init__(self, retriever: Retriever, inference_manager: RequestManager, generator_model: str,
                 sufficiency: RemoteSufficiencyWithGaps, extractor_model: str = None, mode: str = "chat",
                 load_if_exist=False, threshold=None, num_gap_phrases: int = 1, extractor_cap: int = 6):
        self.retriever = retriever
        self.inference_manager = inference_manager
        self.model = generator_model
        self.extractor_model = extractor_model if extractor_model is not None else generator_model  # paper: same backbone as the reasoner
        if mode.lower() not in ["chat", "completion"]:
            raise ValueError("Mode must be either \"chat\" or \"completion\". Completion only affects the final reasoner call; the extractor always uses chat.")
        self.mode = mode.lower()
        self.load_if_exist = load_if_exist
        self.threshold = threshold
        self.suff_model = sufficiency
        self.K = num_gap_phrases
        self.Ke = extractor_cap

        self.template = IRCoTPromptTemplate(
            system_prompt="",
            user_prompt=self.REASONER_INSTRUCTION + "\n\n" + self.REASONER_EXAMPLE + "{reference}Q: Answer the following question by concisely reasoning step-by-step (follow the examples above). {question}\nA: {suffix}",
            reference_template="Wikipedia Title: {title}\n{text}\n\n",
            reference_label=True
        )

    # ------------------------------------------------------------------ utils
    def extract_answer(self, full_answer):
        answer = re.search(r"answer is:?\s*(.+)", full_answer, flags=re.IGNORECASE)
        if answer:
            final_answer = answer.groups()[0].strip()
            return re.sub(r'[^\w\s]', '', final_answer)
        else:
            return None

    def _process_docs(self, new_docs):
        context = []
        for doc in new_docs:
            context_separated = str(doc["contents"]).split('\n', 1)
            title = context_separated[0]
            text = context_separated[1] if len(context_separated) > 1 else ""
            context.append({"title": title, "text": text})
        return context

    @staticmethod
    def _split_sentences(text):
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    @staticmethod
    def _evidence_to_docs(evidence: dict):
        """Evidence context {title: [sentences]} -> [{"title", "text"}] (same format the prompt template expects)."""
        return [{"title": title, "text": " ".join(sents)} for title, sents in evidence.items()]

    @staticmethod
    def _evidence_to_text(evidence: dict):
        """Evidence context as list[str] for the sufficiency/gap judge."""
        return [f"[{title}] {' '.join(sents)}" for title, sents in evidence.items()]

    # ------------------------------------------------------------------ S2G components
    def _is_sufficient(self, query, evidence):
        """Stage 1 of the judge. Returns (bool, tokens_used)."""
        logits, tokens = self.suff_model(query, self._evidence_to_text(evidence))
        if self.threshold is None:
            return bool((logits.argmax(1) == 1).item()), tokens
        dist = logits.softmax(1)
        return bool(dist.argmax(1).item() == 1 and dist[0][1].item() > self.threshold), tokens

    def _build_query(self, query, gaps):
        """Gap-guided query construction (Sec. 3.3): target+slot, else description; first K valid phrases appended to q."""
        phrases = []
        for gap in gaps:
            target, slot, desc = str(gap.get("target", "")).strip(), str(gap.get("slot", "")).strip(), str(gap.get("description", "")).strip()
            phrase = f"{target} {slot}".replace("_", " ") if (target and slot) else desc
            if phrase:
                phrases.append(phrase)
            if len(phrases) >= self.K:
                break
        return f"{query} {' '.join(phrases)}" if phrases else query

    def _extract_evidence(self, query, gaps, retrieved_docs, evidence):
        """Sentence-pointer extraction (Sec. 3.4). Mutates `evidence`. Returns (input_tokens, output_tokens)."""
        pool = []  # globally indexed (title, sentence)
        already = {s for sents in evidence.values() for s in sents}
        for doc in retrieved_docs:
            for sentence in self._split_sentences(doc["text"]):
                if sentence not in already:
                    pool.append((doc["title"], sentence))
        if not pool:
            return 0, 0

        missing = "\n".join(f"- {g.get('description') or (str(g.get('target', '')) + ' ' + str(g.get('slot', '')))}" for g in gaps) or "- (none specified)"
        numbered = "\n".join(f"[{i}] ({title}) {sentence}" for i, (title, sentence) in enumerate(pool, start=1))
        messages = [
            {"role": "system", "content": self.EXTRACTOR_SYSTEM_PROMPT},
            {"role": "user", "content": f"ORIGINAL QUESTION:\n{query}\n\nMISSING FACTS:\n{missing}\n\nSENTENCES:\n{numbered}\n\nK = {self.Ke}\nOutput ONLY the JSON object:"},
        ]
        selected, input_tok, out_tok = [], 0, 0
        try:
            answer, input_tok, out_tok = self.inference_manager.query_model(messages, self.extractor_model)
            parsed = parse_json_object(answer) or {}
            ids = parsed.get("evidence_global_ids", parsed.get("evidence global ids", []))
            for idx in ids:
                if isinstance(idx, int) and 1 <= idx <= len(pool) and idx not in selected:
                    selected.append(idx)
            selected = selected[:self.Ke]
        except Exception as e:
            print(f"Extractor exception: {e}")

        if not selected:  # Parsing/format failure fallback: leading sentence of each new doc, capped at K_e (empty list from the LLM is respected only if parse succeeded)
            if 'parsed' in locals() and parsed and ("evidence_global_ids" in parsed or "evidence global ids" in parsed):
                return input_tok, out_tok
            seen_titles = set()
            for i, (title, _) in enumerate(pool, start=1):
                if title not in seen_titles:
                    seen_titles.add(title)
                    selected.append(i)
                if len(selected) >= self.Ke:
                    break

        for idx in sorted(selected):  # Deterministic mapping ids -> original sentences (verbatim), keeps document order
            title, sentence = pool[idx - 1]
            evidence.setdefault(title, []).append(sentence)
        return input_tok, out_tok

    def _reason(self, query, evidence):
        messages = self.template.format_prompt(query, retrieved_documents=self._evidence_to_docs(evidence), suffix="")
        start = time.time()
        if self.mode == "chat":
            answer, input_tok, out_tok = self.inference_manager.query_model(messages, self.model)
        else:  # completion
            answer, input_tok, out_tok = self.inference_manager.complete(messages[-1]["content"], self.model)
        return answer, input_tok, out_tok, time.time() - start

    # ------------------------------------------------------------------ main loop (Algorithm 1)
    def process_query(self, query, k=6, max_iter=4):
        """max_iter == T: maximum number of retrieval turns."""
        evidence = {}  # accumulated Evidence Context C_t: {title: [verbatim sentences]}
        complete_docs, seen_titles = [], set()  # everything retrieved (used for retrieval metrics), title-deduplicated
        first_turn_docs = None
        input_tokens_used, output_tokens_generated, suff_tokens_used, time_used = 0, 0, 0, 0
        ever_sufficient = -1
        gaps = []

        for t in range(max_iter + 1):
            start = time.time()
            # ---- S2G-Judge, stage 1: sufficiency
            if not evidence and not self.judge_empty_context:
                sufficient, tok = False, 0  # Nothing to judge yet; go straight to gap identification
            else:
                sufficient, tok = self._is_sufficient(query, evidence)
            suff_tokens_used += tok
            if sufficient and ever_sufficient == -1:
                ever_sufficient = t + 1

            if sufficient or t == max_iter:
                time_used += time.time() - start
                answer, input_tok, out_tok, reasoning_time = self._reason(query, evidence)
                input_tokens_used += input_tok
                output_tokens_generated += out_tok
                time_used += reasoning_time
                extracted = self.extract_answer(answer)
                return (extracted if extracted is not None else answer), first_turn_docs, complete_docs, self._evidence_to_docs(evidence), \
                    input_tokens_used, suff_tokens_used, output_tokens_generated, time_used, t + 1, ever_sufficient

            # ---- S2G-Judge, stage 2: gaps (only when insufficient)
            gaps, tok = self.suff_model.identify_gaps(query, self._evidence_to_text(evidence))
            suff_tokens_used += tok

            # ---- Gap-guided query + retrieval
            gap_query = self._build_query(query, gaps)
            _, _, returning_docs = self.retriever.retrieve(gap_query, k=k)
            context = self._process_docs(returning_docs)
            if t == 0:
                first_turn_docs = context.copy()
            new_docs = [d for d in context if d["title"] not in seen_titles]  # title-based de-duplication across turns
            for doc in new_docs:
                seen_titles.add(doc["title"])
                complete_docs.append(doc)
            time_used += time.time() - start

            # ---- Sentence-level evidence extraction: C_t = C_{t-1} (+) E_t
            start = time.time()
            input_tok, out_tok = self._extract_evidence(query, gaps, new_docs, evidence)
            input_tokens_used += input_tok
            output_tokens_generated += out_tok
            time_used += time.time() - start

    # ------------------------------------------------------------------ dataset loop
    def process_dataset(self, df: pd.DataFrame, query_col="query", answer_col="answer", docs_col="references", doc_reference_field="title", max_iter=4, k=6, metrics: list[Metric] = [LanguageExactMatch()], display=False, intermediate_results_path=None):
        if doc_reference_field not in ["title", "text"]:
            raise ValueError("Parameter \"doc_reference_field\" must be either \"title\" or \"text\"")
        results = {}
        results["inference_time"], results["input_tokens_used"], results["sufficiency_tokens_used"], results["output_tokens_generated"], results["turns"], results["sufficient"] = [], [], [], [], [], []
        results["answer"], results["answer_references"], results["docs"], results["evidence"] = [], [], [], []
        total_instances = df.shape[0]
        already_processed = 0
        if self.load_if_exist and intermediate_results_path != None and os.path.exists(intermediate_results_path):
            existing_df = pd.read_csv(intermediate_results_path)
            already_processed = existing_df.shape[0]
            results = existing_df.to_dict(orient="list")
            del existing_df
        iterator = tqdm(df.iterrows(), total=total_instances, unit="instances") if display else df.iterrows()
        for i, row in iterator:
            if self.load_if_exist and i < already_processed:
                continue
            key, answer, refs = None, None, None
            try:
                refs = row[answer_col] if (isinstance(row[answer_col], str) or isinstance(row[answer_col], list)) else str(row[answer_col])
                answer, first_turn_docs, complete_docs, evidence_docs, input_tokens_used, sufficiency_tokens_used, output_tokens_generated, inference_time, turns, ever_sufficient = self.process_query(row[query_col], k=k, max_iter=max_iter)
                for metric in metrics:
                    key = metric.name
                    if results.get(key, None) is None:
                        results[key] = []
                    if isinstance(metric, LanguageMetric):
                        result = metric.compute(answer, refs)
                    elif isinstance(metric, RetrievalMetric):
                        analysis_doc = first_turn_docs if key.endswith(f"@{k}") else complete_docs
                        retrieved_docs = []
                        for doc in analysis_doc:
                            doc_title = doc[doc_reference_field]
                            if doc_title not in retrieved_docs:
                                retrieved_docs.append(doc_title)
                        result = metric.compute(list(retrieved_docs), row[docs_col])
                    results[key].append(result)

                results["inference_time"].append(inference_time)
                results["input_tokens_used"].append(input_tokens_used)
                results["sufficiency_tokens_used"].append(sufficiency_tokens_used)
                results["output_tokens_generated"].append(output_tokens_generated)
                results["turns"].append(turns)
                results["sufficient"].append(ever_sufficient)
                results["answer"].append(answer)
                results["answer_references"].append(refs)
                results["docs"].append(complete_docs)
                results["evidence"].append(evidence_docs)

                if intermediate_results_path != None:
                    out_df = pd.DataFrame(results)
                    out_df = out_df.replace(r"\n", r"\\n", regex=True)
                    out_df.to_csv(intermediate_results_path, encoding="utf-8", index=False)
                    del out_df
            except Exception as e:
                print(f"Exception in instance {i}: {e}. Current metric: {key if key is not None else 'Unknown'}. Current pred: {answer}; Current references {refs}")
                traceback.print_exc()
                continue
        return results

    judge_empty_context = False  # If True, also runs the sufficiency classifier on the empty C_0 (paper's Algorithm 1 does; it is wasted work)