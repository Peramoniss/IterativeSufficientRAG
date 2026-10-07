from Inference.prompts import PromptTemplate
from Inference.requestManagers import RequestManager
from SufficiencyModel.modelStructures import GraphDataset, SyntacticGATWithBERT
from Retriever.retriever import Retriever
from Reranker.reranker import Reranker
from Metrics.metrics import LanguageMetric, LanguageExactMatch, LanguageF1, Metric, RetrievalMetric
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import time
import requests
 
class SearchO1: 
    BEGIN_SEARCH_QUERY = "<|begin_search_query|>"
    END_SEARCH_QUERY = "<|end_search_query|>"
    BEGIN_SEARCH_RESULT = "<|begin_search_result|>"
    END_SEARCH_RESULT = "<|end_search_result|>"
 
    def __init__(self, retriever: Retriever, inference_manager: RequestManager, model: str,
                 reranker: Reranker = None, mode: str = "chat"):
        self.retriever = retriever
        self.reranker = reranker
        self.inference_manager = inference_manager
        self.model = model
        if mode.lower() not in ["chat", "completion"]:
            raise ValueError("Mode must be either \"chat\" or \"completion\". We suggest using chat "
                              "for bigger models and completion for open, quantized models with low "
                              "parameter count")
        self.mode = mode.lower()
 
        # I: main reasoning task instruction. Tells the model how to signal a search.
        self.TASK_INSTRUCTION = (
            """You are a reasoning assistant with the ability to perform web searches to help you answer the user's question accurately. You have special tools:
            To perform a search: write <|begin_search_query|> your query here <|end_search_query|>. Then, the system will search and analyze relevant web pages, then provide you with helpful information in the format <|begin_search_result|> ...search results... <|end_search_result|>.
            You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {search_limit}.
            Once you have all the information you need, continue your reasoning.
            Example:
            Question: “...”
            Assistant thinking steps:
            - I might need to look up details about ...
            Assistant: <|begin_search_query|>...<|end_search_query|>
            (System returns processed information from relevant web pages)
            Assistant continues reasoning with the new information...
            Remember:
            - Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.
            - When done searching, continue your reasoning. """
        )
        
 
        # Idocs: instruction for the Reason-in-Documents sub-step.
        self.REASON_IN_DOCS_INSTRUCTION = (
            """Task Instruction:
            You are tasked with reading and analyzing web pages based on the following inputs: Previous Reasoning Steps, Current Search Query, and Searched Web Pages. 
            Your objective is to extract relevant and helpful information for Current Search Query from the Searched Web Pages and seamlessly integrate this information into the Previous Reasoning Steps to continue reasoning for the original question.
            Guidelines:
            1. Analyze the Searched Web Pages:
            - Carefully review the content of each searched web page.
            - Identify factual information that is relevant to the Current Search Query and can aid in the reasoning process for the original question.
            2. Extract Relevant Information:
            - Select the information from the Searched Web Pages that directly contributes to advancing the Previous Reasoning Steps.
            - Ensure that the extracted information is accurate and relevant.
            3. Output Format:
            - If the web pages provide helpful information for current search query: Present the information beginning with "Final Information" as shown below.
            Final Information
            [Helpful information]
            If the web pages do not provide any helpful information for current search query: Output the following text.
            Final Information
            No helpful information found.
            """
        )
 
        # Main reasoning prompt: instruction + one demonstration + the live question.
        self.main_prompt_prefix = f"{self.TASK_INSTRUCTION}\n\nPlease answer the following question. You should provide your final answer in the format <|boxed|>YOUR_ANSWER<|boxed|>. Q: {{question}}\nA: {{suffix}}"

        # Reason-in-documents prompt template, reusing the existing reference formatter.
        self.reason_template = PromptTemplate(
            system_prompt=self.REASON_IN_DOCS_INSTRUCTION,
            user_prompt="""Inputs:
            - Original Query:
            {query}
            - Previous Reasoning Steps:
            {prev_reasoning}
            - Current Search Query:
            {search_query}
            - Searched Web Pages:
            {document}
            Now you should analyze each web page and find helpful information based on the current search query "{search_query}" and previous reasoning steps.""",
        )
 
    def extract_answer(self, full_answer):
        answer = re.search(r"boxed|>(.+)<|boxed|>", full_answer, flags=re.IGNORECASE)
        if answer:
            final_answer = answer.groups()[0].strip()
            return re.sub(r'[^\w\s]', '', final_answer)
        return None
 
    def extract_search_query(self, text):
        """Extracts the query from a trailing <|begin_search_query|>...<|end_search_query|> span."""
        match = re.search(
            re.escape(self.BEGIN_SEARCH_QUERY) + r"(.+?)" + re.escape(self.END_SEARCH_QUERY),
            text, flags=re.DOTALL
        )
        if match:
            return match.group(1).strip()
        return None
 
    def _process_docs(self, new_docs):
        context = []
        for doc in new_docs:
            context_separated = str(doc["contents"]).split('\n', 1)
            title = context_separated[0]
            text = context_separated[1]
            context.append({"title": title, "text": text})
        return context
        
 
    def _reason_in_documents(self, query, search_query, reasoning_so_far, docs):
        """
        Line 15-20 of Algorithm 1: given the docs retrieved for `search_query`,
        distill them into a knowledge-injected reasoning snippet (rfinal).
        """
        prompt_messages = self.reason_template.format_prompt({"query": query, "prev_reasoning": reasoning_so_far, "search_query": search_query, "document": docs})
        # format_prompt returns a messages-style structure per IRCoT's convention;
        # extract the raw text for the completion path, reuse as-is for chat.
        if self.mode == "chat":
            content, in_tok, out_tok = self.inference_manager.query_model(prompt_messages, self.model)
        else:
            content, in_tok, out_tok = self.inference_manager.complete(prompt_messages[0]["content"], self.model)
        return content.strip(), in_tok, out_tok
 
    def process_query(self, query, k=5, max_iter=10):
        seq = self.main_prompt_prefix.format(question=query, search_limit=max_iter, suffix="")
        full_answer = ""
        complete_docs = []
        input_tokens_used, output_tokens_generated, time_used = 0, 0, 0
        turns = 0
 
        for i in range(max_iter):
            turns = i + 1
            try:
                start = time.time()
                if self.mode == "chat":
                    messages = [{"role": "user", "content": seq}]
                    generated, input_tokens_used, output_tokens_generated = self.inference_manager.query_model(messages, self.model)
                    # print("Initial generation: ",generated)
                else:  # completion
                    return self.inference_manager.complete(prompt_text, self.model)
                time_used += time.time() - start
                input_tokens_used += input_tokens_used
                output_tokens_generated += output_tokens_generated
            except Exception as e:
                print(f"Exception during main generation: {e}")
                continue
 
            seq += generated
            full_answer += generated
 
            search_query = self.extract_search_query(generated)
            if search_query is not None:
                # Line 9-12: retrieve, build Reason-in-Documents input
                try:
                    _, _, returning_docs = self.retriever.retrieve(search_query, k=k)
                    # print("Retrieval: ",returning_docs)
                    context = self._process_docs(returning_docs)
                    for doc in context:
                        if doc not in complete_docs:
                            complete_docs.append(doc)
                    if self.reranker is not None:
                        complete_docs, _ = self.reranker.rerank(query, complete_docs)
 
                    start = time.time()
                    rfinal, r_in_tok, r_out_tok = self._reason_in_documents(
                        query, search_query, seq, complete_docs
                    )
                    time_used += time.time() - start
                    input_tokens_used += r_in_tok
                    output_tokens_generated += r_out_tok
                except Exception as e:
                    print(f"Exception during reason-in-documents: {e}")
                    rfinal = "No relevant information was found."
 
                # Line 21: splice the distilled knowledge back into the sequence
                seq += f"\n{self.BEGIN_SEARCH_RESULT}{rfinal}{self.END_SEARCH_RESULT}\n"
                continue  # keep generating in the next iteration
 
            # No search requested this turn -> treat as a finished sequence (line 13-14)
            extracted_answer = self.extract_answer(full_answer)
            if extracted_answer is not None:
                return extracted_answer, complete_docs, input_tokens_used, output_tokens_generated, time_used, turns
            # Model stopped without a search tag and without a final-answer marker;
            # nothing left to trigger further generation, so end here.
            return full_answer, complete_docs, input_tokens_used, output_tokens_generated, time_used, turns
 
        # max_iter exhausted mid-search-loop
        return full_answer, complete_docs, input_tokens_used, output_tokens_generated, time_used, turns
 
    def process_dataset(self, df: pd.DataFrame, query_col="query", answer_col="answer", docs_col="references",
                         doc_reference_field="title", max_iter=10, k=5,
                         metrics: list[Metric] = [LanguageExactMatch()], display=False,
                         intermediate_results_path=None):
        if doc_reference_field not in ["title", "text"]:
            raise ValueError("Parameter \"doc_reference_field\" must be either \"title\" or \"text\"")
        results = {"inference_time": [], "input_tokens_used": [], "output_tokens_generated": [], "turns": [],
                   "answer": [], "answer_references": [], "docs": []}
        total_instances = df.shape[0]
        iterator = tqdm(df.iterrows(), total=total_instances, unit="instances") if display else df.iterrows()
        for i, row in iterator:
            try:
                key = None
                raw_refs = row[answer_col]
                if isinstance(raw_refs, list):
                    refs = [re.sub(r'[^\w\s]', '', str(r)) for r in raw_refs]
                else:
                    refs = re.sub(r'[^\w\s]', '', str(raw_refs))
                answer, complete_docs, input_tokens_used, output_tokens_generated, inference_time, turns = self.process_query(row[query_col], k=k, max_iter=max_iter)
 
                for metric in metrics:
                    key = metric.name
                    if results.get(key) is None:
                        results[key] = []
                    if isinstance(metric, LanguageMetric):
                        result = metric.compute(answer, refs)
                    elif isinstance(metric, RetrievalMetric):
                        retrieved_docs = []
                        for doc in complete_docs:
                            doc_title = doc[doc_reference_field]
                            if doc_title not in retrieved_docs:
                                retrieved_docs.append(doc_title)
                        result = metric.compute(list(retrieved_docs), row[docs_col])
                    results[key].append(result)
 
                results["inference_time"].append(inference_time)
                results["input_tokens_used"].append(input_tokens_used)
                results["output_tokens_generated"].append(output_tokens_generated)
                results["turns"].append(turns)
                results["answer"].append(answer)
                results["answer_references"].append(refs)
                results["docs"].append(complete_docs)
 
                if intermediate_results_path is not None:
                    out_df = pd.DataFrame(results)
                    out_df = out_df.replace(r"\n", r"\\n", regex=True)
                    out_df.to_csv(intermediate_results_path, encoding="utf-8", index=False)
                    del out_df
            except Exception as e:
                print(f"Exception in instance {i}: {e}. Current metric: {key if key is not None else 'Unknown'}")
                continue
 
        return results





class SearchO1WithSufficiency:
    BEGIN_SEARCH_QUERY = "<|begin_search_query|>"
    END_SEARCH_QUERY = "<|end_search_query|>"
    BEGIN_SEARCH_RESULT = "<|begin_search_result|>"
    END_SEARCH_RESULT = "<|end_search_result|>"
 
    def __init__(self, retriever: Retriever, inference_manager: RequestManager, generator_model: str, sufficiency_model_path:str,
                 reranker: Reranker = None, mode: str = "chat"):
        self.retriever = retriever
        self.reranker = reranker
        self.inference_manager = inference_manager
        self.model = generator_model
        if mode.lower() not in ["chat", "completion"]:
            raise ValueError("Mode must be either \"chat\" or \"completion\". We suggest using chat "
                              "for bigger models and completion for open, quantized models with low "
                              "parameter count")
        self.mode = mode.lower()

        if "http" in sufficiency_model_path:
            self.suff_model = RemoteSufficiency(sufficiency_model_path)       
            self.remote_suff = True
        else:
            self.suff_model =  SyntacticGATWithBERT("bert-base-uncased", hidden_channels=32, num_classes=2, heads=8, dropout_rate=0.2, freeze_bert_layers=6)
            self.suff_model.eval()
            checkpoint = torch.load(sufficiency_model_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            self.suff_model.load_state_dict(checkpoint['model_state_dict'])
            self.remote_suff = False
 
        # I: main reasoning task instruction. Tells the model how to signal a search.
        self.TASK_INSTRUCTION = (
            """You are a reasoning assistant with the ability to perform web searches to help you answer the user's question accurately. You have special tools:
            To perform a search: write <|begin_search_query|> your query here <|end_search_query|>. Then, the system will search and analyze relevant web pages, then provide you with helpful information in the format <|begin_search_result|> ...search results... <|end_search_result|>.
            You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {search_limit}.
            Once you have all the information you need, continue your reasoning.
            Example:
            Question: “...”
            Assistant thinking steps:
            - I might need to look up details about ...
            Assistant: <|begin_search_query|>...<|end_search_query|>
            (System returns processed information from relevant web pages)
            Assistant continues reasoning with the new information...
            Remember:
            - Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.
            - When done searching, continue your reasoning. """
        )
        
 
        # Idocs: instruction for the Reason-in-Documents sub-step.
        self.REASON_IN_DOCS_INSTRUCTION = (
            """Task Instruction:
            You are tasked with reading and analyzing web pages based on the following inputs: Previous Reasoning Steps, Current Search Query, and Searched Web Pages. 
            Your objective is to extract relevant and helpful information for Current Search Query from the Searched Web Pages and seamlessly integrate this information into the Previous Reasoning Steps to continue reasoning for the original question.
            Guidelines:
            1. Analyze the Searched Web Pages:
            - Carefully review the content of each searched web page.
            - Identify factual information that is relevant to the Current Search Query and can aid in the reasoning process for the original question.
            2. Extract Relevant Information:
            - Select the information from the Searched Web Pages that directly contributes to advancing the Previous Reasoning Steps.
            - Ensure that the extracted information is accurate and relevant.
            3. Output Format:
            - If the web pages provide helpful information for current search query: Present the information beginning with "Final Information" as shown below.
            Final Information
            [Helpful information]
            If the web pages do not provide any helpful information for current search query: Output the following text.
            Final Information
            No helpful information found.
            """
        )
 
        # Main reasoning prompt: instruction + one demonstration + the live question.
        self.main_prompt_prefix = f"{self.TASK_INSTRUCTION}\n\nPlease answer the following question. You should provide your final answer in the format <|boxed|>YOUR_ANSWER<|boxed|>. Q: {{question}}\nA: {{suffix}}"

        # Reason-in-documents prompt template, reusing the existing reference formatter.
        self.reason_template = PromptTemplate(
            system_prompt=self.REASON_IN_DOCS_INSTRUCTION,
            user_prompt="""Inputs:
            - Original Query:
            {query}
            - Previous Reasoning Steps:
            {prev_reasoning}
            - Current Search Query:
            {search_query}
            - Searched Web Pages:
            {document}
            Now you should analyze each web page and find helpful information based on the current search query "{search_query}" and previous reasoning steps.""",
        )
 
    def extract_answer(self, full_answer):
        answer = re.search(r"boxed|>(.+)<|boxed|>", full_answer, flags=re.IGNORECASE)
        if answer:
            final_answer = answer.groups()[0].strip()
            return re.sub(r'[^\w\s]', '', final_answer)
        return None
 
    def extract_search_query(self, text):
        """Extracts the query from a trailing <|begin_search_query|>...<|end_search_query|> span."""
        match = re.search(
            re.escape(self.BEGIN_SEARCH_QUERY) + r"(.+?)" + re.escape(self.END_SEARCH_QUERY),
            text, flags=re.DOTALL
        )
        if match:
            return match.group(1).strip()
        return None
 
    def _process_docs(self, new_docs):
        context = []
        for doc in new_docs:
            context_separated = str(doc["contents"]).split('\n', 1)
            title = context_separated[0]
            text = context_separated[1]
            context.append({"title": title, "text": text})
        return context
        
 
    def _reason_in_documents(self, query, search_query, reasoning_so_far, docs):
        """
        Line 15-20 of Algorithm 1: given the docs retrieved for `search_query`,
        distill them into a knowledge-injected reasoning snippet (rfinal).
        """
        prompt_messages = self.reason_template.format_prompt({"query": query, "prev_reasoning": reasoning_so_far, "search_query": search_query, "document": docs})
        # format_prompt returns a messages-style structure per IRCoT's convention;
        # extract the raw text for the completion path, reuse as-is for chat.
        if self.mode == "chat":
            content, in_tok, out_tok = self.inference_manager.query_model(prompt_messages, self.model)
        else:
            content, in_tok, out_tok = self.inference_manager.complete(prompt_messages[0]["content"], self.model)
        return content.strip(), in_tok, out_tok
 
    def process_query(self, query, k=5, max_iter=10):
        seq = self.main_prompt_prefix.format(question=query, search_limit=max_iter, suffix="")
        complete_docs = []
        input_tokens_used, output_tokens_generated, time_used = 0, 0, 0
        turns = 0
 
        for i in range(max_iter):
            turns = i + 1
            try:
                start = time.time()
                if self.mode == "chat":
                    messages = [{"role": "user", "content": seq}]
                    generated, input_tokens_used, output_tokens_generated = self.inference_manager.query_model(messages, self.model)
                else:  # completion
                    return self.inference_manager.complete(prompt_text, self.model)
                time_used += time.time() - start
                input_tokens_used += input_tokens_used
                output_tokens_generated += output_tokens_generated
            except Exception as e:
                print(f"Exception during main generation: {e}")
                continue
 
            seq += generated
 
            search_query = self.extract_search_query(generated)
            if search_query is not None:
                # Line 9-12: retrieve, build Reason-in-Documents input
                try:
                    _, _, returning_docs = self.retriever.retrieve(search_query, k=k)
                    context = self._process_docs(returning_docs)
                    for doc in context:
                        if doc not in complete_docs:
                            complete_docs.append(doc)
                    if self.reranker is not None:
                        complete_docs, _ = self.reranker.rerank(query, complete_docs)
 
                    start = time.time()
                    docs_for_graph = self._textual_docs(complete_docs)
                    # docs_for_graph.append(previous_answer)
                    if self.remote_suff:
                        logits, suff_tokens_used = self.suff_model(query, docs_for_graph) 
                    else:
                        with torch.no_grad():
                            logits, suff_tokens_used = self.suff_model(GraphDataset([(query, docs_for_graph, 0)])) # Placeholder y) 

                    pred = logits.argmax(1)
                    
                    if pred == 1: # If it is sufficient
                        full_answer += " The context is sufficient to answer the question. <|boxed|>" # Induces the model to generate the final answer
                        ever_sufficient = turns
                    # otherwise, just keep searching
                    bert_time = time.time() - start

                    start = time.time()
                    rfinal, r_in_tok, r_out_tok = self._reason_in_documents(
                        query, search_query, seq, complete_docs
                    )
                    time_used += time.time() - start
                    input_tokens_used += r_in_tok
                    output_tokens_generated += r_out_tok
                except Exception as e:
                    print(f"Exception during reason-in-documents: {e}")
                    rfinal = "No relevant information was found."
 
                # Line 21: splice the distilled knowledge back into the sequence
                seq += f"\n{self.BEGIN_SEARCH_RESULT}{rfinal}{self.END_SEARCH_RESULT}\n"
                continue  # keep generating in the next iteration
 
            # No search requested this turn -> treat as a finished sequence (line 13-14)
            extracted_answer = self.extract_answer(seq)
            if extracted_answer is not None:
                return extracted_answer, complete_docs, input_tokens_used, output_tokens_generated, time_used, turns
            # Model stopped without a search tag and without a final-answer marker;
            # nothing left to trigger further generation, so end here.
            return seq, complete_docs, input_tokens_used, output_tokens_generated, time_used, turns, ever_sufficient
 
        # max_iter exhausted mid-search-loop
        return seq, complete_docs, input_tokens_used, output_tokens_generated, time_used, turns, ever_sufficient
 
    def process_dataset(self, df: pd.DataFrame, query_col="query", answer_col="answer", docs_col="references",
                         doc_reference_field="title", max_iter=10, k=5,
                         metrics: list[Metric] = [LanguageExactMatch()], display=False,
                         intermediate_results_path=None):
        if doc_reference_field not in ["title", "text"]:
            raise ValueError("Parameter \"doc_reference_field\" must be either \"title\" or \"text\"")
        results = {"inference_time": [], "input_tokens_used": [], "output_tokens_generated": [], "turns": [], "sufficient": [],
                   "answer": [], "answer_references": [], "docs": []}
        total_instances = df.shape[0]
        iterator = tqdm(df.iterrows(), total=total_instances, unit="instances") if display else df.iterrows()
        for i, row in iterator:
            try:
                key = None
                raw_refs = row[answer_col]
                if isinstance(raw_refs, list):
                    refs = [re.sub(r'[^\w\s]', '', str(r)) for r in raw_refs]
                else:
                    refs = re.sub(r'[^\w\s]', '', str(raw_refs))
                answer, complete_docs, input_tokens_used, output_tokens_generated, inference_time, turns, ever_sufficient = self.process_query(row[query_col], k=k, max_iter=max_iter)
 
                for metric in metrics:
                    key = metric.name
                    if results.get(key) is None:
                        results[key] = []
                    if isinstance(metric, LanguageMetric):
                        result = metric.compute(answer, refs)
                    elif isinstance(metric, RetrievalMetric):
                        retrieved_docs = []
                        for doc in complete_docs:
                            doc_title = doc[doc_reference_field]
                            if doc_title not in retrieved_docs:
                                retrieved_docs.append(doc_title)
                        result = metric.compute(list(retrieved_docs), row[docs_col])
                    results[key].append(result)
 
                results["inference_time"].append(inference_time)
                results["input_tokens_used"].append(input_tokens_used)
                results["output_tokens_generated"].append(output_tokens_generated)
                results["turns"].append(turns)
                results["sufficient"].append(ever_sufficient)
                results["answer"].append(answer)
                results["answer_references"].append(refs)
                results["docs"].append(complete_docs)
 
                if intermediate_results_path is not None:
                    out_df = pd.DataFrame(results)
                    out_df = out_df.replace(r"\n", r"\\n", regex=True)
                    out_df.to_csv(intermediate_results_path, encoding="utf-8", index=False)
                    del out_df
            except Exception as e:
                print(f"Exception in instance {i}: {e}. Current metric: {key if key is not None else 'Unknown'}")
                continue
 
        return results