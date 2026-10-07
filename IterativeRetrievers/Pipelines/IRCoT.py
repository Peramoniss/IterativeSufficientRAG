import os
from SufficiencyModel.modelStructures import SemanticOnlyBERT, TextDataset
from Inference.prompts import IRCoTPromptTemplate
from Inference.requestManagers import RequestManager
from Retriever.retriever import Retriever
from Reranker.reranker import Reranker
from Metrics.metrics import LanguageMetric, LanguageExactMatch, LanguageF1, Metric, RecallAtK, RetrievalMetric
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import time
import traceback
import requests
from nltk.tokenize import sent_tokenize

class IRCoT:
    def __init__(self, retriever: Retriever, inference_manager: RequestManager, model: str, reranker: Reranker = None, mode:str="chat", load_if_exist=False):
        self.retriever = retriever
        self.reranker = reranker
        self.inference_manager = inference_manager
        self.model = model
        if mode.lower() not in ["chat", "completion"]:
            raise ValueError("Mode must be either \"chat\" or \"completion\". We suggest using chat for bigger models and completion for open, quantized models with low parameter count")
        self.mode = mode.lower()
        self.load_if_exist = load_if_exist

        IRCOT_INSTRUCTION = (
            "You serve as an intelligent assistant, adept at facilitating users through complex, multi-hop reasoning across multiple documents. "
            "This task is illustrated through demonstrations, each consisting of a document set paired with a question and its multi-hop reasoning thoughts that lead to the answer. "
            'The process is iterative, so your task is to generate a single concise sentence for the current step that will help to retrieve more supporting documents with information about the question. DO NOT generate the whole thinking process, only a SINGLE sentence! If the thoughts and the supporting documents lead you to believe you can give the final answer, start your answer with "So the answer is:". DO NOT GIVE THE FINAL ANSWER unless the wikipedia articles can sustain that information.'
        )

        # IRCOT_EXAMPLE = """Wikipedia Title: Kurram Garhi
        # Kurram Garhi is a small village located near the city of Bannu, which is the part of Khyber Pakhtunkhwa province of Pakistan. Its population is approximately 35000. Barren hills are near this village. This village is on the border of Kurram Agency. Other nearby villages are Peppal, Surwangi and Amandi Kala.

        # Wikipedia Title: 2001–02 UEFA Champions League second group stage
        # Eight winners and eight runners- up from the first group stage were drawn into four groups of four teams, each containing two group winners and two runners- up. Teams from the same country or from the same first round group could not be drawn together. The top two teams in each group advanced to the quarter- finals.

        # Wikipedia Title: Satellite tournament
        # A satellite tournament is either a minor tournament or event on a competitive sporting tour or one of a group of such tournaments that form a series played in the same country or region.

        # Wikipedia Title: Trojkrsti
        # Trojkrsti is a village in Municipality of Prilep, Republic of Macedonia.

        # Wikipedia Title: Telephone numbers in Ascension Island
        # Country Code:+ 247< br> International Call Prefix: 00 Ascension Island does not share the same country code( +290) with the rest of St Helena.

        # Q: Are both Kurram Garhi and Trojkrsti located in the same country?
        # A: Kurram Garhi is located in the country of Pakistan. Trojkrsti is located in the country of Republic of Macedonia. Thus, they are not in the same country. So the answer is: no."""

        IRCOT_EXAMPLE = """
        Q: When did the director of the film Hypocrite die?
        A: The film Hypocrite was directed by Miguel Morayta. Miguel Morayta died on 19 June 2013. So the answer is: 19 June 2013.

        Q: Do the director of the film Coolie No. 1 (1995 film) and the director of the film The Sensational Trial have the same nationality?   
        A: Coolie No. 1 (1995 film) was directed by David Dhawan. The Sensational Trial was directed by Karl Freund. David Dhawan's nationality is India. Karl Freund's nationality is Germany. Thus, they do not have the same nationality. So the answer is: no.   

        Q: Are both Kurram Garhi and Trojkrsti located in the same country?   
        A: Kurram Garhi is located in the country of Pakistan. Trojkrsti is located in the country of Republic of Macedonia. Thus, they are not in the same country. So the answer is: no.   

        Q: Who was born first out of Martin Hodge and Ivania Martinich?
        A: Martin Hodge was born on 4 February 1959. Ivania Martinich was born on 25 July 1995. Thus, Martin Hodge was born first. So the answer is: Martin Hodge.

        Q: Which film came out first, The Night Of Tricks or The Genealogy?
        A: The Night of Tricks was published in the year 1939. The Genealogy was published in the year 1979. Thus, The Night of Tricks came out first. So the answer is: The Night Of Tricks.

        Q: When did the director of the film Laughter In Hell die?
        A: The film Laughter In Hell was directed by Edward L. Cahn. Edward L. Cahn died on August 25, 1963. So the answer is: August 25, 1963.

        Q: Which film has the director who died later, The Gal Who Took the West or Twenty Plus Two?
        A: The film Twenty Plus Two was directed by Joseph M. Newman. The Gal Who Took the West was directed by Frederick de Cordova. Joseph M. Newman died on January 23, 2006. Fred de Cordova died on September 15, 2001. Thus, the director who died later is from Twenty Plus Two. So the answer is: Twenty Plus Two.
        """
        
        base_user_prompt = f"{IRCOT_INSTRUCTION}\n\n{IRCOT_EXAMPLE}\n\n"
        self.template = IRCoTPromptTemplate(
            system_prompt="", #base_user_prompt OR IRCOT_INSTRUCTION OR ""
            user_prompt= IRCOT_EXAMPLE+"{reference}Q: Answer the following question by concisely reasoning step-by-step (follow the examples above). {question}\nA: {suffix}", #base_user_prompt+ OR "Answer the following question..."
            reference_template="Wikipedia Title: {title}\n{text}\n\n",
            reference_label=True
        )

    def extract_answer(self, full_answer):
        for pattern in [r"answer is:?\s*(.+)"]: #, r"answer\s*i?s?:?\s*(.+)"
            answer = re.search(pattern, full_answer, flags=re.IGNORECASE)
            if answer:
                final_answer = answer.groups()[0].strip()
                return re.sub(r'[^\w\s]', '', final_answer)

    def _process_docs(self, new_docs):
        context = []
        for doc in new_docs:
            # Docs must be re-formatted before sending for the formatter
            context_separated = str(doc["contents"]).split('\n', 1)
            title = context_separated[0]
            text = context_separated[1]
            context.append({"title": title, "text": text})

        return context    

    def process_query(self, query, k=5, max_iter=8, log=True):
        full_answer = ""
        current_answer = query
        previous_answer = ""
        complete_docs = []
        input_tokens_used, output_tokens_generated, time_used = 0, 0, 0
        first_turn_docs = None
        for i in range(max_iter):
            _, _, returning_docs = self.retriever.retrieve(current_answer, k=k)
            # print(returning_docs)
            context = self._process_docs(returning_docs)
            if i == 0:
                first_turn_docs = context.copy()
            for doc in context: # Only add to complete_docs new documents
                if doc not in complete_docs and len(complete_docs) < 15: # Original article caps at 15 docs
                    complete_docs.append(doc)
            if self.reranker is not None:
                complete_docs, _ = self.reranker.rerank(query, complete_docs)

            # suffix = "" if full_answer == "" else "Previous partial conclusions: "+full_answer
            suffix = full_answer
            
            messages = self.template.format_prompt(query, retrieved_documents=complete_docs, suffix=suffix)
            # messages.append({"role": "assistant", "content": full_answer})
            # if i == 0:
            #     print(messages)
            try:
                start = time.time()
                if self.mode == "chat":
                    previous_answer, input_tok, out_tok = self.inference_manager.query_model(messages, self.model)
                else: #completion
                    previous_answer, input_tok, out_tok = self.inference_manager.complete(messages[-1]["content"], self.model)
                    # raise NotImplementedError("TODO: see how completion API works to extract the same information")
                inference_time = time.time() - start
                time_used += inference_time
                input_tokens_used += input_tok
                output_tokens_generated += out_tok
            except Exception as e:
                print(f"Exception: {e}")
                traceback.print_exc()
                continue
            # full_answer += previous_answer
            included = False
            for potential_answer in sent_tokenize(previous_answer): #previous_answer.split(".")
                if potential_answer.lower().strip() not in full_answer.lower().strip(): # If this reasoning step is not already in the answer
                    current_answer = potential_answer.strip()+" "#". "
                    included = True
                    break
            if not included: 
                current_answer = "So the answer is: " # Even if full_answer has doubled So the answer is:, the extraction happen on previous answer, not in full_answer, so it is fine and does not unformat it
            
            full_answer += current_answer

            # print(messages)
            # print(full_answer)
            # print(previous_answer)
            # print("="*30)
            if included: # Only extract answer if found new reason. Otherwise, no reason to look on the same answers. Carry forward the None of the previous round
                extracted_answer = self.extract_answer(current_answer)  
            if log:
                print(full_answer)
                print("-"*20)
            if extracted_answer != None:
                return extracted_answer, first_turn_docs, complete_docs, input_tokens_used, output_tokens_generated, time_used, i+1

        # print(full_answer)
        extracted_answer = self.extract_answer(previous_answer) # Analyze the complete last answer
        return_answer = extracted_answer if extracted_answer is not None else previous_answer
        return return_answer, first_turn_docs, complete_docs, input_tokens_used, output_tokens_generated, time_used, i+1

    def process_dataset(self, df:pd.DataFrame, query_col="query", answer_col="answer", docs_col="references", doc_reference_field="title", max_iter=8, k=5, metrics:list[Metric] = [LanguageExactMatch()], display=False, intermediate_results_path=None):
        if doc_reference_field not in ["title", "text"]:
            raise ValueError("Parameter \"doc_reference_field\" must be either \"title\" or \"text\"")
        results = {}
        results["inference_time"] = []
        results["input_tokens_used"] = []
        results["output_tokens_generated"] = []
        results["turns"] = []
        results["answer"], results["answer_references"], results["docs"] = [], [], []
        total_instances = df.shape[0]
        if intermediate_results_path != None and not os.path.exists(intermediate_results_path): # If the file doesn't exist yet
            self.load_if_exist = False
        if self.load_if_exist: # If still able to load if exist
            existing_df = pd.read_csv(intermediate_results_path)
            already_processed = existing_df.shape[0]
            results = existing_df.to_dict(orient="list")
            del existing_df
        iterator = tqdm(df.iterrows(), total=total_instances, unit="instances") if display else df.iterrows()
        for i, row in iterator:
            if self.load_if_exist and i < already_processed:
                continue
            try:
                key = None
                refs = row[answer_col] if ( isinstance(row[answer_col], str) or isinstance(row[answer_col], list) ) else str(row[answer_col])
                answer, first_turn_docs, complete_docs, input_tokens_used, output_tokens_generated, inference_time, turns = self.process_query(row[query_col], k=k, max_iter=max_iter)
                for metric in metrics:
                    key = metric.name
                    metric_list = results.get(key, None)
                    if metric_list == None:
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
                results["output_tokens_generated"].append(output_tokens_generated)
                results["turns"].append(turns)
                results["answer"].append(answer)
                results["answer_references"].append(refs)
                results["docs"].append(complete_docs)

                if intermediate_results_path != None:
                    df = pd.DataFrame(results)
                    df = df.replace(r"\n", r"\\n", regex=True)
                    df.to_csv(intermediate_results_path, encoding="utf-8", index=False)
                    del df
            except Exception as e:
                print(f"Exception in instance {i}: {e}. Current metric: {key if key is not None else "Unknown"}. Current pred: {answer}; Current references {refs}")
                continue

        return results

class RemoteSufficiency:
    def __init__(self, host, dataset:str, llm:bool = False, model:str = None, max_attempts : int = 5):
        self.host = host
        self.max_attempts = max_attempts
        self.llm = llm
        self.model = model
        self.dataset = dataset
        self.headers = {
            "ngrok-skip-browser-warning": "69420"
        }
        self.prompt = """
        You are an expert LLM evaluator that excels at evaluating a QUESTION and REFERENCES.
        Consider the following criteria:
        Sufficient Context: 1 IF the CONTEXT is sufficient to infer the answer to the question and 0 IF the CONTEXT cannot be used to infer the answer to the question.
        First, output a list of step-by-step questions that would be used to arrive at a label for the criteria. Make sure to include questions about assumptions implicit in the QUESTION. Include questions about any mathematical calculations or arithmetic that would be required.
        Next, answer each of the questions. Make sure to work step by step through any required mathematical calculations or arithmetic. 
        Finally, use these answers to evaluate the criteria and output the ###EVALUATION (JSON).
        EXAMPLE:
        ### QUESTION
        In which year did the publisher of Roald Dahl's Guide to Railway Safety cease to exist?
        ### References
        Roald Dahl's Guide to Railway Safety was published in 1991 by the British Railways Board. The British Railways Board had asked Roald Dahl to write the text of the booklet, and Quentin Blake to illustrate it, to help young people enjoy using the railways safely. The British Railways Board (BRB) was a nationalised industry in the United Kingdom that operated from 1963 to 2001. Until 1997 it was responsible for most railway services in Great Britain, trading under the brand name British Railways and, from 1965, British Rail. It did not operate railways in Northern Ireland, where railways were the responsibility of the Government of Northern Ireland.
        ### EXPLANATION
        The context mentions that Roald Dahl's Guide to Railway Safety was published by the British Railways Board. It also states that the British Railways Board operated from 1963 to 2001, meaning the year it ceased to exist was 2001. Therefore, the context does provide a precise answer to the question.
        ### JSON
        {{"Sufficient Context": 1}}
        ### QUESTION
        {question}
        ### REFERENCES
        {context}
        """
    
    def __call__(self, query:str, docs:list[str]):
        if self.llm:
            attempts = 0
            while attempts < self.max_attempts:
                try:
                    messages = [{"role": "user", "content": self.prompt.format(question=query, context=docs)}]
                    answer, input_tok, out_tok = self.host.query_model(messages, self.model)
                    tokens_used = input_tok+out_tok
                    if 'Sufficient Context": 1' in str(answer):
                        return torch.tensor([[0, 1]]), tokens_used
                    return torch.tensor([[1, 0]]), tokens_used
                except Exception as e:
                    print(e)
                    print("An error occurred:")
                    print_print_exc()()

                    # Optional: Get traceback as a string for logging
                    tb_str = print_format_exc()()
                    print("\nTraceback string for logging:\n", tb_str)
                    attempts += 1
                    continue

            if attempts >= self.max_attempts:
                raise ConnectionError("Maximum attempts reached")
        
        # If it's a BERT hosting server instead
        payload: dict = {
            "query": query,
            "password": "kTOIe4wQQn 8wQ1XH2jS1 MjkZKOMab6 Rp3eMVT34D 1DitCzvPEm",
            "docs": docs,
        }

        status = 0
        attempts = 0 
        while status != 200 and attempts < self.max_attempts:
            response = requests.post(f"{self.host}/sufficiency/{self.dataset}", json=payload, headers=self.headers)
            status = response.status_code
            attempts += 1
        
        if attempts >= self.max_attempts:
            raise ConnectionError("Maximum attempts reached")

        response_dict = response.json()
        return torch.tensor(response_dict["logits"]), response_dict["suff_tokens_used"]

class IRCoTWithSufficiency:
    def __init__(self, retriever: Retriever, inference_manager: RequestManager, sufficiency_model_path: str | RequestManager, generator_model: str, dataset:str, reranker:Reranker = None, mode:str = "chat", load_if_exist=False, threshold = None):
        self.retriever = retriever
        self.reranker = reranker
        self.inference_manager = inference_manager
        self.model = generator_model
        if mode.lower() not in ["chat", "completion"]:
            raise ValueError("Mode must be either \"chat\" or \"completion\". We suggest using chat for bigger models and completion for open, quantized models with low parameter count")
        self.mode = mode.lower()
        self.load_if_exist = load_if_exist
        self.threshold = threshold
        if isinstance(sufficiency_model_path, RequestManager):
            self.suff_model = RemoteSufficiency(sufficiency_model_path, dataset, llm=True, model=generator_model)       
            self.remote_suff = True
        elif "http" in sufficiency_model_path:
            self.suff_model = RemoteSufficiency(sufficiency_model_path, dataset)       
            self.remote_suff = True
        else:
            self.suff_model =  SemanticOnlyBERT("bert-base-uncased", num_classes=2, dropout_rate=0.2, freeze_bert_layers=6)
            self.suff_model.eval()
            checkpoint = torch.load(sufficiency_model_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            self.suff_model.load_state_dict(checkpoint['model_state_dict'])
            self.remote_suff = False

        # IRCOT_INSTRUCTION = (
        #     "You serve as an intelligent assistant, adept at facilitating users through complex, multi-hop reasoning across multiple documents. "
        #     "This task is illustrated through demonstrations, each consisting of a document set paired with a relevant question and its multi-hop reasoning thoughts. "
        #     'Your task is to generate one thought for current step, DON\'T generate the whole thoughts at once! If you reach what you believe to be the final step, start with "So the answer is:".'
        # )

        IRCOT_INSTRUCTION = (
            "You serve as an intelligent assistant, adept at facilitating users through complex, multi-hop reasoning across multiple documents. "
            "This task is illustrated through demonstrations, each consisting of a document set paired with a question and its multi-hop reasoning thoughts that lead to the answer. "
            'The process is iterative, so your task is to generate a single concise sentence for the current step that will help to retrieve more supporting documents with information about the question. DO NOT generate the whole thinking process, only a SINGLE sentence! If the thoughts and the supporting documents lead you to believe you can give the final answer, start your answer with "So the answer is:". DO NOT GIVE THE FINAL ANSWER unless the wikipedia articles can sustain that information.'
        )

        # IRCOT_EXAMPLE = """Wikipedia Title: Kurram Garhi
        # Kurram Garhi is a small village located near the city of Bannu, which is the part of Khyber Pakhtunkhwa province of Pakistan. Its population is approximately 35000. Barren hills are near this village. This village is on the border of Kurram Agency. Other nearby villages are Peppal, Surwangi and Amandi Kala.

        # Wikipedia Title: 2001–02 UEFA Champions League second group stage
        # Eight winners and eight runners- up from the first group stage were drawn into four groups of four teams, each containing two group winners and two runners- up. Teams from the same country or from the same first round group could not be drawn together. The top two teams in each group advanced to the quarter- finals.

        # Wikipedia Title: Satellite tournament
        # A satellite tournament is either a minor tournament or event on a competitive sporting tour or one of a group of such tournaments that form a series played in the same country or region.

        # Wikipedia Title: Trojkrsti
        # Trojkrsti is a village in Municipality of Prilep, Republic of Macedonia.

        # Wikipedia Title: Telephone numbers in Ascension Island
        # Country Code:+ 247< br> International Call Prefix: 00 Ascension Island does not share the same country code( +290) with the rest of St Helena.

        # Q: Are both Kurram Garhi and Trojkrsti located in the same country?
        # A: Kurram Garhi is located in the country of Pakistan. Trojkrsti is located in the country of Republic of Macedonia. Thus, they are not in the same country. So the answer is: no."""

        IRCOT_EXAMPLE = """
        Q: When did the director of the film Hypocrite die?
        A: The film Hypocrite was directed by Miguel Morayta. Miguel Morayta died on 19 June 2013. So the answer is: 19 June 2013.

        Q: Do the director of the film Coolie No. 1 (1995 film) and the director of the film The Sensational Trial have the same nationality?   
        A: Coolie No. 1 (1995 film) was directed by David Dhawan. The Sensational Trial was directed by Karl Freund. David Dhawan's nationality is India. Karl Freund's nationality is Germany. Thus, they do not have the same nationality. So the answer is: no.   

        Q: Are both Kurram Garhi and Trojkrsti located in the same country?   
        A: Kurram Garhi is located in the country of Pakistan. Trojkrsti is located in the country of Republic of Macedonia. Thus, they are not in the same country. So the answer is: no.   

        Q: Who was born first out of Martin Hodge and Ivania Martinich?
        A: Martin Hodge was born on 4 February 1959. Ivania Martinich was born on 25 July 1995. Thus, Martin Hodge was born first. So the answer is: Martin Hodge.

        Q: Which film came out first, The Night Of Tricks or The Genealogy?
        A: The Night of Tricks was published in the year 1939. The Genealogy was published in the year 1979. Thus, The Night of Tricks came out first. So the answer is: The Night Of Tricks.

        Q: When did the director of the film Laughter In Hell die?
        A: The film Laughter In Hell was directed by Edward L. Cahn. Edward L. Cahn died on August 25, 1963. So the answer is: August 25, 1963.

        Q: Which film has the director who died later, The Gal Who Took the West or Twenty Plus Two?
        A: The film Twenty Plus Two was directed by Joseph M. Newman. The Gal Who Took the West was directed by Frederick de Cordova. Joseph M. Newman died on January 23, 2006. Fred de Cordova died on September 15, 2001. Thus, the director who died later is from Twenty Plus Two. So the answer is: Twenty Plus Two.
        """
        
        self.template = IRCoTPromptTemplate(
            system_prompt="", #base_user_prompt OR IRCOT_INSTRUCTION OR ""
            user_prompt= IRCOT_EXAMPLE+"{reference}Q: Answer the following question by concisely reasoning step-by-step (follow the examples above). {question}\nA: {suffix}", #base_user_prompt+ OR "Complete the answer to the following question..."
            reference_template="Wikipedia Title: {title}\n{text}\n\n",
            reference_label=True
        )

    def extract_answer(self, full_answer):
        for pattern in [r"answer is:?\s*(.+)"]: #, r"answer\s*i?s?:?\s*(.+)"
            answer = re.search(pattern, full_answer, flags=re.IGNORECASE)
            if answer:
                final_answer = answer.groups()[0].strip()
                return re.sub(r'[^\w\s]', '', final_answer)
        
        return None

    def _process_docs(self, new_docs):
        context = []
        for doc in new_docs:
            # Docs must be re-formatted before sending for the formatter
            context_separated = str(doc["contents"]).split('\n', 1)
            title = context_separated[0]
            text = context_separated[1]
            context.append({"title": title, "text": text})

        return context

    def _textual_docs(self, docs):
        new_docs = []
        for doc in docs:
            # Docs must be re-formatted before sending for the formatter
            context_separated = str(doc["text"])
            new_docs.append(context_separated)

        return new_docs

    def process_query(self, query, refs, k=5, max_iter=8, log=True):
        full_answer = ""
        previous_answer = ""
        current_answer = query
        recall = RecallAtK(15)
        turn_when_sufficient = -1
        total_sufficiency_tokens_used = 0
        suff_tokens_used = 0
        suff_time = 0
        complete_docs = []
        first_turn_docs = None
        input_tokens_used, output_tokens_generated, time_used = 0,0,0
        ever_sufficient = -1
        productive_turns = 0
        for i in range(max_iter):
            try:
                _, _, returning_docs = self.retriever.retrieve(current_answer, k=k)
                context = self._process_docs(returning_docs)
                if i == 0:
                    first_turn_docs = context.copy()
                for doc in context:
                    if doc not in complete_docs and len(complete_docs) < 15: # Original article caps at 15 docs - Add docs with the same tiel
                        complete_docs.append(doc)

                retrieved_docs = []
                for doc in complete_docs:
                    doc_title = doc["title"]
                    if doc_title not in retrieved_docs: # Does not add docs with the same title, since this is for metric computation only
                        retrieved_docs.append(doc_title)

                # print(retrieved_docs, refs)
                curr_recall = recall.compute(list(retrieved_docs), refs)
                if turn_when_sufficient == -1 and curr_recall == 1.0:
                    turn_when_sufficient = i+1
                if self.reranker is not None:
                    complete_docs, _ = self.reranker.rerank(query, complete_docs)

                # if i != 0:
                start = time.time()
                sending_docs = complete_docs[:k] if self.reranker != None else complete_docs
                docs_for_suff = self._textual_docs(sending_docs)
                if self.remote_suff:
                    logits, suff_tokens_used = self.suff_model(query, docs_for_suff) 
                else:
                    with torch.no_grad():
                        logits, suff_tokens_used = self.suff_model(TextDataset([(query, docs_for_suff, 0)])) # Placeholder y) 
                        # logits, suff_tokens_used = self.suff_model(GraphDataset([(query, docs_for_suff, 0)])) # Placeholder y) 
                
                if self.threshold == None:
                    pred = logits.argmax(1)
                    pred_bool = (pred == 1)
                else:
                    dist = logits.softmax(1)
                    pred = dist.argmax(1)
                    if log:
                        print(dist)
                    if pred == 1 and dist[0][1] > self.threshold:
                        pred_bool = True
                    else:
                        pred_bool = False
                
                if pred_bool and ever_sufficient == -1: # If it is sufficient
                    # print("Found sufficiency")
                    # print(logits)
                    ever_sufficient = i+1
                # if ever_sufficient != -1 and i != 0: # forces the first round of reasoning even if the first retrieval is sufficient; otherwise, the model is less precise in its answer
                #     full_answer += " There is sufficient context to answer the question. " # Induces the model to generate the final answer
                # otherwise, just keep searching Let's review the information and answer the question.
                suff_time = time.time() - start

                # suffix = "" if full_answer == "" else "Previous partial conclusions: "+full_answer
                suffix = full_answer
                # suffix = ""
                messages = self.template.format_prompt(query, retrieved_documents=complete_docs, suffix=suffix)
                # if i == 0:
                #     print(messages)
                # messages.append({"role": "assistant", "content": full_answer})
                start = time.time()
                if self.mode == "chat":
                    previous_answer, input_tok, out_tok = self.inference_manager.query_model(messages, self.model)
                else: #completion
                    raise NotImplementedError("TODO: see how completion API works to extract the same information")
                inference_time = time.time() - start

                input_tokens_used += input_tok
                total_sufficiency_tokens_used += suff_tokens_used
                output_tokens_generated += out_tok
                time_used += (inference_time + suff_time)

                included = False
                for potential_answer in sent_tokenize(previous_answer): #previous_answer.split(".") # Sometimes loses a turn to just place a single word because of name like Jason C. Mayer.
                    if potential_answer.lower().strip() not in full_answer.lower().strip(): # If this reasoning step is not already in the answer
                        current_answer = potential_answer.strip().strip()+" "#". "
                        included = True
                        break
                if not included: # If no new information, restate something as in the original article
                    current_answer = "So the answer is: " # Even if full_answer has doubled So the answer is:, the extraction happen on previous answer, not in full_answer, so it is fine and does not unformat it# When the model has nothing new to reason over, it will induce the answer. The extraction, however, will not use this
                full_answer += current_answer
                if (ever_sufficient != -1 and i >= ever_sufficient):  # ever_sufficient is i+1 of when it is found. Therefore, we wait until the next turn to let the model reason over the sufficient context first, append it on prompt, and in the next round it is able to give the answer
                    extracted_answer = self.extract_answer(previous_answer) # If found sufficiency, can search in the whole answer because this should be sufficient
                elif included: # Doesnt't update if there's no new information. Extracted answer carries the None of the previous turn to the next
                    extracted_answer = self.extract_answer(current_answer) # If insufficient but with a new sentence, reason normally

                if log:
                    print(full_answer)
                    print("-"*20)
                if extracted_answer != None:
                    return extracted_answer, first_turn_docs, complete_docs, input_tokens_used, total_sufficiency_tokens_used, output_tokens_generated, time_used, productive_turns, ever_sufficient, turn_when_sufficient
                productive_turns += 1
            except Exception as e:
                print(f"Exception: {e}")
                print_print_exc()()
                continue
        # print(full_answer)
        extracted_answer = self.extract_answer(previous_answer) # Analyze the complete last answer
        return_answer = extracted_answer if extracted_answer is not None else previous_answer
        return return_answer, first_turn_docs, complete_docs, input_tokens_used, total_sufficiency_tokens_used, output_tokens_generated, time_used, productive_turns, ever_sufficient, turn_when_sufficient

    def process_dataset(self, df:pd.DataFrame, query_col="query", answer_col="answer", docs_col="references", doc_reference_field="title", max_iter=8, k=5, metrics:list[LanguageMetric] = [LanguageExactMatch()], display=False, intermediate_results_path=None):
        results = {}
        results["inference_time"], results["input_tokens_used"], results["sufficiency_tokens_used"], results["output_tokens_generated"], results["turns"], results["sufficient"], results["turn_recall_1"] = [], [], [], [], [], [], []
        results["answer"], results["answer_references"], results["docs"] = [], [], []
        total_instances = df.shape[0]
        if intermediate_results_path != None and not os.path.exists(intermediate_results_path): # If the file doesn't exist yet
            self.load_if_exist = False
        if self.load_if_exist: # If still able to load if exist
            existing_df = pd.read_csv(intermediate_results_path)
            already_processed = existing_df.shape[0]
            results = existing_df.to_dict(orient="list")
            del existing_df
        iterator = tqdm(df.iterrows(), total=total_instances, unit="instances") if display else df.iterrows()
        for i, row in iterator:
            if self.load_if_exist and i < already_processed:
                continue
            try:
                key = None
                ans_refs = row[answer_col]
                refs = row[docs_col]
                answer, first_turn_docs, complete_docs, input_tokens_used, sufficiency_tokens_used, output_tokens_generated, inference_time, turns, ever_sufficient, turn_when_sufficient = self.process_query(row[query_col], refs, k=k, max_iter=max_iter)
                # print(f"Q: {row[query_col]}; A: {answer}")
                for metric in metrics:
                    key = metric.name
                    metric_list = results.get(key, None)
                    if metric_list == None:
                        results[key] = []

                    if isinstance(metric, LanguageMetric):
                        result = metric.compute(answer, ans_refs)
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
                results["turn_recall_1"].append(turn_when_sufficient)
                results["answer"].append(answer)
                results["answer_references"].append(ans_refs)                   
                results["docs"].append(complete_docs) 

                if intermediate_results_path != None:
                    df = pd.DataFrame(results)
                    df = df.replace(r"\n", r"\\n", regex=True)
                    df.to_csv(intermediate_results_path, encoding="utf-8", index=False)
                    del df
            except Exception as e:
                print(f"Exception in instance {i}: {e}. Current metric: {key if key is not None else "Unknown"}. Current pred: {answer}; Current references {ans_refs}")
                continue
        return results


#============================================================