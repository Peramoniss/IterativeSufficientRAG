import os
# EU TE ODEIO JAVA!!!!!!!!
java_home = r"C:\Program Files\Java\jdk-27"  # Replace with your actual JDK path
jvm_path = os.path.join(java_home, "bin", "server")

os.environ["JAVA_HOME"] = java_home
os.environ["PATH"] = jvm_path + os.pathsep + os.environ.get("PATH", "")

import time
start = time.time()
import os

# Unset the environment variable pointing to the missing Netskope certificate
os.environ.pop("REQUESTS_CA_BUNDLE", None)
os.environ.pop("CURL_CA_BUNDLE", None)
os.environ.pop("SSL_CERT_FILE", None)
import pandas as pd
from Retriever.retriever import DenseRetriever, RemoteRetriever, BM25Retriever
from Reranker.reranker import Reranker, RemoteReranker
import httpx
import tracemalloc
import traceback
# import ssl

# # Disable SSL verification globally for urllib / httpx
# ssl._create_default_https_context = ssl._create_unverified_context

# # Disable SSL verification for requests / huggingface_hub
# os.environ["CURL_CA_BUNDLE"] = ""
# os.environ["HTTPX_CA_BUNDLE"] = ""

# queries = pd.read_csv("../datasets/HotpotTest.csv")
# instance = queries.iloc[4]

# # #=======JEITO 1=======

# #----RESOURCE USAGE----
# tracemalloc.start()
# #----------------------

# start = time.time()
# retriever = DenseRetriever("intfloat/e5-base-v2", None, corpus_path="../documents/hotpot_multihop_corpus.jsonl", corpus_in_memory=True)
# retriever.read_index("../index/hotpot_ivfsq8/e5_IVF1024,SQ8.index")
# retriever.index.nprobe = 20
# reranker = Reranker("BAAI/bge-reranker-v2-m3")
# print(f"Setup took {time.time()-start} s")
# start = time.time()
# distances, indexes, relevant_documents = retriever.retrieve(instance["question"])

# #----RESOURCE USAGE----
# print(f"Query took {time.time()-start} s")
# current, peak = tracemalloc.get_traced_memory()
# print(f"Memory usage: {peak / 1024**2:.2f} MB\nPeak memory usage: {peak / 1024**2:.2f} MB")
# tracemalloc.stop()
# #----------------------

# start = time.time()
# reranked_docs, reranked_scores = reranker.rerank(instance["question"], relevant_documents)
# print(f"Rerank took {time.time()-start} s")


# print(distances)
# print(indexes)
# print(relevant_documents)
# print(len(relevant_documents))
# # print(instance["evidences"])
# print(instance["references"])
# print(reranked_docs)
# print(reranked_scores)

# #=======JEITO 2=======

# #----RESOURCE USAGE----
# tracemalloc.start()
# #----------------------

# for i in range(1,6):
#     start  = time.time()
#     retriever = DenseRetriever("intfloat/e5-base-v2", None, corpus_path=f"../documents/wiki-18_part{i}.jsonl", corpus_in_memory=False)
#     retriever.read_index(f"../index/wiki_e5_ivfsq_part{i}/e5_IVF16384,SQ8.index")
#     retriever.index.nprobe = 10
#     print(f"Setup took {time.time()-start} s")
#     start = time.time()
#     distances, indexes, relevant_documents = retriever.retrieve(instance["question"], corpus_offset=4203065*(i-1))
#     #----RESOURCE USAGE----
#     print(f"Query took {time.time()-start} s")
#     current, peak = tracemalloc.get_traced_memory()
#     print(f"Memory usage: {peak / 1024**2:.2f} MB\nPeak memory usage: {peak / 1024**2:.2f} MB")
#     tracemalloc.stop()
#     tracemalloc.start()
#     #----------------------
#     print(distances)
#     print(indexes)
#     print(relevant_documents)
#     print(len(relevant_documents))

#======================================









# O ABENÇOADO
from Pipelines.IRCoT import IRCoT, IRCoTWithSufficiency
# from Pipelines.SearchO1 import SearchO1, SearchO1WithSufficiency
from Inference.requestManagers import ExternalRequestManager, OllamaRequestManager, FlanT5RequestManager
from Retriever.retriever import DenseRetriever
import Metrics.metrics as metrics
import pandas as pd
from transformers import AutoTokenizer
print(f"Levou {time.time() - start} segundos para importar as bibliotecas")
print("TESTANDO O IRCoT")
MODEL = "qwen3.5:4b" #qwen3.5:4b #"gpt-oss:20b-128k" #gpt-4o #llama3.1:8b #deepseek-r1:8b #gemma4:12b olmo-3:7b-instruct
# JUDGE_MODEL = "mistral-7b-instruct" #qwen3.5:4b llama3.1:8b #olmo-3:7b-instruct
JUDGE_MODEL = "mistral:7b-instruct" #qwen3.5:4b llama3.1:8b #olmo-3:7b-instruct
# API_URL = "https://renewed-lucid-jiffy.ngrok-free.dev"
# API_URL = "https://vbulletin-leo-potatoes-filling.trycloudflare.com"
LOAD = True
# LOAD = False

print("Configurando modelos...")
http_client = httpx.Client(verify='petrobras-ca-root.pem')
# request_manager = ExternalRequestManager(url="https://apit.petrobras.com.br/ia/texto/v1/litellm/litellm/", api_key="d37099ddd8094dfc8fe5c2ebf220532d", http_client=http_client)
request_manager = OllamaRequestManager(api_key="sk-c1d67e5489424c3a8373dd96e9377407", timeout=120) #https://ollama.k8s.inf.ufrgs.br/ollama
# request_manager.start_ollama()
# request_manager = FlanT5RequestManager()
print("Configurando retriever...")
# retriever = DenseRetriever("intfloat/e5-base-v2", None, corpus_path="../documents/hotpot_multihop_corpus.jsonl", corpus_in_memory=True)
# retriever.read_index("../index/hotpot_ivfsq8/e5_IVF1024,SQ8.index")
# retriever.index.nprobe = 20
# retriever = RemoteRetriever(host=API_URL)
retriever = BM25Retriever("../index/wiki_bm25") # Full wiki, not filtered
print("Configurando reranker...")
reranker = None
# reranker = RemoteReranker(host=API_URL)
reranker_string = "_reranker" if reranker is not None else ""

print(f"Testando o método {"com" if reranker is not None else ""} reranker...")
# common_o1 = SearchO1(retriever=retriever, inference_manager=request_manager, model=MODEL, reranker=reranker)
# suff_o1 = SearchO1WithSufficiency(retriever=retriever, inference_manager=request_manager, generator_model=MODEL, sufficiency_model_path="../models/All GAT (seed 42).pt", reranker=reranker)
tokenizer = AutoTokenizer.from_pretrained("DWDMaiMai/tiktoken_cl100k_base") # Tiktoken proxy. Alternative: meta-llama/Meta-Llama-3-8B-Instruct. Tokenizer must match the answer model
metrics_list = [metrics.LanguageExactMatch(), metrics.LanguageF1(tokenizer), metrics.LLMAsAJudge(request_manager, JUDGE_MODEL), metrics.LanguageRouge(), metrics.AveragePrecision(), metrics.AveragePrecision(5), metrics.RecallAtK(15), metrics.RecallAtK(5), metrics.PrecisionAtK(15), metrics.PrecisionAtK(5), metrics.PrecisionRecallCurve()]
HEAD = 500
THRESH = None #0.8
RETRIEVER_TRAINMENT = "BERT" # Graph
# for model in ["qwen3.5:9b"]:
# for model in ["llama3.1:8b"]:
for model in ["qwen3.5:4b"]:
    try:
        common_ircot_manager = IRCoT(retriever=retriever, inference_manager=request_manager, model=model, reranker=reranker, load_if_exist=LOAD) 
        llm_suff_ircot_manager = IRCoTWithSufficiency(retriever=retriever, dataset="placeholder", inference_manager=request_manager, generator_model=model, sufficiency_model_path=request_manager, reranker=reranker, load_if_exist=LOAD) # API_URL 

        print(f"Processando o modelo {model}")
        for dataset, path, suff_path in [("2Wiki", "../datasets/2WikiTest.csv", "../models/Semantic-only 0 Ablation (seed 24).pt"), ("Hotpot","../datasets/HotpotTest.csv", "../models/Semantic-only 0 Ablation (seed 24) Hotpot.pt"), ("Musique","../datasets/MusiqueTest.csv", "../models/Semantic-only 0 Ablation (seed 24) Musique.pt")]: #
            print(f"Dataset: {dataset}")

            out_path_common = f"../results/{RETRIEVER_TRAINMENT}/{model.split(":")[0]}/vanilla/{dataset}{reranker_string}.csv"
            out_path_suff = f"../results/{RETRIEVER_TRAINMENT}/{model.split(":")[0]}/suff/{dataset}{reranker_string}.csv"
            out_path_llm = f"../results/{RETRIEVER_TRAINMENT}/{model.split(":")[0]}/llm/{dataset}{reranker_string}.csv"
            df = pd.read_csv(path)

            print("   Modo tradicional")
            results = common_ircot_manager.process_dataset(df.head(HEAD), query_col="question", k=5, metrics=metrics_list, intermediate_results_path=out_path_common, display=True) 
            # results = common_o1.process_dataset(df.head(HEAD), query_col="question", k=5, metrics=[metrics.LanguageExactMatch(), metrics.LanguageF1(tokenizer), metrics.LLMAsAJudge(request_manager, JUDGE_MODEL), metrics.LanguageRouge(), metrics.AveragePrecision(), metrics.RecallAtK(5), metrics.PrecisionAtK(5), metrics.PrecisionRecallCurve()], intermediate_results_path=f"../results/results_{dataset}_vanilla.csv", display=True) 
            df_results = pd.DataFrame(results)
            df_results = df_results.replace(r"\n", r"\\n", regex=True)
            df_results.to_csv(out_path_common, index=False, encoding="utf-8")
            print("Results for vanilla:", df_results["f1"].mean())


            print("   Modo com suficiência usando meu modelo")
            suff_ircot_manager = IRCoTWithSufficiency(retriever=retriever, dataset=dataset, inference_manager=request_manager, generator_model=model, sufficiency_model_path=suff_path, reranker=reranker, load_if_exist=LOAD, threshold=THRESH) # API_URL 
            results = suff_ircot_manager.process_dataset(df.head(HEAD), query_col="question", k=5, metrics=metrics_list, intermediate_results_path=out_path_suff, display=True)
            # results = suff_o1.process_dataset(df.head(HEAD), query_col="question", k=5, metrics=[metrics.LanguageExactMatch(), metrics.LanguageF1(tokenizer), metrics.LLMAsAJudge(request_manager, JUDGE_MODEL), metrics.LanguageRouge(), metrics.AveragePrecision(), metrics.RecallAtK(5), metrics.PrecisionAtK(5), metrics.PrecisionRecallCurve()], intermediate_results_path=f"../results/results_{dataset}_suff.csv", display=True)
            df_results = pd.DataFrame(results)
            df_results = df_results.replace(r"\n", r"\\n", regex=True)
            df_results.to_csv(out_path_suff, index=False, encoding="utf-8")
            print("Results for suff:", df_results["f1"].mean())
            del suff_ircot_manager

            print("   Modo com suficiência usando LLM-as-a-Judge")
            results = llm_suff_ircot_manager.process_dataset(df.head(HEAD), query_col="question", k=5, metrics=metrics_list, intermediate_results_path=out_path_llm, display=True)
            # results = suff_o1.process_dataset(df.head(HEAD), query_col="question", k=5, metrics=[metrics.LanguageExactMatch(), metrics.LanguageF1(tokenizer), metrics.LLMAsAJudge(request_manager, JUDGE_MODEL), metrics.LanguageRouge(), metrics.AveragePrecision(), metrics.RecallAtK(5), metrics.PrecisionAtK(5), metrics.PrecisionRecallCurve()], intermediate_results_path=f"../results/results_{dataset}_suff.csv", display=True)
            df_results = pd.DataFrame(results)
            df_results = df_results.replace(r"\n", r"\\n", regex=True)
            df_results.to_csv(out_path_llm, index=False, encoding="utf-8")
            print("Results for llm suff:", df_results["f1"].mean())

            print("="*30)
    except Exception as e:
        print("An error occurred:")
        traceback.print_exc()
        continue    
