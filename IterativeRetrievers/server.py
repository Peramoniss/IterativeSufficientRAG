import os
# EU TE ODEIO JAVA!!!!!!!!
java_home = r"C:\Program Files\Java\jdk-27"  # Replace with your actual JDK path
jvm_path = os.path.join(java_home, "bin", "server")

os.environ["JAVA_HOME"] = java_home
os.environ["PATH"] = jvm_path + os.pathsep + os.environ.get("PATH", "")

from Retriever.retriever import DenseRetriever, BM25Retriever
from Reranker.reranker import LocalReranker
from SufficiencyModel.modelStructures import SemanticOnlyBERT, TextDataset, SyntacticGATWithBERT, GraphDataset
from typing import List, Optional, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import torch

retriever = None
reranker = None
sufficiency_2wiki = None
sufficiency_hotpot = None
sufficiency_musique = None

mode = "SEMANTIC"
# mode = "GRAPH"
dataset_class = TextDataset if mode == "SEMANTIC" else GraphDataset
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager loads heavy resources (models, FAISS index)
    once when starting the server and releases them on shutdown.
    """
    global retriever, reranker, sufficiency_2wiki, sufficiency_hotpot, sufficiency_musique
    
    # retriever = DenseRetriever("intfloat/e5-base-v2", None, corpus_path="../documents/hotpot_multihop_corpus.jsonl", corpus_in_memory=True)
    # retriever.read_index("../index/hotpot_ivfsq8/e5_IVF1024,SQ8.index")
    # retriever.index.nprobe = 20
    # print("DenseRetriever initialized and ready.")
    retriever = BM25Retriever("../index/wiki_bm25") # Full wiki, not filtered
    print("BM25 initialized and ready.")

    if mode == "GRAPH":
        sufficiency_2wiki =  SyntacticGATWithBERT("bert-base-uncased", hidden_channels=32, num_classes=2, heads=8, dropout_rate=0.2, freeze_bert_layers=6)
        sufficiency_hotpot =  SyntacticGATWithBERT("bert-base-uncased", hidden_channels=32, num_classes=2, heads=8, dropout_rate=0.2, freeze_bert_layers=6)
        sufficiency_musique =  SyntacticGATWithBERT("bert-base-uncased", hidden_channels=32, num_classes=2, heads=8, dropout_rate=0.2, freeze_bert_layers=6)
        
        checkpoint = torch.load("../models/GAT 0 (seed 24).pt", map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        sufficiency_2wiki.load_state_dict(checkpoint['model_state_dict'])
        checkpoint = torch.load("../models/GAT 0 (seed 24) Hotpot.pt", map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        sufficiency_hotpot.load_state_dict(checkpoint['model_state_dict'])
        checkpoint = torch.load("../models/GAT 0 (seed 24) Musique.pt", map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        sufficiency_musique.load_state_dict(checkpoint['model_state_dict'])
    else:
        sufficiency_2wiki =  SemanticOnlyBERT("bert-base-uncased", num_classes=2, dropout_rate=0.2, freeze_bert_layers=6)
        sufficiency_hotpot =  SemanticOnlyBERT("bert-base-uncased", num_classes=2, dropout_rate=0.2, freeze_bert_layers=6)
        sufficiency_musique =  SemanticOnlyBERT("bert-base-uncased", num_classes=2, dropout_rate=0.2, freeze_bert_layers=6)
        
        checkpoint = torch.load("../models/Semantic-only 0 Ablation (seed 24).pt", map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        sufficiency_2wiki.load_state_dict(checkpoint['model_state_dict'])
        checkpoint = torch.load("../models/Semantic-only 0 Ablation (seed 24) Hotpot.pt", map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        sufficiency_hotpot.load_state_dict(checkpoint['model_state_dict'])
        checkpoint = torch.load("../models/Semantic-only 0 Ablation (seed 24) Musique.pt", map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        sufficiency_musique.load_state_dict(checkpoint['model_state_dict'])
    
    sufficiency_2wiki.eval()
    sufficiency_hotpot.eval()
    sufficiency_musique.eval()
    # checkpoint = torch.load("../models/All GAT (seed 42).pt", map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    print("Sufficiency model initialized and ready.")

    reranker = LocalReranker("BAAI/bge-reranker-v2-m3")
    print("Reranker initialized and ready.")
    
    yield
    
    # Clean up resources on shutdown if necessary
    retriever = None
    sufficiency = None

app = FastAPI(
    title="Dense Retrieval API",
    description="API for retrieving relevant documents using FAISS and Transformer Embeddings.",
    lifespan=lifespan
)

class RetrieveRequest(BaseModel):
    query: str = Field(..., description="Question the retriever will use to select similar documents")
    password: str = Field(..., description="Authentication token")
    k: int = Field(default=5, ge=1, le=100, description="Number of results to retrieve")
    return_corpus_doc: bool = Field(default=True, description="Whether to include matched document content")

class RetrieveResponse(BaseModel):
    scores: List[float]
    indexes: List[int]
    documents: Optional[List[Any]] = None

# --- API Endpoint ---

@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve_endpoint(request: RetrieveRequest):
    if retriever is None:
        raise HTTPException(status_code=500, detail="Retriever model is not initialized.")

    try:
        if request.password != "kTOIe4wQQn 8wQ1XH2jS1 MjkZKOMab6 Rp3eMVT34D 1DitCzvPEm":
            raise HTTPException(status_code=404, detail="Authentication token not allowed")
        if request.return_corpus_doc:
            scores, indexes, docs = retriever.retrieve(
                query=request.query,
                k=request.k,
                return_corpus_doc=True,
            )
            return RetrieveResponse(
                scores=scores[0].tolist(),
                indexes=indexes[0].tolist(),
                documents=docs
            )
        else:
            scores, indexes = retriever.retrieve(
                query=request.query,
                k=request.k,
                return_corpus_doc=False,
            )
            return RetrieveResponse(
                scores=scores[0].tolist(),
                indexes=indexes[0].tolist(),
                documents=None
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class RerankRequest(BaseModel):
    query: str = Field(..., description="Question the reranker will use to cross-encode documents")
    documents: list[dict] = Field(..., description="List of string documents to cross-encode with the query and rerank in order of relevance")
    password: str = Field(..., description="Authentication token")
    k: int | None = Field(default=5, ge=1, le=100, description="Number of top documents to return")
    batch_size: int | None = Field(default=None, description="Number of documents encoded at a time")

class RerankResponse(BaseModel):
    reranked_docs: list[dict]
    reranked_scores: list[float]

@app.post("/rerank", response_model=RerankResponse)
def rerank_endpoint(request: RerankRequest):
    if reranker is None:
        raise HTTPException(status_code=500, detail="Reranker model is not initialized.")

     
    try:
        docs, scores = reranker.rerank(
            query=request.query,
            documents=request.documents,
            k=request.k,
            batch_size=request.batch_size
        )
        return RerankResponse(
            reranked_docs=docs,
            reranked_scores=[float(s) for s in scores]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class SufficiencyRequest(BaseModel):
    query: str = Field(..., description="Question")
    password: str = Field(..., description="Authentication token")
    docs: list[str] = Field(..., description="Reference RAG documents")

class SufficiencyResponse(BaseModel):
    logits: list[list[float]]
    suff_tokens_used: int

@app.post("/sufficiency/2Wiki", response_model=SufficiencyResponse)
def sufficient_endpoint(request: SufficiencyRequest):
    global sufficiency_2wiki, dataset_class
    logits, suff_tokens_used = sufficiency_2wiki(dataset_class([(request.query, request.docs, 0)]))
    logits_list = logits.detach().cpu().tolist()
    return SufficiencyResponse(logits=logits_list, suff_tokens_used=suff_tokens_used)

@app.post("/sufficiency/Hotpot", response_model=SufficiencyResponse)
def sufficient_endpoint(request: SufficiencyRequest):
    global sufficiency_hotpot, dataset_class
    logits, suff_tokens_used = sufficiency_hotpot(dataset_class([(request.query, request.docs, 0)]))
    logits_list = logits.detach().cpu().tolist()
    return SufficiencyResponse(logits=logits_list, suff_tokens_used=suff_tokens_used)

@app.post("/sufficiency/Musique", response_model=SufficiencyResponse)
def sufficient_endpoint(request: SufficiencyRequest):
    global sufficiency_musique, dataset_class
    logits, suff_tokens_used = sufficiency_musique(dataset_class([(request.query, request.docs, 0)]))
    logits_list = logits.detach().cpu().tolist()
    return SufficiencyResponse(logits=logits_list, suff_tokens_used=suff_tokens_used)