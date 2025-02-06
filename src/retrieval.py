import os
import re
import sys
import torch
from typing import List
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

# Dynamically get the project root (one level above 'src')
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the root directory to Python's module search path
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# Import configurations
from config import (
    PERSISTENT_DIR, EMBEDDING_MODEL_NAME, CHAT_MODEL_NAME,
    CHAT_MODEL_TEMPERATURE, VECTOR_DB_K, REPHRASING_PROMPT
)

# Initialize embedding model
embedF = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load vector database
vectorDB = Chroma(embedding_function=embedF, persist_directory=PERSISTENT_DIR)

# Set up retriever
kb_retriever = vectorDB.as_retriever(search_type="similarity", search_kwargs={"k": VECTOR_DB_K})

# Define rephrasing prompt
rephrasing_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", REPHRASING_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Initialize history-aware retriever
chatmodel = CHAT_MODEL_NAME
history_aware_retriever = create_history_aware_retriever(
    llm=chatmodel,
    retriever=kb_retriever,
    prompt=rephrasing_prompt
)

# Load reranking model (cross-encoder)
device = "cuda" if torch.cuda.is_available() else "cpu"
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-en", device=device)

# Placeholder for BM25 Data
bm25_index = None
bm25_docs = []
bm25_corpus = []

def initialize_bm25():
    """
    Loads documents from vectorDB and initializes the BM25 index.
    """
    global bm25_index, bm25_docs, bm25_corpus
    documents = vectorDB.get()
    
    # Debugging print
    print("vectorDB.get() response:", documents)

    documents = documents.get("documents", [])  # Ensure valid key lookup
    
    if not documents:
        print("Warning: No documents found in vectorDB!")
        return  # Exit function if no documents exist
    
    bm25_docs = documents
    bm25_corpus = [re.sub(r'\W+', ' ', doc).lower().split() for doc in documents]

    if not bm25_corpus:
        print("BM25 corpus is empty. Cannot initialize BM25.")
        return

    bm25_index = BM25Okapi(bm25_corpus)  # Initialize BM25
    print("BM25 index initialized successfully with", len(bm25_corpus), "documents.")

initialize_bm25()  # Call function to load BM25 index

def bm25_search(query: str, top_k: int = 3) -> List[str]:
    """
    Performs a BM25-based lexical search.
    
    :param query: User query string
    :param top_k: Number of results to retrieve
    :return: List of top-k matching documents
    """
    if not bm25_index:
        print("BM25 index is not initialized. Returning empty results.")
        return []

    query_tokens = query.lower().split()
    scores = bm25_index.get_scores(query_tokens)  # Get BM25 scores
    top_indices = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]  # Get top-k results
    return [bm25_docs[i] for i in top_indices]

def rerank_results(query: str, docs: List[str], top_k: int = 3) -> List[str]:
    """
    Reranks documents using a cross-encoder model.
    
    :param query: User query
    :param docs: List of retrieved documents
    :param top_k: Number of top results to return
    :return: List of reranked top-k documents
    """
    if not docs:
        return []

    # Prepare input pairs (query, document)
    pairs = [[query, doc] for doc in docs]
    
    # Get relevance scores from the Cross-Encoder
    scores = reranker.predict(pairs)

    # Sort documents by score (higher = better)
    reranked_docs = [doc for _, doc in sorted(zip(scores, docs), reverse=True)]

    # Return only top_k results
    return reranked_docs[:top_k]

def retrieve_documents(query, chat_history):
    """
    Retrieves relevant documents based on user query and chat history.
    
    :param query: User's query
    :param chat_history: Previous conversation history
    :return: List of relevant document chunks
    """
    # Retrieve using Semantic Search
    semantic_docs = history_aware_retriever.invoke({"input": query, "chat_history": chat_history})

    # Retrieve using BM25 Lexical Search
    bm25_results = bm25_search(query, top_k=VECTOR_DB_K)

    # Combine both results, ensuring no duplicates
    combined_results = list({doc.get("text", ""): doc for doc in semantic_docs + [{"text": doc} for doc in bm25_results]}.values())

    # Apply reranking to prioritize the best documents
    ranked_scores = reranker.predict([(query, doc["text"]) for doc in combined_results])

    # Sort results based on the reranker score
    sorted_results = [combined_results[i] for i in sorted(range(len(ranked_scores)), key=lambda x: ranked_scores[x], reverse=True)]

    # Return top-ranked documents
    return sorted_results[:VECTOR_DB_K]
