import os
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline

import sys
import os

# Dynamically get the project root (one level above 'src')
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the root directory to Python's module search path
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# Import configurations
from config import PERSISTENT_DIR, EMBEDDING_MODEL_NAME, CHAT_MODEL_NAME, CHAT_MODEL_TEMPERATURE, VECTOR_DB_K, REPHRASING_PROMPT

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

def retrieve_documents(query, chat_history):
    """
    Retrieves relevant documents based on user query and chat history.
    
    :param query: User's query
    :param chat_history: Previous conversation history
    :return: List of relevant document chunks
    """
    retrieved_docs = history_aware_retriever.invoke({"input": query, "chat_history": chat_history})
    return retrieved_docs