import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain.chains import create_history_aware_retriever

# Import configurations
from config import PERSISTENT_DIR, EMBEDDING_MODEL_NAME, CHAT_MODEL_NAME, CHAT_MODEL_TEMPERATURE, VECTOR_DB_K, REPHRASING_PROMPT

# Initialize embedding model
embedF = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

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
chatmodel = ChatGroq(model=CHAT_MODEL_NAME, temperature=CHAT_MODEL_TEMPERATURE)
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
