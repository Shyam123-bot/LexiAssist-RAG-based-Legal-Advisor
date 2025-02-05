import os
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline


import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import configurations
from config import CHAT_MODEL_NAME, CHAT_MODEL_TEMPERATURE, QA_SYSTEM_PROMPT
from src.retrieval import history_aware_retriever

# Load the Hugging Face model
hf_pipeline = pipeline("text-generation", model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")

# Replace OllamaLLM with HuggingFacePipeline
chatmodel = HuggingFacePipeline(pipeline=hf_pipeline)

# Define the QA prompt
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", QA_SYSTEM_PROMPT),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ]
)

# Create the document chain
qa_chain = create_stuff_documents_chain(llm=chatmodel, prompt=qa_prompt)

# Final RAG retrieval chain
coversational_rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

def generate_response(user_query, chat_history):
    """
    Generates a response for the given user query using the RAG-based retrieval chain.
    :param user_query: The user's question.
    :param chat_history: The previous chat messages.
    :return: The generated response.
    """
    result = coversational_rag_chain.invoke({"input": user_query, "chat_history": chat_history})
    return result["answer"]