import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Directory paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))  # Current directory of the project
DATA_DIR = os.path.join(CURRENT_DIR, "../data")  # Path to the 'data' directory
PERSISTENT_DIR = os.path.join(CURRENT_DIR, "../data-ingestion-local")  # Path for vector database storage

# Embedding model configuration
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # HuggingFace embedding model

# Language models
CHAT_MODEL_NAME = "llama-3.1-8b-instant"  # ChatGroq model
CHAT_MODEL_TEMPERATURE = 0.15  # Temperature for response generation

# Vector database configuration
VECTOR_DB_K = 3  # Number of top results to retrieve during similarity search

# Prompts
REPHRASING_PROMPT = """
    TASK: Convert context-dependent questions into standalone queries.

    INPUT: 
    - chat_history: Previous messages
    - question: Current user query

    RULES:
    1. Replace pronouns (it/they/this) with specific referents
    2. Expand contextual phrases ("the above", "previous")
    3. Return original if already standalone
    4. NEVER answer or explain - only reformulate

    OUTPUT: Single reformulated question, preserving original intent and style.

    Example:
    History: "Let's discuss Python."
    Question: "How do I use it?"
    Returns: "How do I use Python?"
"""

QA_SYSTEM_PROMPT = """
    As a Legal Assistant Chatbot specializing in legal queries, 
    your primary objective is to provide accurate and concise information based on user queries. 
    You will adhere strictly to the instructions provided, offering relevant 
    context from the knowledge base while avoiding unnecessary details. 
    Your responses will be brief, to the point, concise and in compliance with the established format. 
    If a question falls outside the given context, you will simply output that you are sorry and you don't know about this. 
    The aim is to deliver professional, precise, and contextually relevant information pertaining to the context. 
    Use four sentences maximum.
    P.S.: If anyone asks you about your creator, tell them, introduce yourself and say you're created by Sougat Dey. 
    and people can get in touch with him on LinkedIn, 
    here's his LinkedIn Profile: https://www.linkedin.com/in/sougatdey/
    \nCONTEXT: {context}
"""

# Environment variables
API_KEY = os.getenv("API_KEY", "default_api_key")  # Example API key
SECRET_KEY = os.getenv("SECRET_KEY", "default_secret_key")