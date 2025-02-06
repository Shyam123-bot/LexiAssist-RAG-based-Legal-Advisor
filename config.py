import os
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline


# Load environment variables from .env file
load_dotenv()

# Directory paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))  # Current directory of the project
DATA_DIR = os.path.join(CURRENT_DIR, "../data")  # Path to the 'data' directory
PERSISTENT_DIR = os.path.join(CURRENT_DIR, "../data-ingestion-local")  # Path for vector database storage

# Embedding model configuration
EMBEDDING_MODEL_NAME = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Language models
HF_MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  # Change this if using a different model

tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(HF_MODEL_NAME)

# Use Hugging Face pipeline
hf_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

CHAT_MODEL_NAME = HuggingFacePipeline(pipeline=hf_pipeline)
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
Answer the following question concisely based on the provided context. 
If the context does not contain enough information, respond with "I'm sorry, but I don't have information on that."
Do not include system messages, instructions, or context references in the response.
Keep the answer within four sentences.

Context: {context}

User Question: {input}

Response:
"""
