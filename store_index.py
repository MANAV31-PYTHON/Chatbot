'''from langchain_community.vectorstores import Pinecone
from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from dotenv import load_dotenv
import pinecone
import os

# Load Pinecone API key and env from .env
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
INDEX_NAME = "fitness-chatbot"

# Step 1: Initialize Pinecone manually
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

# Load and preprocess PDF
extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)

# Download SentenceTransformer model via LangChain
embeddings = download_hugging_face_embeddings()

# Convert Document objects to plain strings
texts = [doc.page_content for doc in text_chunks]

# Push to Pinecone vector store via LangChain wrapper
doc_store = Pinecone.from_texts(
    texts=texts,
    embedding=embeddings,
    index_name=INDEX_NAME,
    
)

print("✅ Data successfully stored in Pinecone.")
'''
from pinecone import Pinecone, ServerlessSpec
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from dotenv import load_dotenv
import os

# Load API credentials from .env
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")  # e.g., "us-east-1"
INDEX_NAME = "fitness-chatbot"

# ✅ Step 1: Create Pinecone client instance
pc = Pinecone(api_key=PINECONE_API_KEY)

# ✅ Step 2: Make sure the index exists
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,  # if using MiniLM-L6-v2
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV)
    )

# ✅ Step 3: Load and process documents
extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

# ✅ Step 4: Convert docs to plain text
texts = [doc.page_content for doc in text_chunks]

# ✅ Step 5: Store using LangChain wrapper
doc_store = LangchainPinecone.from_texts(
    texts=texts,
    embedding=embeddings,
    index_name=INDEX_NAME
)

print("✅ Data successfully stored in Pinecone.")
