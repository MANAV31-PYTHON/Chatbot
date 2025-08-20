from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
from dotenv import load_dotenv
import os

load_dotenv()
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "fitness-chatbot"  # New index name to avoid dimension conflict

# Create index with correct dimension for HuggingFace embeddings
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,  # HuggingFace MiniLM-L6-v2 embeddings dimension
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# Load documents
extracted_data = load_pdf_file("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

# Store vectors
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings
)

print("✅ Data stored in Pinecone successfully with HuggingFace embeddings")
