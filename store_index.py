from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from pinecone import Pinecone
from dotenv import load_dotenv
import os

# Load API key from environment
load_dotenv()
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

# Process PDF
extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

# Initialize Pinecone client (v5.x)
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "fitness-chatbot"

# Ensure the index exists
if index_name not in pc.list_indexes().names():
    raise ValueError(f"Index '{index_name}' not found!")

# Initialize the index
index = pc.Index(index_name)

# Use Langchain's Pinecone wrapper from the community package
doc_store = LangchainPinecone.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# Add your embedded texts
# Convert Document objects to strings by extracting page_content
text_strings = [doc.page_content for doc in text_chunks]
doc_store.add_texts(text_strings)

print("✅ Data successfully stored in Pinecone.")
