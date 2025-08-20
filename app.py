'''from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
from src.prompt import *
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from pinecone import Pinecone
from dotenv import load_dotenv
import os

# ✅ Initialize Flask
app = Flask(__name__)

# ✅ Load .env variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_API_ENV")
INDEX_NAME = "fitness-chatbot"

# ✅ Step 1: Initialize Pinecone client (new version)
pc = Pinecone(api_key=PINECONE_API_KEY)

# ✅ Step 2: Embeddings (already downloaded)
embeddings = download_hugging_face_embeddings()

# ✅ Step 3: Load existing Pinecone index
docsearch = LangchainPinecone.from_existing_index(
    index_name=INDEX_NAME,
    embedding=embeddings,
)

# ✅ Step 4: Prompt template
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs = {"prompt": PROMPT}

# ✅ Step 5: Load LLM (CTransformers with optimized config)
llm = CTransformers(
    model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
    model_type="llama",
    config={
        "max_new_tokens": 150,  # Reduce if it's too slow
        "temperature": 0.7,
        "repetition_penalty": 1.1,
        "stream": False
    }
)

# ✅ Step 6: Create RetrievalQA chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)

# ✅ Step 7: Routes
@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["POST"])
def chat():
    user_input = request.form["msg"]
    if not user_input.strip():
        return "Please enter a valid question."
    try:
        result = qa.invoke({"query": user_input})
        return result["result"]
    except Exception as e:
        return f"Error: {str(e)}"

# ✅ Run the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)'''
from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
from src.prompt import *
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from pinecone import Pinecone
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI  # ✅ New import

# Initialize Flask
app = Flask(__name__)

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_NAME = "fitness-chatbot"

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Load embeddings
embeddings = download_hugging_face_embeddings()

# Connect to existing index
docsearch = LangchainPinecone.from_existing_index(
    index_name=INDEX_NAME,
    embedding=embeddings,
)

# Prompt template
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs = {"prompt": PROMPT}

# ✅ Use OpenAI Chat model instead of CTransformers
llm = ChatOpenAI(
    model="gpt-4o-mini",  # Fast & cost-efficient
    temperature=0.7,
    max_tokens=150,
    api_key=OPENAI_API_KEY
)

# RetrievalQA chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["POST"])
def chat():
    user_input = request.form["msg"]
    if not user_input.strip():
        return "Please enter a valid question."
    try:
        result = qa.invoke({"query": user_input})
        return result["result"]
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)

