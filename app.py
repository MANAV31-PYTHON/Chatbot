from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from pinecone import Pinecone
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import *
import os
from threading import Thread

import logging

app = Flask(__name__)

# Load API key from environment
load_dotenv()
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    embeddings = download_hugging_face_embeddings()
    logger.info("Embeddings loaded successfully.")
except Exception as e:
    logger.error(f"Error loading embeddings: {e}")
    embeddings = None

try:
    # Initialize Pinecone client (v5.x)
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index_name = "fitness-chatbot"
    docsearch = LangchainPinecone.from_existing_index(index_name, embeddings)
    logger.info("Pinecone client and docsearch initialized successfully.")
except Exception as e:
    logger.error(f"Error initializing Pinecone client or docsearch: {e}")
    docsearch = None

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

chain_type_kwargs = {"prompt": PROMPT}

import torch

try:
    llm = CTransformers(
        model="model/llama-2-7b-chat.ggmlv3.q2_K.bin",
        model_type="llama",
        config={
            'max_new_tokens': 256,  # Reduced for faster response
            'temperature': 0.5      # Lower temperature for more focused output
        }
    )
    logger.info("LLM 'llama' created successfully with optimized config.")
except Exception as e:
    logger.error(f"Failed to create LLM 'llama': {e}", exc_info=True)
    llm = None


qa=RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True, 
    chain_type_kwargs=chain_type_kwargs)



@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result=qa.invoke({"query": input})
    print("Response : ", result["result"])
    return jsonify(result["result"])


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=False)
