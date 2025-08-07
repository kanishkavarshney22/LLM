from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# retriever.py

def get_relevant_docs(query, vectordb, k=5):
    # vectordb is already loaded in main.py and passed here
    docs = vectordb.similarity_search(query, k=k)
    return docs

