# main.py

from ingestion import load_documents, chunk_documents
from embed_store import create_vector_store, load_vector_store
from retriever import get_relevant_docs
from llm_chain import run_llm_query
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

def main():
    persist_path = "faiss_index"
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create the vector store if it doesn't exist
    if not os.path.exists(persist_path) or not os.path.exists(os.path.join(persist_path, "index.faiss")):
        print("📄 No existing vector store found. Creating a new one...")
        docs = load_documents("data")
        chunks = chunk_documents(docs)
        create_vector_store(chunks, embedding, persist_path)

    print("🔄 Loading existing vector store...")
    vectordb = load_vector_store(persist_path, embedding)  # This returns a FAISS object

    query = "Is knee surgery covered for a 46-year-old male in Pune?"
    relevant_docs = get_relevant_docs(query, vectordb)  # <-- now passes the object, not class
    result = run_llm_query(relevant_docs, query)
    print(result)

if __name__ == "__main__":
    main()
