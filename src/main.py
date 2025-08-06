from ingestion import load_documents, chunk_documents
from embed_store import create_vector_store

def main():
    docs = load_documents("data")
    chunks = chunk_documents(docs)
    create_vector_store(chunks)
    

if __name__ == "__main__":
    main()