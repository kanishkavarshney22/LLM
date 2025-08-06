from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def create_vector_store(chunks, persist_directory="vector_store"):
    embedding_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"  # small + fast + accurate
    )
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory
    )
    vectordb.persist()
    return vectordb
