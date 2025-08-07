from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

def create_vector_store(chunks, persist_directory="faiss_index"):
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(chunks, embedding)
    vectordb.save_local(persist_directory)
    print("✅ FAISS vector store created and saved to:", persist_directory)
    return vectordb

def load_vector_store(persist_directory: str, embedding):
    return FAISS.load_local(
        persist_directory,
        embeddings=embedding,
        allow_dangerous_deserialization=True  # 🚨 Required from LangChain 0.2+
    )
