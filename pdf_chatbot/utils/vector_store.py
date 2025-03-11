from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store
