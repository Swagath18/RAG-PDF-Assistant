import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Ollama
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.prompts import PromptTemplate

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text, chunk_size=300, chunk_overlap=30):
    # Very small chunks for fast processing
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    # Using a lightweight embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

def answer_question(vector_store, user_question):
    # Get the 2 most relevant chunks to keep context small
    docs = vector_store.similarity_search(user_question, k=2)
    
    # Format context for the model
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Use Ollama with a tiny model
    llm = Ollama(model="llama2")  # You can also use tinyllama for even faster performance
    
    # Simple prompt template optimized for smaller models
    prompt = f"""
    Context information is below.
    ---------------------
    {context}
    ---------------------
    Given the context information and not prior knowledge, answer the question: {user_question}
    If the answer is not in the context, just say "I don't have enough information to answer this question."
    Keep your answer concise.
    """
    
    response = llm(prompt)
    return response

def main():
    st.set_page_config(page_title="PDF Chat", page_icon="📄")
    st.title("📚 PDF Chat - 100% Free & Local")
    
    # Initialize session state
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    if 'processed' not in st.session_state:
        st.session_state.processed = False
    
    with st.sidebar:
        st.header("Document Settings")
        pdf_docs = st.file_uploader("Upload PDF Files", accept_multiple_files=True)
        
        # Performance settings
        st.subheader("Performance Settings")
        chunk_size = st.slider("Chunk Size", min_value=100, max_value=500, value=300, 
                              help="Smaller chunks process faster but may miss context")
        
        if st.button("Process Documents"):
            if pdf_docs:
                with st.spinner("Processing documents..."):
                    # Get text from PDFs
                    raw_text = get_pdf_text(pdf_docs)
                    
                    # Get text chunks
                    text_chunks = get_text_chunks(raw_text, chunk_size=chunk_size, chunk_overlap=int(chunk_size/10))
                    st.info(f"Created {len(text_chunks)} chunks from your documents")
                    
                    # Create vector store
                    st.session_state.vector_store = get_vector_store(text_chunks)
                    st.session_state.processed = True
                    st.success("Ready to answer questions!")
            else:
                st.error("Please upload at least one PDF document")
    
        st.header("About")
        st.markdown("""
        This app runs 100% locally on your computer:
        - Uses Ollama for local inference
        - Tiny, fast embedding model
        - No API costs or internet required
        """)
    
    # Main area
    if not st.session_state.processed:
        st.info("👈 Please upload and process documents using the sidebar")
    else:
        user_question = st.text_input("Ask a question about your documents:")
        if user_question:
            with st.spinner("Finding answer..."):
                answer = answer_question(st.session_state.vector_store, user_question)
                st.write("### Answer")
                st.write(answer)

if __name__ == "__main__":
    main()
