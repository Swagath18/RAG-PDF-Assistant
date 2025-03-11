import streamlit as st
from utils.pdf_processing import get_pdf_text
from utils.text_chunking import get_text_chunks
from utils.vector_store import get_vector_store
from utils.qa_system import answer_question

def main():
    st.set_page_config(page_title="PDF Chat", page_icon="📄")
    st.title("📚 PDF Chat - 100% Free & Local")

    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    if 'processed' not in st.session_state:
        st.session_state.processed = False

    with st.sidebar:
        st.header("Document Settings")
        pdf_docs = st.file_uploader("Upload PDF Files", accept_multiple_files=True)

        # Performance settings
        st.subheader("Performance Settings")
        chunk_size = st.slider("Chunk Size", min_value=100, max_value=500, value=300, help="Smaller chunks process faster but may miss context")

        if st.button("Process Documents"):
            if pdf_docs:
                with st.spinner("Processing documents..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text, chunk_size=chunk_size, chunk_overlap=int(chunk_size/10))
                    st.info(f"Created {len(text_chunks)} chunks from your documents")

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
