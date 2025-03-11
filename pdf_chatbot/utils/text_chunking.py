from langchain.text_splitter import RecursiveCharacterTextSplitter

def get_text_chunks(text, chunk_size=300, chunk_overlap=30):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return text_splitter.split_text(text)
