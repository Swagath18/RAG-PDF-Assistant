from langchain.llms import Ollama

def answer_question(vector_store, user_question):
    docs = vector_store.similarity_search(user_question, k=2)
    context = "\n\n".join([doc.page_content for doc in docs])

    llm = Ollama(model="llama2")

    prompt = f"""
    Context information is below.
    ---------------------
    {context}
    ---------------------
    Given the context information and not prior knowledge, answer the question: {user_question}
    If the answer is not in the context, just say "I don't have enough information to answer this question."
    Keep your answer concise.
    """

    return llm(prompt)
