# module_c.py

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

def create_retriever(vectorstore, search_type='mmr', k=2, fetch_k=4):
    """FAISS 등 벡터 스토어를 기반으로 한 retriever를 생성."""
    retriever = vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs={'k': k, 'fetch_k': fetch_k}
    )
    return retriever

def create_memory():
    """대화형 메모리 생성 (ConversationBufferMemory)."""
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        output_key='answer',
        return_messages=True
    )
    return memory

def create_conversational_chain(llm, retriever, memory, verbose=False):
    """LLM + Retriever + Memory를 묶어 RAG(ConversationalRetrievalChain)를 만든다."""
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        verbose=verbose
    )
    return qa_chain
