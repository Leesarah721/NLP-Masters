# module_c.py

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_core.prompts import PromptTemplate

def create_retriever(vectorstore, search_type, k, fetch_k,splits):
    """FAISS 등 벡터 스토어를 기반으로 한 retriever를 생성."""
    # retriever = vectorstore.as_retriever(
    #     search_type=search_type,
    #     search_kwargs={'k': k, 'fetch_k': fetch_k}
    # )
    # 앙상블 검색기 사용
    dense_retriever = vectorstore.as_retriever(search_type=search_type,search_kwargs={'k': k})
    
    sparse_retriever = BM25Retriever.from_documents(splits)
    sparse_retriever.k = k
    return dense_retriever, sparse_retriever

def create_memory():
    """대화형 메모리 생성 (ConversationBufferMemory)."""
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        output_key='answer',
        return_messages=True
    )
    return memory

def format_docs(docs):
    # 검색한 문서 결과를 하나의 문단으로 합쳐줍니다.
    return "\n\n".join(doc.page_content for doc in docs)


def create_conversational_chain(llm, dense_retriever, sparse_retriever , memory, verbose, query):
    """LLM + Retriever + Memory를 묶어 RAG(ConversationalRetrievalChain)를 만든다."""
    ensemble_retriever = EnsembleRetriever(
        retrievers=[dense_retriever, sparse_retriever], weights=[0.5, 0.5])
    
    prompt = PromptTemplate.from_template(
        """
            PDF 문서에서 주어진 질문에 대한 답변을 제공하는 프롬프트입니다.
            주어진 문서는 텍스트와 표로 구성되어 있습니다.
            표 항목도 있으므로 표 항목은 주변 텍스트를 확인하고 답변해주세요.
            숫자로 이루어진 표는 되도록 합계쪽을 읽고 대답해주세요
            주어진 문서를 참고하여 질문에 대한 자세한 답변을 제공하고, 관련된 경우 목차를 포함하여 추가적인 지침을 제공합니다.
            답변을 찾지 못한 경우 유사한 답변을 찾아 대답해 주세요
            답변은 너무 길지 않게 요약해서 한국어로만 대답해주세요
             #Contents:
            {context}

            #Question:
            {question}

            #Answer:
        """
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=ensemble_retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt":prompt},
        verbose=verbose
    )
    return qa_chain
