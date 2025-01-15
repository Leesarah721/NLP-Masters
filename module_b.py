# module_b.py

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
# 필요한 경우, BM25 또는 기타 Retriever 라이브러리 import
# from rank_bm25 import BM25Okapi  # 예시

def create_embedding_model():
    """OpenAI 또는 FastEmbedEmbeddings 등을 초기화."""
    # 예: OpenAIEmbeddings()
    return OpenAIEmbeddings()

def create_vectorstore(splits, embedding_model):
    """분할된 문서(splits)를 FAISS 벡터스토어에 저장하고 반환한다."""
    vectordb = FAISS.from_documents(splits, embedding_model)
    return vectordb

# 추가 예시: BM25, EnsembleRetriever 등
# def create_bm25_retriever(docs):
#     pass
#
# def create_ensemble_retriever(...):
#     pass
