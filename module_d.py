# module_d.py

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama

def select_llm_model():
    """
    사이드바에서 모델을 선택하고, 해당 모델에 맞춰 LLM(ChatOpenAI 등) 객체를 생성.
    """
    model_option = st.sidebar.selectbox(
        label="Select LLM Model",
        options=["gpt-3.5-turbo", "gpt-4", "llama3.1", "llama3.2", "gemini1", "gemini2"],
        index=0
    )
    # 예: 실제 오라클, 구글, 메타 등 모델은 따로 API 수정이 필요
    if model_option in ["gpt-3.5-turbo", "gpt-4"]:
        llm = ChatOpenAI(model_name=model_option, temperature=0)
    elif model_option in ["llama3.1", "llama3.2"]:
        # Ollama 예시
        llm = ChatOllama(model=model_option, base_url=st.secrets.get("OLLAMA_ENDPOINT", "http://localhost:11411"))
    elif model_option.startswith("gemini"):
        llm = ChatOpenAI(model_name=model_option, temperature=0)
    else:
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    return llm

def generate_answer(qa_chain, query):
    """
    질의(query)를 받아서 RAG 체인(qa_chain)으로부터 답변을 생성하고,
    답변과 소스 문서를 함께 반환한다.
    """
    result = qa_chain.invoke({"question": query})
    answer = result["answer"]
    source_docs = result["source_documents"]  # 인용 문서들
    return answer, source_docs
