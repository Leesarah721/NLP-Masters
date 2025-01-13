# main.py

import streamlit as st
from streaming import StreamHandler  # 원본에서 사용되었다고 가정
import module_a
import module_b
import module_c
import module_d
import utils

# LangSmith tracking (원본 코드)
from langchain_teddynote import logging
logging.langsmith("LANCHAIN-PROJECT")

# Streamlit 기본 설정
st.set_page_config(page_title="ChatPDF", page_icon="📄")
st.header('📄청약스캐너')
st.write('해당 청약 공고문에 대해 궁금한 점을 질문해보세요!')
st.write('[![view source code ](https://img.shields.io/badge/view_source_code-gray?logo=github)](https://github.com/shashankdeshpande/langchain-chatbot/blob/master/pages/4_%F0%9F%93%84_chat_with_your_documents.py)')

@utils.enable_chat_history
def main():
    # 세션 동기화
    utils.sync_st_session()

    # 1) 사이드바에서 PDF 업로드
    uploaded_files = st.sidebar.file_uploader(
        label='Upload PDF files',
        type=['pdf'],
        accept_multiple_files=True
    )
    if not uploaded_files:
        st.error("공고문을 PDF 파일로 업로드해주세요!")
        st.stop()

    # 2) 모델(LLM) 선택 (module_d 예시, 혹은 utils.py에서 configure_llm() 사용)
    llm = module_d.select_llm_model()
    # 또는: llm = utils.configure_llm()

    # 3) PDF → 로드 → 텍스트 분할
    with st.spinner("Loading and splitting documents..."):
        docs = module_a.load_pdfs(uploaded_files)
        splits = module_a.split_documents(docs)

    # 4) 임베딩/벡터 DB 생성
    with st.spinner("Creating embeddings & vector store..."):
        embedding_model = module_b.create_embedding_model()
        vectordb = module_b.create_vectorstore(splits, embedding_model)

    # 5) Retriever, Memory, Conversational Chain 구성
    retriever = module_c.create_retriever(vectordb, search_type='mmr', k=2, fetch_k=4)
    memory = module_c.create_memory()
    qa_chain = module_c.create_conversational_chain(llm, retriever, memory, verbose=False)

    # 6) 사용자 Query
    user_query = st.chat_input(placeholder="무엇이든 질문해보세요!")
    if user_query:
        # 사용자가 보낸 메시지 UI 출력
        utils.display_msg(user_query, 'user')

        # 7) LLM에 질문을 전달 → RAG 체인으로 답변 생성
        with st.chat_message("assistant"):
            st_cb = StreamHandler(st.empty())  # 스트리밍 콜백
            result_answer, source_docs = module_d.generate_answer(qa_chain, user_query)

            # 8) 답변 표시
            st.session_state.messages.append({"role": "assistant", "content": result_answer})
            st.write(result_answer)

            # 로깅
            utils.print_qa(main, user_query, result_answer)

            # 9) 출처 문서(Reference) 표시
            for idx, doc in enumerate(source_docs, 1):
                filename = doc.metadata.get('source', 'unknown')
                page_num = doc.metadata.get('page', 'N/A')
                ref_title = f":blue[Reference {idx}: *{filename} - page.{page_num}*]"
                with st.popover(ref_title):
                    st.caption(doc.page_content)

if __name__ == "__main__":
    main()
