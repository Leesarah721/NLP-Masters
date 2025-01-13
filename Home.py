import streamlit as st

st.set_page_config(
    page_title="Langchain Chatbot",
    page_icon='💬',
    layout='wide'
)

st.header("💁청약스캐너")
st.markdown("""
### LangChain을 활용한 청약 챗봇
""")
st.write("""
[![view source code ](https://img.shields.io/badge/GitHub%20Repository-gray?logo=github)](https://github.com/shashankdeshpande/langchain-chatbot)
![Visitors](https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Flangchain-chatbot.streamlit.app&label=Visitors&labelColor=%235d5d5d&countColor=%231e7ebf&style=flat)
""")
st.write("""
    LangChain을 활용한 청약 챗봇은 복잡한 청약 정보를 효율적으로 정리하고, 사용자의 질문에 정확한 답변을 제공하기 위해 설계된 스마트 챗봇입니다. 특히, 청약 공고문 PDF를 업로드하면 이를 분석하여 원하는 정보를 쉽게 얻을 수 있도록 돕는 기능이 특징입니다.
    """)

st.markdown("""
    ##### **📄 PDF 업로드로 간편한 청약 정보 제공**
    기존의 청약 공고문은 복잡한 내용과 긴 문장으로 구성되어 있어 필요한 정보를 찾는 데 많은 시간이 소요됩니다. 하지만, 이 챗봇은 사용자가 청약 공고 PDF 파일을 업로드하면, 해당 문서를 자동으로 분석하여 사용자 질문에 맞는 정보를 신속히 제공합니다.

    **예를 들어:**

    - "1순위 신청 조건이 뭐야?"
    - "청약 신청 마감일은 언제야?"
    - "이 공고에서 필요한 서류를 알려줘."
    
    이처럼 사용자는 자유롭게 질문을 입력하고, 챗봇은 PDF 내용을 기반으로 정확한 답변을 제공합니다.      
    """)