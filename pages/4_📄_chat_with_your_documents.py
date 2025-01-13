# # ###########하나만############

# import os
# import utils
# import streamlit as st
# from streaming import StreamHandler

# from dotenv import load_dotenv
# from langchain_teddynote import logging
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
# from langchain_community.vectorstores import DocArrayInMemorySearch, FAISS
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.prompts import PromptTemplate
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain


# # Load environment variables for API keys
# load_dotenv()

# # LangSmith tracking setup
# logging.langsmith("LANCHAIN-PROJECT")

# st.set_page_config(page_title="ChatPDF", page_icon="📄")
# st.header('청약스캐너')
# st.write('해당 청약 공고문에 대해 궁금한 점을 질문해보세요!')
# st.write('[![view source code ](https://img.shields.io/badge/view_source_code-gray?logo=github)](https://github.com/shashankdeshpande/langchain-chatbot/blob/master/pages/4_%F0%9F%93%84_chat_with_your_documents.py)')

# class CustomDocChatbot:

#     def __init__(self):
#         utils.sync_st_session()
#         self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
#         self.embedding_model = OpenAIEmbeddings()

#     def save_file(self, file):
#         folder = 'tmp'
#         if not os.path.exists(folder):
#             os.makedirs(folder)
        
#         file_path = f'./{folder}/{file.name}'
#         with open(file_path, 'wb') as f:
#             f.write(file.getvalue())
#         return file_path

#     @st.spinner('Analyzing documents..')
#     def setup_qa_chain(self, uploaded_files):
#         # Load documents
#         docs = []
#         for file in uploaded_files:
#             file_path = self.save_file(file)
#             loader = PyPDFLoader(file_path)
#             docs.extend(loader.load())
        
#         # Split documents and store in vector db
#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=1000,
#             chunk_overlap=200
#         )
#         splits = text_splitter.split_documents(docs)
#         vectordb = FAISS.from_documents(splits, self.embedding_model)

#         # Define retriever
#         retriever = vectordb.as_retriever(
#             search_type='mmr',
#             search_kwargs={'k': 2, 'fetch_k': 4}
#         )

#         # Setup memory for contextual conversation        
#         memory = ConversationBufferMemory(
#             memory_key='chat_history',
#             output_key='answer',
#             return_messages=True
#         )

#         # Setup LLM and QA chain
#         qa_chain = ConversationalRetrievalChain.from_llm(
#             llm=self.llm,
#             retriever=retriever,
#             memory=memory,
#             return_source_documents=True,
#             verbose=False
#         )
#         return qa_chain

#     @utils.enable_chat_history
#     def main(self):
        
#         # User Inputs
#         uploaded_files = st.sidebar.file_uploader(label='Upload PDF files', type=['pdf'], accept_multiple_files=True)
#         if not uploaded_files:
#             st.error("공고문을 PDF 파일로 업로드해주세요!")
#             st.stop()

#         user_query = st.chat_input(placeholder="무엇이든 질문해보세요!")

#         if uploaded_files and user_query:
#             qa_chain = self.setup_qa_chain(uploaded_files)

#             utils.display_msg(user_query, 'user')

#             with st.chat_message("assistant"):
#                 st_cb = StreamHandler(st.empty())
#                 result = qa_chain.invoke(
#                     {"question": user_query},
#                     {"callbacks": [st_cb]}
#                 )
#                 response = result["answer"]

#                 # Displaying the response in the chat
#                 st.session_state.messages.append({"role": "assistant", "content": response})
#                 st.write(response) 
#                 utils.print_qa(CustomDocChatbot, user_query, response)

#                 # to show references
#                 for idx, doc in enumerate(result['source_documents'], 1):
#                     filename = os.path.basename(doc.metadata['source'])
#                     page_num = doc.metadata['page']
#                     ref_title = f":blue[Reference {idx}: *{filename} - page.{page_num}*]"
#                     with st.popover(ref_title):
#                         st.caption(doc.page_content)

# if __name__ == "__main__":
#     obj = CustomDocChatbot()
#     obj.main()

# ###########기존############
    
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_community.vectorstores import DocArrayInMemorySearch
# from langchain_text_splitters import RecursiveCharacterTextSplitter


# st.set_page_config(page_title="ChatPDF", page_icon="📄")
# st.header('청약스캐너')
# st.write('해당 청약 공고문에 대해 궁금한 점을 질문해보세요!')
# st.write('[![view source code ](https://img.shields.io/badge/view_source_code-gray?logo=github)](https://github.com/shashankdeshpande/langchain-chatbot/blob/master/pages/4_%F0%9F%93%84_chat_with_your_documents.py)')

# class CustomDocChatbot:

#     def __init__(self):
#         utils.sync_st_session()
#         self.llm = utils.configure_llm()
#         self.embedding_model = utils.configure_embedding_model()

#     def save_file(self, file):
#         folder = 'tmp'
#         if not os.path.exists(folder):
#             os.makedirs(folder)
        
#         file_path = f'./{folder}/{file.name}'
#         with open(file_path, 'wb') as f:
#             f.write(file.getvalue())
#         return file_path

#     @st.spinner('Analyzing documents..')
#     def setup_qa_chain(self, uploaded_files):
#         # Load documents
#         docs = []
#         for file in uploaded_files:
#             file_path = self.save_file(file)
#             loader = PyPDFLoader(file_path)
#             docs.extend(loader.load())
        
#         # Split documents and store in vector db
#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=1000,
#             chunk_overlap=200
#         )
#         splits = text_splitter.split_documents(docs)
#         vectordb = DocArrayInMemorySearch.from_documents(splits, self.embedding_model)

#         # Define retriever
#         retriever = vectordb.as_retriever(
#             search_type='mmr',
#             search_kwargs={'k':2, 'fetch_k':4}
#         )

#         # Setup memory for contextual conversation        
#         memory = ConversationBufferMemory(
#             memory_key='chat_history',
#             output_key='answer',
#             return_messages=True
#         )

#         # Setup LLM and QA chain
#         qa_chain = ConversationalRetrievalChain.from_llm(
#             llm=self.llm,
#             retriever=retriever,
#             memory=memory,
#             return_source_documents=True,
#             verbose=False
#         )
#         return qa_chain

#     @utils.enable_chat_history
#     def main(self):

#         # User Inputs
#         uploaded_files = st.sidebar.file_uploader(label='Upload PDF files', type=['pdf'], accept_multiple_files=True)
#         if not uploaded_files:
#             st.error("Please upload PDF documents to continue!")
#             st.stop()

#         user_query = st.chat_input(placeholder="Ask me anything!")

#         if uploaded_files and user_query:
#             qa_chain = self.setup_qa_chain(uploaded_files)

#             utils.display_msg(user_query, 'user')

#             with st.chat_message("assistant"):
#                 st_cb = StreamHandler(st.empty())
#                 result = qa_chain.invoke(
#                     {"question":user_query},
#                     {"callbacks": [st_cb]}
#                 )
#                 response = result["answer"]
#                 st.session_state.messages.append({"role": "assistant", "content": response})
#                 utils.print_qa(CustomDocChatbot, user_query, response)

#                 # to show references
#                 for idx, doc in enumerate(result['source_documents'],1):
#                     filename = os.path.basename(doc.metadata['source'])
#                     page_num = doc.metadata['page']
#                     ref_title = f":blue[Reference {idx}: *{filename} - page.{page_num}*]"
#                     with st.popover(ref_title):
#                         st.caption(doc.page_content)

# if __name__ == "__main__":
#     obj = CustomDocChatbot()
#     obj.main()

# ###########모델선택############

import os
import utils
import streamlit as st
from streaming import StreamHandler

from dotenv import load_dotenv
from langchain_teddynote import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
from langchain_community.vectorstores import DocArrayInMemorySearch, FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Load environment variables for API keys
load_dotenv()

# LangSmith tracking setup
logging.langsmith("LANCHAIN-PROJECT")

st.set_page_config(page_title="ChatPDF", page_icon="📄")
st.header('📄청약스캐너')
st.write('해당 청약 공고문에 대해 궁금한 점을 질문해보세요!')
st.write('[![view source code ](https://img.shields.io/badge/view_source_code-gray?logo=github)](https://github.com/shashankdeshpande/langchain-chatbot/blob/master/pages/4_%F0%9F%93%84_chat_with_your_documents.py)')

class CustomDocChatbot:

    def __init__(self):
        utils.sync_st_session()
        
        # 모델 선택을 위한 코드 추가
        model_option = st.sidebar.selectbox(
            label="Select LLM Model",
            options=["gpt-3.5-turbo", "gpt-4", "llama3.1", "llama3.2", "gemini1", "gemini2"],
            index=0
        )
        
        # 선택된 모델을 기반으로 LLM 초기화
        if model_option == "gpt-3.5-turbo":
            self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        elif model_option == "gpt-4":
            self.llm = ChatOpenAI(model_name="gpt-4", temperature=0)
        elif model_option == "llama3.1":
            self.llm = ChatOpenAI(model_name="llama3.1", temperature=0)
        elif model_option == "llama3.2":
            self.llm = ChatOpenAI(model_name="llama3.2", temperature=0)
        elif model_option == "gemini1":  # 제미나이 모델 1 추가
            self.llm = ChatOpenAI(model_name="gemini1", temperature=0)
        else:  # 제미나이 모델 2 추가
            self.llm = ChatOpenAI(model_name="gemini2", temperature=0)
        
        self.embedding_model = OpenAIEmbeddings()

    def save_file(self, file):
        folder = 'tmp'
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        file_path = f'./{folder}/{file.name}'
        with open(file_path, 'wb') as f:
            f.write(file.getvalue())
        return file_path

    @st.spinner('Analyzing documents..')
    def setup_qa_chain(self, uploaded_files):
        # Load documents
        docs = []
        for file in uploaded_files:
            file_path = self.save_file(file)
            loader = PyPDFLoader(file_path)
            docs.extend(loader.load())
        
        # Split documents and store in vector db
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(docs)
        vectordb = FAISS.from_documents(splits, self.embedding_model)

        # Define retriever
        retriever = vectordb.as_retriever(
            search_type='mmr',
            search_kwargs={'k': 2, 'fetch_k': 4}
        )

        # Setup memory for contextual conversation        
        memory = ConversationBufferMemory(
            memory_key='chat_history',
            output_key='answer',
            return_messages=True
        )

        # Setup LLM and QA chain
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            verbose=False
        )
        return qa_chain

    @utils.enable_chat_history
    def main(self):
        
        # User Inputs
        uploaded_files = st.sidebar.file_uploader(label='Upload PDF files', type=['pdf'], accept_multiple_files=True)
        if not uploaded_files:
            st.error("공고문을 PDF 파일로 업로드해주세요!")
            st.stop()

        user_query = st.chat_input(placeholder="무엇이든 질문해보세요!")

        if uploaded_files and user_query:
            qa_chain = self.setup_qa_chain(uploaded_files)

            utils.display_msg(user_query, 'user')

            with st.chat_message("assistant"):
                st_cb = StreamHandler(st.empty())
                result = qa_chain.invoke(
                    {"question": user_query},
                    {"callbacks": [st_cb]}
                )
                response = result["answer"]

                # Displaying the response in the chat
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.write(response) 
                utils.print_qa(CustomDocChatbot, user_query, response)

                # to show references
                for idx, doc in enumerate(result['source_documents'], 1):
                    filename = os.path.basename(doc.metadata['source'])
                    page_num = doc.metadata['page']
                    ref_title = f":blue[Reference {idx}: *{filename} - page.{page_num}*]"
                    with st.popover(ref_title):
                        st.caption(doc.page_content)

if __name__ == "__main__":
    obj = CustomDocChatbot()
    obj.main()
