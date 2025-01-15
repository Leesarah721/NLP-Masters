# module_a.py

import os
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def save_file(file, folder="tmp"):
    """PDF 파일을 지정된 폴더에 저장하고 경로를 반환한다."""
    if not os.path.exists(folder):
        os.makedirs(folder)
    file_path = os.path.join(folder, file.name)
    with open(file_path, 'wb') as f:
        f.write(file.getvalue())
    return file_path

def load_pdfs(uploaded_files):
    """업로드된 여러 PDF 파일을 로드하여 langchain 문서 리스트를 생성한다."""
    docs = []
    for file in uploaded_files:
        file_path = save_file(file)
        loader = PyPDFLoader(file_path)   # 필요 시 PyMuPDFLoader 사용 가능
        docs.extend(loader.load())
    return docs

def split_documents(docs, chunk_size=1000, chunk_overlap=200):
    """문서 리스트를 chunk_size, chunk_overlap에 맞춰 분할."""
    splits_text = ["■","▣"]
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=splits_text
    )
    return text_splitter.split_documents(docs)