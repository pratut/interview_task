from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from typing import List
from langchain.schema import Document

def recursive_chunk(docs: List[Document], chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)

def fixed_size_chunk(docs: List[Document], chunk_size=500, chunk_overlap=50):
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)

chunking_methods = {
    "recursive": recursive_chunk,
     "fixed": fixed_size_chunk,
}
