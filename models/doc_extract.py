from langchain_community.document_loaders import PyPDFLoader, TextLoader
from pathlib import Path

def load_file():
    '''
    This uses Pypdf to load the pdf document and textloader for txt file
    '''
    uploads_dir = (Path(__file__).resolve().parent / "../uploads").resolve()
    docs = []

    for path in uploads_dir.rglob("*"):
        if path.suffix.lower() == ".pdf":
            docs.extend(PyPDFLoader(str(path)).load())
        elif path.suffix.lower() == ".txt":
            docs.extend(TextLoader(str(path)).load())

    return docs
