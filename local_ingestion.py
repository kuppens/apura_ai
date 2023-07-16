from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
import time


def ingest_docs(pdf_doc):
    db = None
    loader = PyPDFLoader(pdf_doc)
    raw_documents = loader.load()
    print(f"{len(raw_documents)} documentos carregados!")

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1200,
        chunk_overlap=250,
        length_function=len
    )

    documents = text_splitter.split_documents(documents=raw_documents)

    print(f"Dividido em {len(documents)} chunks...")

    embeddings = OpenAIEmbeddings()
    
    if db is None:
        db = FAISS.from_documents(documents,embedding=embeddings)
    else:
        for doc in documents:
            db.add_documents([doc])
            time.sleep(1)
        
        return db
    print("Carregado com Sucesso!")
    return db
if __name__ == "__main__":
    ingest_docs()
