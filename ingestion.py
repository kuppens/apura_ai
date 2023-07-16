import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone
from dotenv import load_dotenv

load_dotenv()

pinecone.init(
    api_key=os.environ['PINECONE_API_KEY'], environment="asia-southeast1-gcp-free"
)


def ingest_docs(pdf_doc):
    raw_documents = []
    loader = PyPDFLoader(pdf_doc)
    raw_documents = loader.load()
    print(f"{len(raw_documents)} carregados!")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=170, separators=["\n\n", "\n", " ", ""]
    )

    documents = text_splitter.split_documents(documents=raw_documents)

    print(f"Dividido em {len(documents)} chunks...")

    embeddings = OpenAIEmbeddings()
    Pinecone.from_documents(
        documents=documents, embedding=embeddings, index_name="dou-parser"
    )
    print("***** ADDED TO PINECONE VECTORSTORE *****")


if __name__ == "__main__":
    ingest_docs()
