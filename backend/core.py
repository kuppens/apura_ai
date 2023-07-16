import os
from typing import Any, Dict, List

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain import VectorDBQA, OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS

from dotenv import load_dotenv

load_dotenv()


# pinecone.init(
#     api_key=os.environ['PINECONE_API_KEY'], environment="asia-southeast1-gcp-free"
# )


def run_llm(query: str, db, chat_history: List[Dict[str, Any]] = []):
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
    # docsearch = Pinecone.from_existing_index(
    #     embedding=embeddings,
    #     index_name="dou-parser",
    # )
    chat = ChatOpenAI(
        verbose=True,
        temperature=0,
    )

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)

    # qa = ConversationalRetrievalChain.from_llm(
    #     llm=chat, retriever=db.as_retriever(), memory=memory, chain_type='map_reduce'
    # )
    
    db.save_local("faiss_index")
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(query)

    qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=new_db.as_retriever())

    return qa({"question": query, "chat_history": chat_history})


if __name__ == "__main__":
    print(run_llm("What is LangChain?"))
