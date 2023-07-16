import os
from typing import Any, Dict, List

from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory




def custom_run(query: str, db, chat_history: List[Dict[str, Any]] = []):
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)

    
    db.save_local("faiss_index")
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(query)


    question_prompt_template = """Use the following portion of a long document to see if any of the text is relevant to answer the question. 
    Return any relevant text translated into Portugese.
    {context}
    Question: {question}
    Relevant text, if any, in Portugese:"""
    QUESTION_PROMPT = PromptTemplate(
        template=question_prompt_template, input_variables=["context", "question"]
    )

    combine_prompt_template = """Given the following extracted parts of a long document and a question, create a final answer Portugese. 
    If you don't know the answer, just say that you don't know. Don't try to make up an answer.

    QUESTION: {question}
    =========
    {summaries}
    =========
    Answer in Portuguese:"""
    COMBINE_PROMPT = PromptTemplate(
        template=combine_prompt_template, input_variables=["summaries", "question"]
    )
    chain = load_qa_chain(OpenAI(temperature=0), chain_type="map_reduce", return_map_steps=True, question_prompt=QUESTION_PROMPT, combine_prompt=COMBINE_PROMPT)
    return chain({"input_documents": docs, "question": query}, return_only_outputs=True)