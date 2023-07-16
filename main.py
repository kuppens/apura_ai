from typing import Set
from backend.core import run_llm
from backend.custom_core import custom_run
import streamlit as st
# import ingestion
import local_ingestion
import tempfile
import os

st.set_page_config(page_title="Apuração", page_icon="🦁", layout="centered")

st.header("Apura.ai")
st.text("Conslida, resume e processa as normas mais relevantes no contexto de Auditoria.")

def create_sources_string(source_urls: Set[str]) -> str:
    if not source_urls:
        return ""
    sources_list = list(source_urls)
    sources_list.sort()
    sources_string = "sources:\n"
    for i, source in enumerate(sources_list):
        sources_string += f"{i+1}. {source}\n"
    return sources_string


if (
    "chat_answers_history" not in st.session_state
    and "user_prompt_history" not in st.session_state
    and "chat_history" not in st.session_state
    and "db_load" not in st.session_state
):
    st.session_state["chat_answers_history"] = []
    st.session_state["user_prompt_history"] = []
    st.session_state["chat_history"] = []
    st.session_state["db_load"] = []


def main (prompt):
    with st.sidebar:
        st.subheader("Seus Documentos")
        pdf_doc = st.file_uploader(
            "Faça o Upload de seus PDFs e clique em 'Processar'")
        
        temp_file_path = os.getcwd()
                
        if pdf_doc is not None:
            # Save the uploaded file to a temporary location
            temp_dir = tempfile.TemporaryDirectory()
            temp_file_path = os.path.join(temp_dir.name, pdf_doc.name)
            with open(temp_file_path, "wb") as temp_file:
                temp_file.write(pdf_doc.read())

            if st.button("Processar"):
                    if pdf_doc is not None:
                        with st.spinner("Processando na velocidade da luz... ⚡️"):
                            st.session_state.db_load = local_ingestion.ingest_docs(pdf_doc=temp_file_path)
                        st.success('Documento processado com Sucesso!', icon="✅")
                
    if prompt:
        with st.spinner("Gerando uma resposta muito top..."):
            generated_response = custom_run(
                query=prompt, db=st.session_state.db_load, chat_history=st.session_state["chat_history"]
            )

            # sources = set(
            #     [doc.metadata["source"] for doc in generated_response["source_documents"]]
            # )
            formatted_response = (
                f"{generated_response['output_text']}"
            )

            st.session_state.chat_history.append((prompt, generated_response['output_text']))
            st.session_state.user_prompt_history.append(prompt)
            st.session_state.chat_answers_history.append(formatted_response)

    if st.session_state["chat_answers_history"]:
        for generated_response, user_query in zip(
            st.session_state["chat_answers_history"],
            st.session_state["user_prompt_history"],
        ):
            with st.chat_message("user"):
                st.write(user_query)

            with st.chat_message("assistant"):
                st.write(generated_response)

prompt = st.chat_input(placeholder="Digite sua mensagem aqui...")

main(prompt)