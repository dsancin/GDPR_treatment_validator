#upload manuale dei documenti e gestione del vectorstore in sessione
import os
import tempfile
import streamlit as st
from RAG import load_and_process_documents, create_embeddings, initialize_vectorstore

PERSIST_DIRECTORY = "./new_chroma_db"
DOCUMENTS_PATH = "./data"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"


@st.cache_resource
def load_embeddings():
    return create_embeddings(EMBEDDING_MODEL_NAME)


@st.cache_resource
def load_initial_vectorstore():
    embeddings = load_embeddings()
    docs = load_and_process_documents(DOCUMENTS_PATH)
    return initialize_vectorstore(docs, embeddings, PERSIST_DIRECTORY)


def initialize_session():
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = load_initial_vectorstore()

    if "retriever" not in st.session_state:
        st.session_state.retriever = st.session_state.vectorstore.as_retriever(
            search_type="similarity"
        )


def render_upload_section():
    st.sidebar.header("Upload documents")

    uploaded_files = st.sidebar.file_uploader(
        "Carica PDF, TXT o DOCX",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True
    )

    if uploaded_files:
        with st.spinner("Doc"):
            with tempfile.TemporaryDirectory() as tmp_dir:

                for uploaded_file in uploaded_files:
                    file_path = os.path.join(tmp_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                new_docs = load_and_process_documents(tmp_dir)

                st.session_state.vectorstore.add_documents(new_docs)

                st.session_state.retriever = (
                    st.session_state.vectorstore.as_retriever(
                        search_type="similarity"
                    )
                )

        st.sidebar.success("Documents added!")