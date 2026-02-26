import streamlit as st
from RAG import load_and_process_documents, create_embeddings, initialize_vectorstore
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_ollama import ChatOllama
from utils.vectorstore_manager import initialize_session, render_upload_section
from utils.online_update import online_search_and_update
from fpdf import FPDF

# -------------------- Inizializzazione --------------------
initialize_session()
render_upload_section()

# -------------------- Parametri aggiornamento online --------------------
st.sidebar.header("Update documents online")
max_pages_edpb = st.sidebar.number_input("Max number of pages EDPB", min_value=1, max_value=50, value=3, step=1)
max_pages_garante = st.sidebar.number_input("Max number of pages Garante Privacy", min_value=1, max_value=50, value=4, step=1)

if st.sidebar.button("Update documents online"):
    with st.spinner("Downloading and indexing documents..."):
        online_search_and_update(
            st.session_state.vectorstore,
            max_pages_edpb=max_pages_edpb,
            max_pages_garante=max_pages_garante
        )
    st.sidebar.success("Documents updated!")

# -------------------- Vectorstore --------------------
PERSIST_DIRECTORY = "./new_chroma_db"
DOCUMENTS_PATH = "./data"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

docs = load_and_process_documents(DOCUMENTS_PATH)
embeddings = create_embeddings(EMBEDDING_MODEL_NAME)
vectorstore = initialize_vectorstore(docs, embeddings, PERSIST_DIRECTORY)
retriever = vectorstore.as_retriever(search_type="similarity")

# -------------------- LangGraph RAG --------------------
from typing import TypedDict, Any

class RAGstate(TypedDict):
    query: str
    retriever: Any
    answer: str
    route: str
    documents: Any

# LLM
llm = ChatOllama(model="gemma3:1b", temperature=0)

# Prompt per report ID Compliance
prompt = ChatPromptTemplate.from_template("""
Sei un assistente esperto in data compliance e privacy. 
Genera un report di ID Compliance basato sulla descrizione del trattamento fornita. 
Rispondi in maniera strutturata con le seguenti sezioni:

1. Sintesi del trattamento
2. Normativa applicabile
3. Identifica categorie di possibili dati coinvolti
4. Rischi principali
6. Misure di mitigazione suggerite
7. Stato compliance attuale
8. Note e raccomandazioni

Se il contesto dei documenti forniti non è sufficiente, indica chiaramente che le informazioni non sono disponibili.

Contesto documenti rilevanti:
{context}

Descrizione trattamento:
{query}
""")

# -------------------- Node Functions --------------------
def query_router(state: RAGstate) -> RAGstate:
    query = state["query"].lower()
    keywords = ["documento", "pdf", "file", "manuale", "contesto", "testo"]
    return {"route": "rag"} if any(k in query for k in keywords) else {"route": "direct"}

def direct_generate_node(state: RAGstate) -> RAGstate:
    response = llm.invoke(f"Genera un report ID Compliance per il seguente trattamento:\n{state['query']}")
    return {"answer": response.content, "route": "Query gestita internamente"}

def fetch_node(state: RAGstate) -> RAGstate:
    return {"query": state["query"], "retriever": retriever, "route": "Query gestita con RAG"}

def generate_node(state: RAGstate) -> RAGstate:
    qa_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(state["retriever"], qa_chain)
    result = rag_chain.invoke({"input": state["query"]})
    return {"answer": result["answer"], "documents": result["context"], "route": state["route"]}

# Build Graph
graph = StateGraph(RAGstate)
graph.add_node("router", query_router)
graph.add_node("fetch", fetch_node)
graph.add_node("rag_generate", generate_node)
graph.add_node("direct_generate", direct_generate_node)
graph.set_entry_point("router")
graph.add_conditional_edges("router", lambda state: state["route"], {"rag": "fetch", "direct": "direct_generate"})
graph.add_edge("fetch", "rag_generate")
graph.add_edge("rag_generate", END)
graph.add_edge("direct_generate", END)
rag_app = graph.compile()

# -------------------- PDF Generator --------------------
def generate_pdf(report_text, filename="report_compliance.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Titolo
    pdf.set_font("Arial", "B", 16)
    pdf.multi_cell(0, 10, "ID Compliance Report", align="C")
    pdf.ln(10)
    
    # Corpo del report
    pdf.set_font("Arial", "", 12)
    lines = report_text.split("\n")
    for line in lines:
        pdf.multi_cell(0, 8, line)
    
    pdf_bytes = pdf.output(dest="S").encode("latin1")
    return pdf_bytes

# -------------------- Streamlit UI --------------------
st.title("New Treatment Compliance Report")

if "last_query" not in st.session_state:
    st.session_state.last_query = ""

with st.form("query_form", clear_on_submit=True):
    treatment_description = st.text_area("Descrivi il trattamento dati")
    submitted = st.form_submit_button("Genera report ID Compliance")

if submitted and treatment_description:
    with st.spinner("Generazione report in corso..."):
        result = rag_app.invoke({"query": treatment_description})
    st.session_state.last_query = treatment_description

    st.subheader("Descrizione trattamento:")
    st.write(st.session_state.last_query)

    st.subheader("Percorso seguito:")
    st.info(result["route"])

    st.subheader("Report ID Compliance:")
    st.code(result["answer"])

    # Genera PDF e bottone per scaricarlo
    pdf_data = generate_pdf(result["answer"])
    st.download_button(
        label="Scarica report PDF",
        data=pdf_data,
        file_name="ID_Compliance_Report.pdf",
        mime="application/pdf"
    )