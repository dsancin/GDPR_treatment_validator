import streamlit as st
from utils.vectorstore_manager import initialize_session, render_upload_section
from utils.online_update import online_search_and_update

# -------------------- Inizializzazione --------------------
initialize_session()
render_upload_section()

# -------------------- Sezione parametri --------------------
st.sidebar.header("📄 Parametri aggiornamento online")

max_pages_edpb = st.sidebar.number_input(
    "Numero massimo pagine EDPB",
    min_value=1,
    max_value=50,
    value=3,
    step=1
)

max_pages_garante = st.sidebar.number_input(
    "Numero massimo pagine Garante Privacy",
    min_value=1,
    max_value=50,
    value=4,
    step=1
)

# -------------------- Bottone aggiornamento --------------------
if st.sidebar.button("Update documents online"):
    with st.spinner("Downloading and indexing documents..."):
        online_search_and_update(
            st.session_state.vectorstore,
            max_pages_edpb=max_pages_edpb,
            max_pages_garante=max_pages_garante
        )
    st.sidebar.success("Documents updated!")

st.set_page_config(page_title="HOME")
st.title("Dashboard GDPR treatment validator")
st.write(
    "Use menu on the left to navigate the pages: "
    "New treatment, Existing treatment."
)
