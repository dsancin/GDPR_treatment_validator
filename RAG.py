#RAG, migliora chunk size e retrieval 
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, CSVLoader, UnstructuredWordDocumentLoader
from langchain_text_splitters import TokenTextSplitter

#carica i documenti e processa con chunking
def load_and_process_documents(DOCUMENTS_PATH):
    """Carica e processa i documenti PDF con chunking"""
    print(f"Caricamento documenti da {DOCUMENTS_PATH}...")

    if not os.path.exists(DOCUMENTS_PATH):
        try:
            os.makedirs(DOCUMENTS_PATH)
            print(f"Creata directory {DOCUMENTS_PATH}. Inserire file PDF qui.")
            return []
        except OSError as e:
            print(f"Errore nella creazione della directory: {e}")
            return []

    # Carica tutti i PDF ricorsivamente
    pdf_loader = DirectoryLoader(
        DOCUMENTS_PATH,
        glob="**/*.pdf",  # Ricerca ricorsiva
        loader_cls=PyPDFLoader,
        show_progress=True
    )

    csv_loader = DirectoryLoader(
        DOCUMENTS_PATH,
        glob="**/*.csv",  # Ricerca ricorsiva
        loader_cls=CSVLoader,
        show_progress=True
    )

    docx_loader = DirectoryLoader(
        DOCUMENTS_PATH,
        glob="**/*.docx",  # Ricerca ricorsiva
        loader_cls=UnstructuredWordDocumentLoader,
        show_progress=True
    )

    documents = pdf_loader.load()
    documents.extend(csv_loader.load())
    documents.extend(docx_loader.load())

    if not documents:
        print("Nessun documento PDF trovato.")
        return []

    print(f"Trovati {len(documents)} documenti.")

    # Text splitter ottimizzato per contenuto PDF
    text_splitter = TokenTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len,
    )

    splits = text_splitter.split_documents(documents)
    print(f"Divisi in {len(splits)} chunk.")

    return splits


#crea vettori embedding
def initialize_vectorstore(splits, embeddings, PERSIST_DIRECTORY):
    """Inizializza il vector store con controllo delle duplicazioni"""

    # Verifica se il DB esiste già
    db_exists = os.path.exists(os.path.join(PERSIST_DIRECTORY, "chroma.sqlite3"))

    vectorstore = Chroma(
        collection_name="rag_collection",
        embedding_function=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )

    if splits and not db_exists:
        print("Prima indicizzazione del vector store...")
        vectorstore.add_documents(documents=splits)
        print(f"Indicizzati {len(splits)} chunk.")
    elif splits:
        print("Vector store esistente trovato. Per aggiornare, eliminare la directory chroma_db/")
        # Opzionale: aggiungi logica per upsert qui

    return vectorstore

def create_embeddings(EMBEDDING_MODEL_NAME):
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cuda" if os.getenv('CUDA_AVAILABLE') else "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return embeddings