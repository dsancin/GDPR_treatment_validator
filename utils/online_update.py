#ricerca online dei documenti e gestione del vectorstore in sessione
import os
import requests
import hashlib
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from RAG import load_and_process_documents

DOWNLOAD_DIR = "./downloaded_docs"
TRACKING_FILE = "./downloaded_docs/downloaded_hashes.txt"

EDPB_BASE_URL = "https://www.edpb.europa.eu/our-work-tools/documents/our-documents_en"
GARANTE_BASE_URL = "https://www.garanteprivacy.it/home/ricerca"

# ---------- HASH UTILS ----------
def get_file_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()

def load_known_hashes():
    if not os.path.exists(TRACKING_FILE):
        return set()
    with open(TRACKING_FILE, "r") as f:
        return set(line.strip() for line in f.readlines())

def save_hash(hash_value):
    with open(TRACKING_FILE, "a") as f:
        f.write(hash_value + "\n")

# ---------- PDF Extraction ----------
def extract_pdf_links_from_html(html, base_url):
    soup = BeautifulSoup(html, "html.parser")
    pdf_links = []

    for link in soup.find_all("a", href=True):
        href = link["href"]
        if ".pdf" in href.lower():
            full_url = urljoin(base_url, href)
            pdf_links.append(full_url)

    return pdf_links

# ---------- Scraper EDPB ----------
def scrape_edpb(max_pages=5):
    all_links = []
    for page in range(max_pages):
        url = f"{EDPB_BASE_URL}?page={page}"
        response = requests.get(url, timeout=30)
        if response.status_code != 200:
            break
        pdf_links = extract_pdf_links_from_html(response.text, url)
        if not pdf_links:
            break
        all_links.extend(pdf_links)
    return list(set(all_links))

# ---------- Scraper GARANTE ----------
def scrape_garante(max_pages=5):
    all_links = []
    base_params = (
        "p_p_id=g_gpdp5_search_GGpdp5SearchPortlet&"
        "p_p_lifecycle=0&"
        "p_p_state=normal&"
        "p_p_mode=view&"
        "_g_gpdp5_search_GGpdp5SearchPortlet_mvcRenderCommandName=%2FrenderSearch&"
        "_g_gpdp5_search_GGpdp5SearchPortlet_text=&"
        "_g_gpdp5_search_GGpdp5SearchPortlet_dataInizio=&"
        "_g_gpdp5_search_GGpdp5SearchPortlet_dataFine=&"
        "_g_gpdp5_search_GGpdp5SearchPortlet_paginaWeb=false&"
        "_g_gpdp5_search_GGpdp5SearchPortlet_allegato=false&"
        "_g_gpdp5_search_GGpdp5SearchPortlet_ordinamentoPer=DESC&"
        "_g_gpdp5_search_GGpdp5SearchPortlet_ordinamentoTipo=data"
    )
    for page in range(1, max_pages + 1):
        url = f"{GARANTE_BASE_URL}?{base_params}&_g_gpdp5_search_GGpdp5SearchPortlet_cur={page}"
        response = requests.get(url, timeout=30)
        if response.status_code != 200:
            break
        pdf_links = extract_pdf_links_from_html(response.text, url)
        if not pdf_links:
            break
        all_links.extend(pdf_links)
    return list(set(all_links))

# ---------- Download + Indicizzazione ----------
def download_and_index(pdf_links, vectorstore):
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    known_hashes = load_known_hashes()
    new_files = []

    for pdf_url in pdf_links:
        try:
            response = requests.get(pdf_url, timeout=60)
            if response.status_code != 200:
                continue
            file_hash = get_file_hash(response.content)
            if file_hash in known_hashes:
                continue
            filename = pdf_url.split("/")[-1]
            local_path = os.path.join(DOWNLOAD_DIR, filename)
            with open(local_path, "wb") as f:
                f.write(response.content)
            save_hash(file_hash)
            known_hashes.add(file_hash)
            new_files.append(local_path)
        except Exception as e:
            print(f"Errore su {pdf_url}: {e}")

    if not new_files:
        print("No new files.")
        return

    # Processa i documenti e append al vectorstore
    new_docs = load_and_process_documents(DOWNLOAD_DIR)
    vectorstore.add_documents(new_docs)
    try:
        vectorstore.persist()
    except:
        pass
    print("Update completed.")

# ---------- MAIN FUNCTION ----------
def online_search_and_update(vectorstore, max_pages_edpb, max_pages_garante):
    edpb_links = scrape_edpb(max_pages_edpb)
    garante_links = scrape_garante(max_pages_garante)
    all_links = list(set(edpb_links + garante_links))
    print(f"Trovati {len(all_links)} PDF totali.")
    download_and_index(all_links, vectorstore)