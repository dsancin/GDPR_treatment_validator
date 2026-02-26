descrizione struttura file e funzionamento:
home.py è la home page della app.
RAG.py processa e carica i documenti
diagramma_casi_uso.png è il diagramma dei casi d'uso
ID_Compliance_Report.pdf è un test fatto per verificare funzionamento (prompt: la mia azienda si occupa di dati medici, voglio creare database contenente nome, cognome e malattie dei pazienti)

/pages/1_new_treatment_App.py è la pagina per i nuovi trattamenti
/pages/2_existing_treatment_App.py è la pagina per i trattamenti esistenti
/utils/vectorstore_manager.py crea il bottone per l'upload manuale di documenti
/utils/online updates.py crea il bottone per l'upload documenti online
/data è la cartella contente i documenti

Per far partire la app da CLI: streamlit run home.py

Per ora è stata testa solo la funzione di report per un nuovo trattamento, soddisfacente per essere primo tentativo
