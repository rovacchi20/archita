# -*- coding: utf-8 -*-
import os
import io
import re
import tempfile
import json
import streamlit as st
import pandas as pd
import pdfplumber
import fitz  # PyMuPDF
import easyocr
import numpy as np
import cv2
from PIL import Image
import openai
import matplotlib.pyplot as plt

# Configurazione pagina Streamlit
st.set_page_config(page_title="Automazione Bollette v2", layout="wide")

# Sidebar: API Key OpenAI
st.sidebar.header("OpenAI Configuration")
api_key = st.sidebar.text_input(
    "OpenAI API Key",
    type="password",
    help="Chiave per fallback AI (gpt-3.5-turbo)"
)
openai.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = "gpt-3.5-turbo"

# Inizializza EasyOCR reader italiano (CPU)
reader = easyocr.Reader(["it"], gpu=False)

def detect_provider(text: str, templates: dict) -> str:
    for provider in templates:
        if re.search(provider, text, re.IGNORECASE):
            return provider
    return None


def parse_euro_number(s: str) -> float:
    try:
        s = s.replace('.', '').replace(',', '.') if ',' in s else s
        return float(s)
    except:
        return 0.0

def parse_bill_text(text: str) -> dict:
    data = {
        'consumo_kwh': None,
        'consumo_smc': None,
        'costo_euro': None,
        'energia': None
    }
    # Cerca il totale consumi in kWh (per energia elettrica)
    kwh_regex = r"Totale consumi fatturati:\s*([\d\.,]+)\s*kWh"
    kwh_matches = re.findall(kwh_regex, text, re.IGNORECASE)
    if kwh_matches:
        data['consumo_kwh'] = parse_euro_number(kwh_matches[-1])
        data['energia'] = "energia elettrica"
        
    # Cerca l'importo finale da pagare in euro
    euro_regex = r"Totale Importo da pagare.*?€\s*([\d\.,]+)"
    euro_matches = re.findall(euro_regex, text, re.IGNORECASE | re.DOTALL)
    if euro_matches:
        data['costo_euro'] = parse_euro_number(euro_matches[-1])
    
    return data

def clean_openai_output(text: str) -> dict:
    try:
        parsed = json.loads(text)
        # Validazione esempio: se il costo è eccessivamente alto, consideralo anomalo
        if parsed.get("costo_euro", 0) > 10000:
            raise ValueError("Valore anomalo")
        return parsed
    except Exception:
        return {}

def extract_text(path: str, tipo: str) -> str:
    if tipo == "pdf":
        try:
            with pdfplumber.open(path) as pdf:
                return "\n".join(page.extract_text() or '' for page in pdf.pages)
        except:
            return ""
    elif tipo == "ocr_pdf":
        doc = fitz.open(path)
        text = ""
        for page in doc:
            pix = page.get_pixmap(dpi=300)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            arr = np.array(img)
            res = reader.readtext(arr, detail=0)
            text += "\n".join(res) + "\n"
        doc.close()
        return text
    elif tipo == "img":
        img = cv2.imdecode(
            np.frombuffer(open(path, "rb").read(), np.uint8),
            cv2.IMREAD_COLOR
        )
        res = reader.readtext(img, detail=0)
        return "\n".join(res)
    return ""

def openai_extract_fields_text(text: str, filename: str = "") -> dict:
    prompt = (
        f"""<RUOLO> 
Sei un sistema di estrazione dati specializzato nell’analisi automatica di bollette elettriche e del gas. 
Hai competenze avanzate in OCR (EasyOCR), parsing testuale con regex e, se necessario, utilizzo di modelli di linguaggio OpenAI. 
Il tuo obiettivo è restituire un output JSON completo con i dati essenziali e rilevanti per ciascuna bolletta processata.
</RUOLO>
<ISTRUZIONI> 
- **Input**: testo grezzo estratto da PDF, immagini o altri documenti di bolletta. 
- **Output**: un oggetto JSON per ciascun file processato.
Deve contenere almeno:
- `consumo_kwh`: valore numerico (float) del consumo in kWh se energia elettrica
- `costo_euro`: valore numerico (float) dell’importo totale da pagare, in euro, relativo all'energia fatturata a bolletta.
- Se disponibili, aggiungi anche:
- `energia`: specifica se la bolletta si riferisce a "energia elettrica" o "gas naturale"
- `file`: nome esatto del file da cui provengono i dati
- Inoltre, applica regex per individuare l’ultima occorrenza di “kWh”, “Smc” e “€” e convertire i numeri dal formato italiano (rimuovi i punti per le migliaia, sostituisci la virgola con il punto e cast a float).
- Se il parsing diretto fallisce, usa un modello OpenAI a temperatura zero applicato sul paragrafo più pertinente per ottenere i dati richiesti in **strict JSON**.
</ISTRUZIONI>
<DETTAGLI>
 - Per ogni numero in formato europeo:
   1. Rimuovi i punti (“.”) delle migliaia.
   2. Sostituisci la virgola (“,”) con il punto (“.”).
   3. Cast a `float`. Se “kWh” o “€” compaiono più volte, prendi l’ultima occorrenza.
 - In tabelle PDF, cerca la riga con “Totale” e prendi il valore nella colonna successiva.
- Nel fallback AI, invia al modello solo il paragrafo pertinente e richiedi output in **strict JSON**.
</DETTAGLI>
<TARGET> 
Un modello AI per l’estrazione automatica e scalabile dei dati principali dalle bollette per fini di monitoraggio e analisi energetica.
</TARGET>
"Dai il numero totale dei kWh consumati (somma se su più righe) e l'importo finale da pagare. "
"Ignora numeri parziali o intermedi. Riporta solo:\n"
"- consumo_kwh: float\n"
"- costo_euro: float\n"
f"- file: \"{filename}\"\n\n"

"""
    )
    response = openai.ChatCompletion.create(
        model=OPENAI_MODEL,
        messages=[{"role": "system", "content": prompt + text}],
        temperature=0
    )
    content = response.choices[0].message.content
    st.subheader("🧠 Risposta AI (debug)")
    st.code(content, language="json")
    return clean_openai_output(content)

# --- App Tabs ---
tab1, tab2, tab3 = st.tabs([
    "Estrazione Dati",
    "Confronto Bollette",
    "Estrazione Immagine"
])

with tab1:
    st.header("📄Estrazione Dati da Bollette")
    uploads = st.file_uploader(
        "Carica file (PDF, PNG, JPG, TXT)",
        type=["pdf","png","jpg","jpeg","txt"],
        accept_multiple_files=True
    )
    if uploads:
        records = []
        for uploaded in uploads:
            name = uploaded.name
            ext = name.rsplit(".",1)[-1].lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
                tmp.write(uploaded.getbuffer())
                tmp_path = tmp.name
            # Estrai il testo grezzo in base al tipo: pdf oppure img
            tipo = "pdf" if ext == "pdf" else "img"
            txt = extract_text(tmp_path, tipo)
            st.expander(f"Testo {ext.upper()} ({name})").text_area("RAW", txt, height=300)
            # Utilizza direttamente il modulo AI per estrarre i dati dal testo
            data = openai_extract_fields_text(txt, name)
            st.success(f"✅ Dati AI estratti per {name}")
            record = {**data, 'file': name}
            os.remove(tmp_path)
            records.append(record)
        df = pd.DataFrame(records)
        st.subheader("Dati Estratti")
        st.dataframe(df)
        json_str = df.to_json(orient="records", force_ascii=False)
        st.download_button(
            "\u2b07\ufe0f Download JSON",
            json_str,
            file_name="dati_bollette.json",
            mime="application/json"
        )

with tab2:
    st.header("🔍 Confronto Bollette")
    col1, col2 = st.columns(2)
    with col1:
        json1 = st.file_uploader("Carica JSON Bolletta Originale", type=["json"], key="j1")
    with col2:
        json2 = st.file_uploader("Carica JSON Bolletta Modificata", type=["json"], key="j2")

    if json1 and json2:
        data1 = json.load(json1)
        data2 = json.load(json2)
        if isinstance(data1, list): data1 = data1[0]
        if isinstance(data2, list): data2 = data2[0]
        keys = set(data1.keys()) | set(data2.keys())
        keys.discard('file')
        comp = []
        for k in sorted(keys):
            comp.append({
                'Voce': k,
                'Originale': data1.get(k),
                'Modificata': data2.get(k)
            })
        dfc = pd.DataFrame(comp)
        st.subheader("Tabella Comparativa")
        st.dataframe(dfc)

        dfc_plot = dfc.copy()
        dfc_plot['Originale'] = pd.to_numeric(dfc_plot['Originale'].astype(str).str.replace(',', '.'), errors='coerce')
        dfc_plot['Modificata'] = pd.to_numeric(dfc_plot['Modificata'].astype(str).str.replace(',', '.'), errors='coerce')
        dfc_plot = dfc_plot.dropna(subset=['Originale', 'Modificata'])
        dfc_plot.set_index('Voce', inplace=True)
        st.subheader("Grafico di Confronto (valori numerici)")
        fig, ax = plt.subplots(figsize=(10, 8))
        dfc_plot[['Originale', 'Modificata']].plot(kind='barh', ax=ax)
        ax.set_title("Confronto Voci Bolletta")
        ax.set_xlabel("Valore")
        ax.set_ylabel("Voce")
        st.pyplot(fig)

with tab3:
    st.header("📷 Estrazione Immagine e CSV")
    imgs = st.file_uploader("Carica immagini (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    csvs = st.file_uploader("Carica file CSV", type=["csv"], accept_multiple_files=True)

    if imgs or csvs:
        records = []
        if imgs:
            for uploaded in imgs:
                name = uploaded.name
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(name)[1]) as tmp:
                    tmp.write(uploaded.getbuffer())
                    tmp_path = tmp.name
                txt_ocr = extract_text(tmp_path, "img")
                # Non esponiamo il testo OCR direttamente
                data = parse_bill_text(txt_ocr)
                if (data.get('costo_euro') is None or data.get('energia') is None) and openai.api_key:
                    ai_data = openai_extract_fields_text(txt_ocr, name)
                    st.success(f"✅ Dati AI estratti per {name}")
                    data = {**data, **ai_data}
                data['file'] = name
                os.remove(tmp_path)
                records.append(data)

        if csvs:
            for uploaded in csvs:
                name = uploaded.name
                try:
                    df_csv = pd.read_csv(uploaded)
                    # Combina tutte le righe in un unico testo
                    txt_csv = df_csv.astype(str).apply(lambda row: " ".join(row.values), axis=1).str.cat(sep="\n")
                except Exception as e:
                    st.error(f"Errore lettura CSV {name}: {e}")
                    continue
                # Non esponiamo il testo CSV direttamente
                data_csv = parse_bill_text(txt_csv)
                if (data_csv.get('costo_euro') is None or data_csv.get('energia') is None) and openai.api_key:
                    ai_data_csv = openai_extract_fields_text(txt_csv, name)
                    st.success(f"✅ Dati AI estratti per {name} (CSV)")
                    data_csv = {**data_csv, **ai_data_csv}
                data_csv['file'] = name
                records.append(data_csv)

        df = pd.DataFrame(records)
        st.subheader("Dati Estratti da Immagini e CSV")
        st.dataframe(df)
        json_str = df.to_json(orient="records", force_ascii=False)
        st.download_button("\u2b07\ufe0f Download JSON", json_str, file_name="dati_immagini_csv.json", mime="application/json")

# Footer
st.markdown("---")
st.markdown("<em>Powered by Hego0ne</em>", unsafe_allow_html=True)
