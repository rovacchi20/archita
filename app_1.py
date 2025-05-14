import os
import io
import re
import tempfile
import json
import streamlit as st
import pandas as pd
import pdfplumber  # type: ignore
import fitz  # PyMuPDF
import easyocr  # type: ignore
import numpy as np  # type: ignore
import cv2  # type: ignore
from PIL import Image
import openai

# --- Configurazione pagina Streamlit ---
st.set_page_config(page_title="Automazione Bollette v2", layout="wide")

# --- Sidebar: API Key OpenAI ---
st.sidebar.header("OpenAI Configuration")
api_key = st.sidebar.text_input(
    "OpenAI API Key",
    type="password",
    help="Chiave per fallback AI (gpt-3.5-turbo)"
)
openai.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = "gpt-3.5-turbo"

# --- Inizializza EasyOCR reader italiano (CPU) ---
reader = easyocr.Reader(["it"], gpu=False)

# --- Inizializza EasyOCR reader italiano (CPU) ---
reader = easyocr.Reader(["it"], gpu=False)

# ————— Template-Driven Parsing —————
TEMPLATES_PATH = "templates.json"

def load_templates() -> dict:
    """Carica i template JSON per fornitore."""
    try:
        with open(TEMPLATES_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        st.warning(f"File {TEMPLATES_PATH} non trovato. Verifica il percorso o crea il file dei template.")
        return {}

def detect_provider(text: str, templates: dict) -> str:
    """Ritorna il nome del provider se trovato nel testo, altrimenti None."""
    for provider in templates:
        if re.search(provider, text, re.IGNORECASE):
            return provider
    return None

# Carico i template all’avvio
templates = load_templates()
# ————————————————————————————

# --- Utility funcs ---
def parse_euro_number(s: str) -> float:
    if '.' in s and ',' in s:
        s = s.replace('.', '').replace(',', '.')
    elif ',' in s:
        s = s.replace(',', '.')
    try:
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
    # 1) kWh (elettricità)
    kwh_matches = re.findall(r'([\d\.,]+)\s*kWh', text, re.IGNORECASE)
    if kwh_matches:
        data['consumo_kwh'] = parse_euro_number(kwh_matches[-1])
        data['energia'] = "energia elettrica"

    # 2) Smc (gas)
    smc_matches = re.findall(r'([\d\.,]+)\s*Smc', text, re.IGNORECASE)
    if smc_matches:
        data['consumo_smc'] = parse_euro_number(smc_matches[-1])
        data['energia'] = "gas naturale"

    # 3) Euro (sia € 123,45 che 123,45 € o EUR 123,45)
    euro_pattern = r'(?:€\s*([\d\.,]+)|([\d\.,]+)\s*€|EUR\s*([\d\.,]+))'
    euro_matches = re.findall(euro_pattern, text, re.IGNORECASE)
    if euro_matches:
        last = euro_matches[-1]
        # Seleziona nella seguente priorità: gruppo1, gruppo2, gruppo3
        valore = last[0] if last[0] else (last[1] if last[1] else last[2])
        data['costo_euro'] = parse_euro_number(valore)

    return data



def extract_values_from_pdf_table(path: str) -> dict:
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                for table in page.extract_tables() or []:
                    df = pd.DataFrame(table[1:], columns=table[0])
                    mask = df[df.columns[0]].str.contains("Totale", case=False, na=False)
                    if mask.any():
                        row = df[mask].iloc[-1]
                        m = re.search(r'([\d.,]+)', row[1])
                        return {'consumo_kwh': None,
                                'costo_euro': parse_euro_number(m.group(1)) if m else None}
    except:
        pass
    return {'consumo_kwh': None, 'costo_euro': None}


def extract_text_from_pdf(path: str) -> str:
    try:
        with pdfplumber.open(path) as pdf:
            return "\n".join(page.extract_text() or '' for page in pdf.pages)
    except:
        return ""


def ocr_pdf_scanned(path: str) -> str:
    text = ""
    doc = fitz.open(path)
    for page in doc:
        pix = page.get_pixmap(dpi=300)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        arr = np.array(img)
        res = reader.readtext(arr, detail=0)
        text += "\n".join(res) + "\n"
    doc.close()
    return text


def ocr_image_full(path: str) -> str:
    img = cv2.imdecode(
        np.frombuffer(open(path, "rb").read(), np.uint8),
        cv2.IMREAD_COLOR
    )
    res = reader.readtext(img, detail=0)
    return "\n".join(res)


def openai_extract_fields_text(text: str) -> dict:
    prompt = """
    <RUOLO> 
Sei un sistema di estrazione dati specializzato nell’analisi automatica di bollette elettriche e del gas. 
Hai competenze avanzate in OCR (EasyOCR), parsing testuale con regex e, se necessario, utilizzo di modelli di linguaggio OpenAI. 
Il tuo obiettivo è restituire un output JSON completo con i dati essenziali e rilevanti per ciascuna bolletta processata.
</RUOLO>
<ISTRUZIONI> 
- **Input**: testo grezzo estratto da PDF, immagini o altri documenti di bolletta. 
- **Output**: un oggetto JSON per ciascun file processato.
Deve contenere almeno:
- `consumo_kwh`: valore numerico (float) del consumo in kWh se energia elettrica
- `costo_euro`: valore numerico (float) dell’importo totale da pagare, in euro. Del totale dell'energia fatturata a bolletta. Guarda bene che sia il valore delal fattura e non numeri casuali. esempio: Totale Fattura € 1.234,09 o comunque il riferimento rispetto l'energia usata.
- Se disponibili, aggiungi anche:
- `energia`: specifica se la bolletta si riferisce a "energia elettrica" o "gas naturale" o qualsiasi altro tipo di energia
- `file`: nome esatto del file PDF da cui provengono i dati
- Se il consumo è espresso in Smc (per gas), usa `consumo_smc` al posto di `consumo_kwh`, riconosci amnceh gli altri titpi di energia e modifica nel caso il file per esportare il giusto valore.
- Altri metadati opzionali (es. cliente, periodo fatturazione, POD, scadenza) sono ammessi purché mantengano compatibilità JSON
- Applica regex per individuare l’ultima occorrenza di “kWh”, “Smc” e “€” e converti i numeri dal formato italiano:
   1. Rimuovi i punti “.” delle migliaia
   2. Sostituisci la virgola “,” con il punto “.”
   3. Cast a `float`
- Se il parsing diretto fallisce, usa un modello OpenAI a temperatura zero applicato sul paragrafo più pertinente per ottenere i dati richiesti in **strict JSON**.
- Restituisci **solo** JSON puro, senza commenti o markup extra.
- Se vengono forniti più documenti, restituisci un array JSON con un oggetto per ciascuna bolletta.
</ISTRUZIONI>
<DETTAGLI>
 - Per ogni numero in formato europeo:
   1. Rimuovi i punti “.” delle migliaia.
   2. Sostituisci la virgola “,” con il punto “.”.
   3. Cast a `float`. - Se “kWh” o “€” compaiono più volte, prendi l’ultima occorrenza.
 - In tabelle PDF, cerca la riga con “Totale” e prendi il valore nella colonna successiva.
- Nel fallback AI, invia al modello solo il paragrafo pertinente e richiedi output in **strict JSON**. - Assicurati che il JSON sia sintatticamente corretto e parsabile.
Se sono processi più file, processa tutti i file nella stessa maniera.
</DETTAGLI>
<TARGET> 
Un modello AI per l’estrazione automatica e scalabile dei dati principali dalle bollette per fini di monitoraggio e analisi energetica.
</TARGET>
<OUTPUT>
```json
[
{
"consumo_kwh": valore_float,
"costo_euro": valore_float,
"energia": "energia elettrica",
"file": "nome_file.pdf"
},
...
]
```"""
    # Invia al modello solo un messaggio con il prompt aggiornato
    resp = openai.ChatCompletion.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": prompt + "\n\n" + text}
        ],
        temperature=0
    )
    try:
        return json.loads(resp.choices[0].message.content)
    except Exception:
        return {'consumo_kwh': None, 'costo_euro': None}

def openai_extract_fields_text(text: str, filename: str) -> dict:
    prompt = f"""<RUOLO> 
Sei un sistema di estrazione dati specializzato nell’analisi automatica di bollette elettriche e del gas. 
Hai competenze avanzate in OCR (EasyOCR), parsing testuale con regex e, se necessario, utilizzo di modelli di linguaggio OpenAI. 
Il tuo obiettivo è restituire un output JSON completo con i dati essenziali e rilevanti per ciascuna bolletta processata.
</RUOLO>
<ISTRUZIONI> 
- **Input**: testo grezzo estratto da PDF, immagini o altri documenti di bolletta. 
- **Output**: un oggetto JSON per ciascun file processato.
Deve contenere almeno:
- `consumo_kwh`: valore numerico (float) del consumo in kWh se energia elettrica
- `costo_euro`: valore numerico (float) dell’importo totale da pagare, in euro. Del totale dell'energia fatturata a bolletta.
- Se disponibili, aggiungi anche:
- `energia`: specifica se la bolletta si riferisce a "energia elettrica" o "gas naturale" o qualsiasi altro tipo di energia
- `file`: nome esatto del file PDF da cui provengono i dati
- Se il consumo è espresso in Smc (per gas), usa `consumo_smc` al posto di `consumo_kwh`, riconosci amnceh gli altri titpi di energia e modifica nel caso il file per esportare il giusto valore.
- Altri metadati opzionali (es. cliente, periodo fatturazione, POD, scadenza) sono ammessi purché mantengano compatibilità JSON
- Applica regex per individuare l’ultima occorrenza di “kWh”, “Smc” e “€” e converti i numeri dal formato italiano:
   1. Rimuovi i punti “.” delle migliaia
   2. Sostituisci la virgola “,” con il punto “.”
   3. Cast a `float`
- Se il parsing diretto fallisce, usa un modello OpenAI a temperatura zero applicato sul paragrafo più pertinente per ottenere i dati richiesti in **strict JSON**.
- Restituisci **solo** JSON puro, senza commenti o markup extra.
- Se vengono forniti più documenti, restituisci un array JSON con un oggetto per ciascuna bolletta.
</ISTRUZIONI>
<DETTAGLI>
 - Per ogni numero in formato europeo:
   1. Rimuovi i punti “.” delle migliaia.
   2. Sostituisci la virgola “,” con il punto “.”.
   3. Cast a `float`. - Se “kWh” o “€” compaiono più volte, prendi l’ultima occorrenza.
 - In tabelle PDF, cerca la riga con “Totale” e prendi il valore nella colonna successiva.
- Nel fallback AI, invia al modello solo il paragrafo pertinente e richiedi output in **strict JSON**. - Assicurati che il JSON sia sintatticamente corretto e parsabile.
Se sono processi più file, processa tutti i file nella stessa maniera.
</DETTAGLI>
<TARGET> 
Un modello AI per l’estrazione automatica e scalabile dei dati principali dalle bollette per fini di monitoraggio e analisi energetica.
</TARGET>
<OUTPUT>
```json
[
{
"consumo_kwh": valore_float,
"costo_euro": valore_float,
"energia": "energia elettrica",
"file": "nome_file.pdf"
},
...
]
```
...
]

[
{{
  \"potenza_kw\": valore_float,
  \"consumo_kwh\": valore_float,
  \"tensione_v\": valore_float,
  \"corrente_a\": valore_float,
  \"macchinario\": \"codice_macchina_o_descrizione\",
  \"file\": \"{filename}\"
}}
]
</OUTPUT>
```"""

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt + "\n\n" + text}
        ],
        temperature=0
    )
    try:
        return json.loads(response.choices[0].message.content)
    except Exception:
        return {
            "potenza_kw": None,
            "consumo_kwh": None,
            "tensione_v": None,
            "corrente_a": None,
            "macchinario": None,
            "file": filename
        }

# --- App Tabs ---
tab1, tab2, tab3 = st.tabs([
    "Estrazione Dati",
    "Confronto Bollette",
    "Estrazione Immagine"
])

with tab1:
    st.header("📄 Estrazione Dati da Bollette")
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
            if ext == "pdf":
                txt = extract_text_from_pdf(tmp_path)
                st.expander(f"📄 Testo PDF ({name})").text_area("RAW", txt, height=300)
                data = parse_bill_text(txt)
                if data['costo_euro'] is None and openai.api_key:
                    txt_ocr = ocr_pdf_scanned(tmp_path)
                    st.expander(f"🖨️ OCR PDF ({name})").text_area("OCR", txt_ocr, height=300)
                    data = parse_bill_text(txt_ocr)
                record = {**data, 'file': name}
                os.remove(tmp_path)
                records.append(record)
        df = pd.DataFrame(records)
        st.subheader("Dati Estratti")
        st.dataframe(df)
        json_str = df.to_json(orient="records", force_ascii=False)
        st.download_button(
            "⬇️ Download JSON",
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
        import matplotlib.pyplot as plt
        dfc_plot = dfc.copy()
        dfc_plot['Originale'] = pd.to_numeric(dfc_plot['Originale'].astype(str).str.replace(',', '.'), errors='coerce')
        dfc_plot['Modificata'] = pd.to_numeric(dfc_plot['Modificata'].astype(str).str.replace(',', '.'), errors='coerce')
        dfc_plot = dfc_plot.dropna(subset=['Originale', 'Modificata'])
        dfc_plot = dfc_plot[~dfc_plot['Voce'].isin(['numero_fattura'])]
        dfc_plot.set_index('Voce', inplace=True)
        st.subheader("Grafico di Confronto (valori numerici)")
        fig, ax = plt.subplots(figsize=(10, 8))
        dfc_plot[['Originale', 'Modificata']].plot(kind='barh', ax=ax)
        ax.set_title("Confronto Voci Bolletta (valori numerici bilanciati)")
        ax.set_xlabel("Valore (€ / kWh / kW)")
        ax.set_ylabel("Voce")
        plt.tight_layout()
        st.pyplot(fig)

with tab3:
    st.header("📷 Estrazione Dati da Immagine e CSV")
    imgs = st.file_uploader(
        "Carica immagini (PNG, JPG, JPEG)",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key="imgs"
    )
    csvs = st.file_uploader(
        "Carica file CSV", 
        type=["csv"],
        accept_multiple_files=True,
        key="csvs"
    )
    if (imgs and len(imgs)>0) or (csvs and len(csvs)>0):
        records = []
        # Elaborazione immagini
        if imgs:
            for uploaded in imgs:
                name = uploaded.name
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(name)[1]) as tmp:
                    tmp.write(uploaded.getbuffer())
                    tmp_path = tmp.name
                txt_ocr = ocr_image_full(tmp_path)
                st.expander(f"🖨️ OCR Immagine ({name})").text_area("OCR", txt_ocr, height=300)
                data = parse_bill_text(txt_ocr)
                if (data.get('costo_euro') is None or data.get('energia') is None) and openai.api_key:
                    ai_data = openai_extract_fields_text(txt_ocr, name)
                    data = {**data, **ai_data}
                data['file'] = name
                os.remove(tmp_path)
                records.append(data)
        # Elaborazione CSV
        if csvs:
            for uploaded in csvs:
                name = uploaded.name
                try:
                    df_csv = pd.read_csv(uploaded)
                    # Converti righe in testo unico
                    txt_csv = df_csv.astype(str) \
                        .apply(lambda row: " ".join(row.values), axis=1) \
                        .str.cat(sep="\n")
                except Exception as e:
                    st.error(f"Errore lettura CSV {name}: {e}")
                    continue
                st.expander(f"📄 Testo CSV ({name})").text_area("CSV RAW", txt_csv, height=300)
                data = parse_bill_text(txt_csv)
                if (data.get('costo_euro') is None or data.get('energia') is None) and openai.api_key:
                    ai_data = openai_extract_fields_text(txt_csv, name)
                    data = {**data, **ai_data}
                data['file'] = name
                records.append(data)
        # Mostra e scarica risultati
        df = pd.DataFrame(records)
        st.subheader("Dati Estratti da Immagini e CSV")
        st.dataframe(df)
        json_str = df.to_json(orient="records", force_ascii=False)
        st.download_button(
            "⬇️ Download JSON",
            json_str,
            file_name="dati_immagini_csv.json",
            mime="application/json"
        )
