# MAPATGE SEMÀNTIC I VISUALITZACIÓ ANALÍTICA INTERACTIVA DE L’ESTRUCTURA TEMÀTICA D’UN REPOSITORI INSTITUCIONAL

## Visualització de l’anàlisi per tòpics mitjançant embeddings i *clústering* de densitat del text complet dels articles dipositats en accés obert al repositori de la Universitat Rovira i Virgili

Aquest repositori conté el **codi font de l’aplicació Streamlit** desenvolupada en el marc del **Treball Final de Màster (TFM)** del *Màster en Ciència de Dades* de la **:contentReference[oaicite:0]{index=0}** (curs acadèmic 2025–2026).

L’aplicació permet l’**exploració visual i analítica de l’estructura temàtica** d’un repositori institucional mitjançant tècniques d’*embedding* semàntic i *clústering* no supervisat, aplicades al **text complet** dels articles científics dipositats en accés obert al repositori de la **:contentReference[oaicite:1]{index=1}**.

---

## 🎯 Objectiu del projecte

L’objectiu principal és **analitzar, representar i interpretar l’estructura temàtica latent** d’un repositori institucional a partir del contingut textual dels documents, superant les limitacions de les classificacions manuals o basades exclusivament en metadades.

El projecte combina:
- **Embeddings semàntics contextuals (SBERT)**  
- **Reducció dimensional amb UMAP**  
- **Clústering de densitat amb HDBSCAN**  
- **Visualització interactiva orientada a l’anàlisi exploratòria**

---

## 🖥️ Aplicació web

🔗 **Aplicació Streamlit (en producció):**  
👉 https://app-tfm-ciencia-dades-2025.streamlit.app/

L’aplicació inclou:
- Visió general del corpus i dels indicadors clau
- Paisatge semàntic UMAP interactiu
- Anàlisi temàtica per clústers
- Evolució institucional temporal
- Descobriment i exploració de documents
- Mapa narratiu dels resultats

---

## 📚 Corpus i dades

- **Repositori institucional analitzat:**  
  🔗 https://repositori.urv.cat/ca/

- **Dades originals del corpus (text complet dels articles):**  
  Dipositades a **:contentReference[oaicite:2]{index=2}**  
  🔗 https://doi.org/10.5281/zenodo.18007973

---

## 💻 Codi i reproductibilitat

- **Codi informàtic per replicar l’estudi (TFM):**  
  Sánchez Martos, M. J. S. (2025).  
  *Codi informàtic per replicar l’estudi: MAPATGE SEMÀNTIC I VISUALITZACIÓ ANALÍTICA INTERACTIVA DE L’ESTRUCTURA TEMÀTICA D’UN REPOSITORI INSTITUCIONAL*.  
  Dipositat a **RDR (Recerca Digital de Catalunya)**.  
  🔗 https://doi.org/10.34810/data2634

Aquest repositori conté **exclusivament el codi de l’aplicació de visualització**, optimitzat per a l’execució en entorns de producció (Streamlit Cloud).

---

## 🧠 Metodologia (resum)

1. Extracció i neteja del text complet dels documents
2. Generació d’*embeddings* semàntics amb **SBERT**
3. Reducció dimensional amb **UMAP**
4. Clústering no supervisat amb **HDBSCAN**
5. Etiquetatge automàtic dels clústers
6. Generació d’artefactes analítics (parquet, JSON)
7. Visualització interactiva amb **Streamlit + Plotly**

---

## 👩‍🎓 Autoria i filiació

- **Autora:** María José Sánchez Martos  
- **ORCID:** https://orcid.org/0000-0001-6419-3268  
- **Filiació institucional:** CRAI, Universitat Rovira i Virgili  

---

## 📜 Llicència

El codi d’aquest repositori està publicat sota llicència:

**Creative Commons Attribution 4.0 International (CC BY 4.0)**  
🔗 https://creativecommons.org/licenses/by/4.0/

Es permet la reutilització, adaptació i redistribució del codi, sempre que se’n reconegui l’autoria.

---

## 🛠️ Requisits tècnics

- Python ≥ 3.10  
- Streamlit  
- Pandas, NumPy  
- Plotly  
- PyArrow / Fastparquet  

Consulta el fitxer `requirements.txt` per al detall complet de dependències.

---

## 📌 Notes finals

Aquest projecte s’emmarca en els principis de la **ciència oberta**, la **reproductibilitat** i la **visualització responsable de dades**, i pot ser reutilitzat com a base per a:
- anàlisi de repositoris institucionals,
- estudis bibliomètrics avançats,
- sistemes de descoberta semàntica,
- o quadres de comandament institucionals.

