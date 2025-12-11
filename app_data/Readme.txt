app_data/
Benvinguda a la carpeta on resideixen tots els artefactes precomputats que permeten que l’aplicació Streamlit funcioni ràpidament, de manera lleugera i sense recalculem res en producció.
Són fitxers compactes, optimitzats i preparats per ser carregats directament al núvol.

🔍 Què hi ha dins aquesta carpeta?

A continuació tens un resum clar, visual i elegant de cada fitxer i la seva funció dins l’app. Els he agrupat perquè es vegi d’un sol cop d’ull.

📚 1. Corpus principal
df_docs_kw_enriched_with_labels.parquet

Taula principal del corpus (8.619 documents) amb:

Metadades bibliogràfiques

Departaments normalitzats

Keywords processades i kw_list_str

Clústers (cluster_hdbscan, cluster_label_auto, cluster_label_best)

Camps derivats (Dept_main, Dept_list)

És la base de gairebé totes les pàgines de l’app.

🗺️ 2. Coordenades UMAP (versió lleugera, SLIM)
df_docs_full_umap_simple.parquet

Només conserva:

doc_id

umap_x, umap_y

cluster_hdbscan

prob_hdbscan

💡 Aquest fitxer passa de 229 MB a 0.19 MB.
És essencial per carregar el Narrative Map i les visualitzacions de manera instantània en Streamlit Cloud.

🗂️ 3. Taules per a Document Discovery
doc_table_minimal.parquet

Versió ultra lleugera:

doc_id, Title, AnyPubARPC, Dept_main, cluster_hdbscan, handle_url

Ideal per a taules i cerques ràpides.

doc_table_enriched.parquet

Versió rica amb:

Departaments (main + col·laboradors)

Etiquetes automàtiques i “best label”

Resums i paraules clau

Any normalitzat

S’utilitza en:

Document Explorer

Cerca avançada

Contextualització de resultats

📈 4. Estadístiques i KPI del Dashboard
cluster_year_counts.parquet

Recompte anual per clúster.
Serveix per visualitzar l'evolució temporal.

overview_stats.parquet

Estadístiques bàsiques
(per nombre d’articles, distribucions, etc.)

dashboard_overview_kpis.parquet

KPIs ultra lleugers per la capçalera del dashboard.
(<5 KB)

🧠 5. Narrative Map
narrative_map_docs.parquet

Conté la fusió de:

Metadades

Coordenades UMAP

Clústers

Paraules clau

És l’artefacte central per al mapa narratiu i exploracions interactives.

📄 6. Resum del clustering
summary_03i_sbert_clustering.json

Resultats principals:

Paràmetres de SBERT + HDBSCAN

Nombre de clústers

Estadístiques del procés

Ull per quan vols mostrar informació tècnica o de mètode.

🧩 Relació general dels fitxers
app_data/
 ├── df_docs_kw_enriched_with_labels.parquet      ← Corpus principal
 ├── df_docs_full_umap_simple.parquet             ← UMAP lleuger
 ├── doc_table_enriched.parquet                   ← Navegació rica
 ├── doc_table_minimal.parquet                    ← Navegació ràpida
 ├── narrative_map_docs.parquet                   ← Narrative Map
 ├── cluster_year_counts.parquet                  ← Evolució temporal
 ├── overview_stats.parquet                       ← Estadístiques
 ├── dashboard_overview_kpis.parquet              ← KPIs inicials
 └── summary_03i_sbert_clustering.json            ← Resum SBERT

📝 Notes finals

Aquests fitxers són estàtics: l’app només els llegeix, no els modifica.

Si actualitzes els models, UMAP o el corpus, només cal regenerar aquests artefactes i tornar-los a pujar.

L'objectiu és optimitzar velocitat, memòria i fiabilitat a Streamlit Cloud.
