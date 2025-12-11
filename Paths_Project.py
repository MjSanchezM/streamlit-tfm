# Paths_Project.py
from pathlib import Path

# ----------------------------------------------------------------------
# Directori base del projecte (arrel del repo streamlit-tfm)
# ----------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
REPO_DIR = THIS_FILE.parent          # arrel del repo a Streamlit Cloud
APP_DATA_DIR = REPO_DIR / "app_data"

# Si algun fitxer fa servir tools/plot_style.py en local, aquí mantenim la ruta
TOOLS_DIR = REPO_DIR / "tools"

# ----------------------------------------------------------------------
# Artefactes per a l'app (sempre des de app_data)
# ----------------------------------------------------------------------

# Corpus principal “aprimant” + metadades
DOCS_ENRICHED_FILE = APP_DATA_DIR / "df_docs_kw_enriched_with_labels.parquet"

# ---------- Overview / Home ----------
DASHBOARD_KPIS_FILE     = APP_DATA_DIR / "dashboard_overview_kpis.parquet"
OVERVIEW_STATS_FILE     = APP_DATA_DIR / "overview_stats.parquet"

# ---------- Evolució institucional ----------
CLUSTER_YEAR_COUNTS_FILE = APP_DATA_DIR / "cluster_year_counts.parquet"
# Alias per si algun codi antic encara fa servir aquest nom:
CLUSTER_YEAR_FILE        = CLUSTER_YEAR_COUNTS_FILE

# ---------- Document discovery ----------
DOC_TABLE_MINIMAL_FILE  = APP_DATA_DIR / "doc_table_minimal.parquet"
DOC_TABLE_ENRICHED_FILE = APP_DATA_DIR / "doc_table_enriched.parquet"

# ---------- UMAP / Narrative map ----------
UMAP_PARQUET_FILE = APP_DATA_DIR / "df_docs_full_umap_simple.parquet"
NARRATIVE_FILE    = APP_DATA_DIR / "narrative_map_docs.parquet"

# ---------- Resum del model SBERT + HDBSCAN ----------
SUMMARY_03I_FILE  = APP_DATA_DIR / "summary_03i_sbert_clustering.json"

