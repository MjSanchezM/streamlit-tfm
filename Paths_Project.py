# Paths_Project.py
from pathlib import Path

# ----------------------------------------------------------------------
# Directori base del projecte visual (VisualAnalytics)
# ----------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
VISUAL_DIR = THIS_FILE.parent          # carpeta VisualAnalytics
REPO_DIR = VISUAL_DIR.parent           # arrel del repo (TFM/GitHub)
APP_DATA_DIR = VISUAL_DIR / "app_data"

# Si tens un directori de tools:
TOOLS_DIR = VISUAL_DIR / "tools"

# ----------------------------------------------------------------------
# Artefactes per a l'app (sempre des de app_data)
# ----------------------------------------------------------------------

# Corpus principal “aprimant” + metadades
DOCS_ENRICHED_FILE = APP_DATA_DIR / "df_docs_kw_enriched_with_labels.parquet"

# Taules per a les diferents pàgines
DOC_TABLE_MINIMAL_FILE   = APP_DATA_DIR / "doc_table_minimal.parquet"
DOC_TABLE_ENRICHED_FILE  = APP_DATA_DIR / "doc_table_enriched.parquet"
CLUSTER_YEAR_FILE        = APP_DATA_DIR / "cluster_year_counts.parquet"
OVERVIEW_STATS_FILE      = APP_DATA_DIR / "overview_stats.parquet"

# Embeddings 2D UMAP per documents (versió slim)
UMAP_PARQUET_FILE = APP_DATA_DIR / "df_docs_full_umap_simple.parquet"

# Mapa narratiu (ja fusionat docs + UMAP)
NARRATIVE_FILE = APP_DATA_DIR / "narrative_map_docs.parquet"

# Resum del model SBERT + HDBSCAN
SUMMARY_03I_FILE = APP_DATA_DIR / "summary_03i_sbert_clustering.json"






