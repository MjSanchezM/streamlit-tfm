# VisualAnalytics/Paths_Project.py
# ------------------------------------------------------------------
# Paths únics i oficials de l'aplicació Streamlit
# TOTS els artefactes consumits per l'app han d'estar a /app_data
# ------------------------------------------------------------------

from pathlib import Path

# ============================================================
# 1. Directori arrel de la visualització
# ============================================================
# Aquest fitxer està a: VisualAnalytics / Paths_Project.py
# → BASE_DIR = VisualAnalytics
BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# 2. Directoris clau de l'app
# ============================================================

APP_DATA_DIR = BASE_DIR / "app_data"
PAGES_DIR    = BASE_DIR / "pages"
TOOLS_DIR    = BASE_DIR / "tools"
CONFIG_DIR   = BASE_DIR / ".streamlit"


# ============================================================
# 3. Artefactes de dades (TOTS dins app_data)
# ============================================================

# ---- Corpus principal enriquit ----
DOCS_ENRICHED_FILE = "https://rovira-my.sharepoint.com/:u:/g/personal/39893407-e_epp_urv_cat/IQDEaBDKzwQfQ5eamm1cFS0eAQKLldqyL5EWnpDXJ-_E6Ss?download=1"


# ---- KPIs / Overview ----
DASHBOARD_KPIS_FILE      = APP_DATA_DIR / "dashboard_overview_kpis.parquet"
OVERVIEW_STATS_FILE      = APP_DATA_DIR / "overview_stats.parquet"

# ---- Agregats temporals i institucionals ----
CLUSTER_YEAR_COUNTS_FILE = APP_DATA_DIR / "cluster_year_counts.parquet"

# ---- Taules per Document discovery ----
DOC_TABLE_MINIMAL_FILE   = APP_DATA_DIR / "doc_table_minimal.parquet"
DOC_TABLE_ENRICHED_FILE  = APP_DATA_DIR / "doc_table_enriched.parquet"

# ---- Narrative Map ----
NARRATIVE_FILE           = APP_DATA_DIR / "narrative_map_docs.parquet"

# ---- UMAP 2D complet (Semantic landscape) ----
UMAP_PARQUET_FILE        = APP_DATA_DIR / "df_docs_full_umap_simple.parquet"

# ---- Altres artefactes opcionals ----
SUMMARY_SBERT_FILE       = APP_DATA_DIR / "summary_03i_sbert_clustering.json"


# ============================================================
# 4. Helper de validació (opcional però recomanat)
# ============================================================

def check_app_data(strict: bool = False) -> None:
    """
    Comprova que la carpeta app_data existeix i
    informa dels fitxers que falten.

    strict=True → llança error si falta algun fitxer
    """
    required_files = [
        DOCS_ENRICHED_FILE,
        DASHBOARD_KPIS_FILE,
        OVERVIEW_STATS_FILE,
        CLUSTER_YEAR_COUNTS_FILE,
        DOC_TABLE_MINIMAL_FILE,
        DOC_TABLE_ENRICHED_FILE,
        NARRATIVE_FILE,
        UMAP_PARQUET_FILE,
        SUMMARY_SBERT_FILE,
    ]

    missing = [p.name for p in required_files if not p.exists()]

    if missing:
        msg = (
            "⚠️  Fitxers que falten a app_data:\n"
            + "\n".join(f" - {m}" for m in missing)
        )
        if strict:
            raise FileNotFoundError(msg)
        else:
            print(msg)
    else:
        print("✅ app_data OK — tots els artefactes presents.")

