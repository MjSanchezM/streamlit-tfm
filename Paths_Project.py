# Paths_Project.py
# ------------------------------------------------------------------
# Paths únics i oficials de l'aplicació Streamlit
# Versió adaptada per a Streamlit Cloud:
# - Les dades grans viuen a OneDrive URV i es carreguen via URL
# ------------------------------------------------------------------

from pathlib import Path

# ============================================================
# 1. Directori arrel de la visualització
# ============================================================
# Aquest fitxer està a l'arrel del repo: /Paths_Project.py
BASE_DIR = Path(__file__).resolve().parent

# ============================================================
# 2. Directoris clau de l'app (locals dins del repo)
# ============================================================
# Es mantenen per compatibilitat, tot i que app_data ja no s'usa com a carpeta local
APP_DATA_DIR = BASE_DIR / "app_data"   # ja no contindrà els parquets al núvol
PAGES_DIR    = BASE_DIR / "pages"
CONFIG_DIR   = BASE_DIR / ".streamlit"


# ============================================================
# 3. Artefactes de dades (ARA: URLs de OneDrive)
# ============================================================

# ---- Corpus principal enriquit ----
# df_docs_kw_enriched_with_labels.parquet
DOCS_ENRICHED_FILE = (
    "https://rovira-my.sharepoint.com/:u:/g/personal/"
    "39893407-e_epp_urv_cat/IQDEaBDKzwQfQ5eamm1cFS0eAQKLldqyL5EWnpDXJ-_E6Ss?download=1"
)

# ---- KPIs / Overview ----
# dashboard_overview_kpis.parquet
DASHBOARD_KPIS_FILE = (
    "https://rovira-my.sharepoint.com/:u:/g/personal/"
    "39893407-e_epp_urv_cat/IQALsi3VV5chTLCDUybLpDpYAbGUvJVVlShS-DaE8omOrCk?download=1"
)

# overview_stats.parquet
OVERVIEW_STATS_FILE = (
    "https://rovira-my.sharepoint.com/:u:/g/personal/"
    "39893407-e_epp_urv_cat/IQAhi1h_WAzZTZL5DRuGhmVWAfQazmmZ-sAne56TuZVVrTY?download=1"
)

# ---- Agregats temporals i institucionals ----
# cluster_year_counts.parquet
CLUSTER_YEAR_COUNTS_FILE = (
    "https://rovira-my.sharepoint.com/:u:/g/personal/"
    "39893407-e_epp_urv_cat/IQDAgSyayyeBTJLiYeudn_8uAdlpu11R1P4zMwAeUxeWhIY?download=1"
)

# ---- Taules per Document discovery ----
# doc_table_minimal.parquet
DOC_TABLE_MINIMAL_FILE = (
    "https://rovira-my.sharepoint.com/:u:/g/personal/"
    "39893407-e_epp_urv_cat/IQAJXypwEmEiQLBX5mAfbt7wAdGAA7XBsRYk3rP8BWoT6L8?download=1"
)

# doc_table_enriched.parquet
DOC_TABLE_ENRICHED_FILE = (
    "https://rovira-my.sharepoint.com/:u:/g/personal/"
    "39893407-e_epp_urv_cat/IQDsjxCHj0x5RLNAY_Y72zgUAal38C328AIfPrDPmlQ4_VE?download=1"
)

# ---- Narrative Map ----
# narrative_map_docs.parquet
NARRATIVE_FILE = (
    "https://rovira-my.sharepoint.com/:u:/g/personal/"
    "39893407-e_epp_urv_cat/IQB_KhItPpwqSoQKqmAyfy_OAR7Cia50rsLlnMZ415bj-fY?download=1"
)

# ---- UMAP 2D complet (Semantic landscape) ----
# df_docs_full_umap_simple.parquet
UMAP_PARQUET_FILE = (
    "https://rovira-my.sharepoint.com/:u:/g/personal/"
    "39893407-e_epp_urv_cat/IQDjiWjm0WSEQLa-gDDnW_yxAVgjlQpeDp5KnhGcrg6xI6E?download=1"
)

# ---- Altres artefactes opcionals ----
# summary_03i_sbert_clustering.json
SUMMARY_SBERT_FILE = (
    "https://rovira-my.sharepoint.com/:u:/g/personal/"
    "39893407-e_epp_urv_cat/IQC7Di6LsvZbQ4R2GMvAIhRHAcwLpbIr7Tco5wvSJR4Umts?download=1"
)


# ============================================================
# 4. Helper de validació (desactivat en mode Cloud)
# ============================================================

def check_app_data(strict: bool = False) -> None:
    """
    En la versió per a Streamlit Cloud, els artefactes són remots (URLs)
    i no comprovem la seva existència al sistema de fitxers local.
    Aquesta funció queda com a 'stub'.
    """
    print("check_app_data(): validació desactivada en mode Streamlit Cloud (fitxers remots a OneDrive).")




