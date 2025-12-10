# Home.py
import json
import streamlit as st
import pandas as pd
import plotly.express as px
import os

# -------------------------------------------------------------------
# 0. IMPORT DE RUTES DEL PROJECTE (TOTS ELS FITXERS → app_data)
# -------------------------------------------------------------------
from Paths_Project import (
    DOCS_ENRICHED_FILE,      # df_docs_kw_enriched_with_labels.parquet
    DASHBOARD_KPIS_FILE,     # dashboard_overview_kpis.parquet
    CLUSTER_YEAR_COUNTS_FILE,# cluster_year_counts.parquet
    OVERVIEW_STATS_FILE,     # overview_stats.parquet
    SUMMARY_SBERT_FILE,      # summary_03i_sbert_clustering.json
)


# -------------------------------------------------------------------
# Artefactes principals utilitzats a l'Overview
# -------------------------------------------------------------------
MAIN_PARQUET          = DOCS_ENRICHED_FILE
DASHBOARD_KPIS_PATH   = DASHBOARD_KPIS_FILE
CLUSTER_YEAR_COUNTS_PATH = CLUSTER_YEAR_COUNTS_FILE
OVERVIEW_STATS_PATH   = OVERVIEW_STATS_FILE
SUMMARY_JSON_PATH     = SUMMARY_SBERT_FILE


# -------------------------------------------------------------------
# Branding
# -------------------------------------------------------------------
try:
    from plot_style import (
        COLOR_PRIMARY,
        COLOR_PRIMARY_DARK,
        COLOR_COMP1,
        COLOR_NEUTRAL_1,
        COLOR_NEUTRAL_2,
        COLOR_NEUTRAL_3,
    )
except Exception:
    COLOR_PRIMARY = "#73EDFF"
    COLOR_PRIMARY_DARK = "#000078"
    COLOR_COMP1 = "#D6FAFF"
    COLOR_NEUTRAL_1 = "#f0f4ff"
    COLOR_NEUTRAL_2 = "#cccccc"
    COLOR_NEUTRAL_3 = "#878787"

px.defaults.template = "plotly_white"
BRAND_SCALE = [COLOR_COMP1, COLOR_PRIMARY, COLOR_PRIMARY_DARK]


# ===============================================================
# 1. CONFIG STREAMLIT
# ===============================================================


st.set_page_config(
    page_title="TFM: Thematic Mapping Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        html, body, [data-testid="stAppViewContainer"] * {
            color: #000078 !important;
            font-size: 14px;
        }
        h1 { font-size: 26px !important; color:#000078 !important; }
        h2 { font-size: 20px !important; color:#000078 !important; }
        h3 { font-size: 18px !important; color:#000078 !important; }
        .stMetric label { font-size: 12px !important; }
        .stMetric div { font-size: 20px !important; }
    </style>
    """,
    unsafe_allow_html=True,
)



# =============================================================================
# 2. FUNCIONS DE CÀRREGA
# =============================================================================

@st.cache_data
def load_main_df():
    """Carrega el parquet principal des d'una URL (OneDrive descarregable)."""
    try:
        return pd.read_parquet(MAIN_PARQUET, engine="pyarrow")
    except Exception as e:
        st.error("No s'ha pogut carregar el parquet principal des de la URL:")
        st.error(MAIN_PARQUET)
        st.error(str(e))
        return pd.DataFrame()

# -------------------------------------------------------------------------
# Helper recursiu per trobar silhouette al JSON
# -------------------------------------------------------------------------

def _find_first_numeric_by_key_fragment(obj, fragment: str):
    fragment = fragment.lower()

    if isinstance(obj, dict):
        for k, v in obj.items():
            if fragment in str(k).lower() and isinstance(v, (int, float)):
                return float(v)
            found = _find_first_numeric_by_key_fragment(v, fragment)
            if found is not None:
                return found

    elif isinstance(obj, list):
        for item in obj:
            found = _find_first_numeric_by_key_fragment(item, fragment)
            if found is not None:
                return found

    return None


# -------------------------------------------------------------------------
# Silhouette via JSON
# -------------------------------------------------------------------------

@st.cache_data
def load_silhouette_from_json():
    if not os.path.exists(SUMMARY_JSON_PATH):
        return None

    try:
        with open(SUMMARY_JSON_PATH, "r", encoding="utf-8") as f:
            summary = json.load(f)

        # Claus típiques
        for key in ["silhouette", "silhouette_score", "silhouette_avg"]:
            if key in summary and isinstance(summary[key], (int, float)):
                return float(summary[key])

        # Cerca profunda
        value = _find_first_numeric_by_key_fragment(summary, "silhouette")
        if value is not None:
            return value

    except Exception:
        return None

    return None


# -------------------------------------------------------------------------
# overview_stats.parquet → KPIs globals
# -------------------------------------------------------------------------

@st.cache_data
def load_overview_stats():
    if not os.path.exists(OVERVIEW_STATS_PATH):
        return None

    df_stats = None
    last_err = None

    for engine in ["fastparquet", "pyarrow"]:
        try:
            df_stats = pd.read_parquet(OVERVIEW_STATS_PATH, engine=engine)
            break
        except Exception as e:
            last_err = e

    if df_stats is None or df_stats.empty:
        st.info("No s'ha pogut carregar overview_stats.parquet; es calcularan els KPIs des de df.")
        return None

    row = df_stats.iloc[0]

    return {
        "n_docs": int(row.get("n_docs", 0)),
        "n_clusters": int(row.get("n_clusters", 0)),
        "min_year": int(row.get("min_year")) if pd.notna(row.get("min_year")) else None,
        "max_year": int(row.get("max_year")) if pd.notna(row.get("max_year")) else None,
        "silhouette": float(row.get("silhouette_sbert_hdbscan"))
        if pd.notna(row.get("silhouette_sbert_hdbscan"))
        else None,
        "generated_at": row.get("generated_at"),
    }


# -------------------------------------------------------------------------
# cluster_year_counts
# -------------------------------------------------------------------------

@st.cache_data
def load_cluster_year_counts():
    if not os.path.exists(CLUSTER_YEAR_COUNTS_PATH):
        return None

    for engine in ["pyarrow", "fastparquet"]:
        try:
            return pd.read_parquet(CLUSTER_YEAR_COUNTS_PATH, engine=engine)
        except:
            pass

    return None


# -------------------------------------------------------------------------
# KPIs addicionals (no obligatoris)
# -------------------------------------------------------------------------

@st.cache_data
def load_dashboard_kpis():
    if not os.path.exists(DASHBOARD_KPIS_PATH):
        return None

    for engine in ["pyarrow", "fastparquet"]:
        try:
            return pd.read_parquet(DASHBOARD_KPIS_PATH, engine=engine)
        except:
            pass

    return None


# =============================================================================
# 3. CÀRREGA
# =============================================================================

df = load_main_df()
stats = load_overview_stats()
cluster_year_counts = load_cluster_year_counts()
dashboard_kpis = load_dashboard_kpis()
silhouette_json = load_silhouette_from_json()


# =============================================================================
# 4. PÀGINA PRINCIPAL
# =============================================================================

if df.empty:
    st.error("No s'ha pogut carregar el corpus principal.")
    st.stop()

st.title("Visió general del projecte de mapatge temàtic")
st.subheader("Resum del corpus, metodologia i clústers temàtics")
st.markdown("---")
st.markdown(
    f"""
    <div style="
        border-left:6px solid {COLOR_PRIMARY_DARK};
        padding:10px 14px;
        margin-top:10px;
        margin-bottom:10px;
        background-color:{COLOR_NEUTRAL_1};
        border-radius:6px;
        font-size:14px;">
      Aquesta pàgina actua com a <strong>porta d’entrada</strong> al 
      <strong>dashboard de mapatge temàtic</strong> del repositori institucional. 
      A partir del corpus complet analitzat amb <strong>SBERT + UMAP + HDBSCAN</strong> 
      i de diversos artefactes de síntesi, ofereix una visió global de:
      <ul>
        <li>la <strong>mida i abast del corpus</strong> (nombre de documents, clústers i període temporal),</li>
        <li>la <strong>qualitat del model de clustering</strong> mitjançant la mètrica de silhouette,</li>
        <li>la <strong>distribució dels temes principals</strong> a través d’un Treemap basat en les etiquetes automàtiques,</li>
        <li>i l’evolució anual de la <strong>producció científica en accés obert</strong> durant el període 2011–2025.</li>
      </ul>
      Aquesta visió general permet contextualitzar la resta de pestanyes del dashboard 
      (paisatge semàntic UMAP, anàlisi temàtica, evolució institucional i descobriment de documents) 
      i ajuda a entendre <strong>què s’ha analitzat</strong>, <strong>com</strong> i 
      <strong>en quin marc temporal i temàtic</strong>.
    </div>
    """,
    unsafe_allow_html=True,
)



# -------------------------------------------------------------------------
# 1. METODOLOGIA
# -------------------------------------------------------------------------

st.header("1. Descripció general i metodologia")
st.markdown(
    """
    Aquest *dashboard* interactiu permet explorar els resultats de l'anàlisi del text complet
    dels documents dipositats en accés obert al **Repositori Institucional de la URV**, 
    mitjançant un pipeline basat en:

    • **SBERT** per obtenir embeddings semàntics  
    • **UMAP** per reducció dimensional  
    • **HDBSCAN** per a agrupament no supervisat  
    • Etiquetatge automàtic i enriquiment semàntic dels clústers  
    """
)

# -------------------------------------------------------------------------
# 2. KPIs
# -------------------------------------------------------------------------

st.header("2. Indicadors clau del projecte")

# --- Vista de KPIs segons disponibilitat del parquet de stats

if stats is not None:
    n_docs = stats["n_docs"]
    n_clusters = stats["n_clusters"]
    min_year = stats["min_year"]
    max_year = stats["max_year"]

    silhouette_val = stats["silhouette"] if stats["silhouette"] is not None else silhouette_json

else:
    # Fallback total
    n_docs = len(df)
    n_clusters = df[df["cluster_hdbscan"] != -1]["cluster_hdbscan"].nunique()

    years = pd.to_numeric(df["AnyPubARPC"], errors="coerce")
    years_valid = years[(years >= 1900) & (years <= 2025)]
    min_year = int(years_valid.min())
    max_year = int(years_valid.max())

    silhouette_val = silhouette_json


# --- Render KPIs ---

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Documents analitzats", f"{n_docs:,}", delta="Corpus complet")

with col2:
    st.metric("Nombre de clústers", n_clusters)

with col3:
    st.metric("Període temporal", f"{min_year} — {max_year}")

with col4:
    if silhouette_val is not None:
        st.metric("Silhouette (SBERT + HDBSCAN)", f"{silhouette_val:.3f}", delta="Model final 03i")
    else:
        st.metric("Silhouette (SBERT + HDBSCAN)", "N/A", delta="Sense dades")

st.markdown("---")

# -------------------------------------------------------------------------
# 3. Distribució i mida dels temes principals (Treemap)
# -------------------------------------------------------------------------

st.header("3. Distribució i mida dels temes principals (Treemap)")
st.write(
    "El Treemap mostra la distribució de la producció científica "
    "pels clústers temàtics on s'ha exlcòs el soroll."
    "\nL'etiqueta que és mostra és la 5 del conjunt de keywords"
)

# Excloem soroll (-1)
if "cluster_hdbscan" in df.columns:
    df_treemap_src = df[df["cluster_hdbscan"] != -1].copy()
else:
    df_treemap_src = df.copy()

if df_treemap_src["cluster_label_auto"].nunique() > 1:
    df_treemap = (
        df_treemap_src
        .groupby(["cluster_hdbscan", "cluster_label_auto"])
        .size()
        .reset_index(name="n_docs")
        .sort_values("n_docs", ascending=False)
    )

    # ----------- ★ AGAFA LA CINQUENA KEYWORD ★ -----------
    def get_fifth_keyword(label):
        parts = str(label).split(";")
        return parts[4].strip() if len(parts) >= 5 else ""

    df_treemap["short_label"] = df_treemap["cluster_label_auto"].apply(get_fifth_keyword)

    TREEMAP_COLOR_SCALE = ["#D8D8EA", "#B3B3D5", "#9999C9"]

    fig_treemap = px.treemap(
        df_treemap,
        path=["cluster_label_auto"],      
        values="n_docs",
        color="n_docs",
        color_continuous_scale=TREEMAP_COLOR_SCALE,
        custom_data=[
            "cluster_hdbscan",        # 0
            "n_docs",                 # 1
            "cluster_label_auto",     # 2 (label complet)
            "short_label",            # 3 (CINQUENA KEYWORD)
        ],
    )

    # Text dins de cada casella + hover
    fig_treemap.update_traces(
        textinfo="text",
        texttemplate="<b>C%{customdata[0]}</b><br>%{customdata[3]}<br>%{value}",
        insidetextfont=dict(size=12, color="#000078"),
        hovertemplate=(
            "<b>Clúster C%{customdata[0]}</b><br><br>"
            "%{customdata[2]}<br>"
            "Documents: %{customdata[1]}<extra></extra>"
        ),
        selector=dict(type="treemap"),
    )

    fig_treemap.update_layout(
        font=dict(color="#000078"),
        margin=dict(l=0, r=0, t=40, b=0),
        uniformtext=dict(minsize=9, mode="hide"),
        coloraxis_colorbar=dict(
            title=dict(
                text="Documents",
                font=dict(color="#000078"),
            ),
            tickfont=dict(color="#000078"),
        ),
    )


    st.plotly_chart(fig_treemap, use_container_width=True)

else:
    st.info("No hi ha prou diversitat de clústers per generar el Treemap.")

st.markdown("---")

# -------------------------------------------------------------------------
# 4. EVOLUCIÓ ANUAL DE LA PRODUCCIÓ CIENTÍFICA EN ACCÉS OBERT ANALITZADA 
# -------------------------------------------------------------------------

st.header("4. Evolució anual de la producció científica")

st.markdown(
    """
Aquesta gràfica mostra els articles **a partir de l'any 2011**, any en què s'aprova la 
[Ley 14/2011, de 1 de junio, de la Ciencia, la Tecnología y la Innovación](https://www.boe.es/buscar/act.php?id=BOE-A-2011-9617),
la primera norma de l'Estat espanyol que estableix que la recerca finançada amb fons públics
s'ha de publicar en **accés obert**.

Les dades de **2025** només cobreixen els **primers sis mesos**, ja que l'extracció del corpus
es va realitzar al juliol de 2025.
"""
)

# =====================================================================================
# Construcció de df_year
# =====================================================================================

if cluster_year_counts is not None and not cluster_year_counts.empty:
    # Cas ideal: ja tenim l’artefacte d’evolució per clúster
    if {"year", "n_docs"}.issubset(cluster_year_counts.columns):
        df_year = (
            cluster_year_counts
            .groupby("year")["n_docs"]
            .sum()
            .reset_index()
            .sort_values("year")
        )
    else:
        df_year = pd.DataFrame(columns=["year", "n_docs"])
else:
    # Fallback: calculem a partir del df principal
    year_col = None
    for col in ["year", "AnyPubARPC_int", "AnyPubARPC", "any"]:
        if col in df.columns:
            year_col = col
            break

    if year_col is not None:
        years = pd.to_numeric(df[year_col], errors="coerce")
        df_tmp = df.copy()
        df_tmp["year"] = years
        df_year = (
            df_tmp[df_tmp["year"].notna()]
            .groupby("year")
            .size()
            .reset_index(name="n_docs")
            .sort_values("year")
        )
    else:
        df_year = pd.DataFrame(columns=["year", "n_docs"])

# =====================================================================================
# Filtre 2011–2025 i traç de la gràfica
# =====================================================================================

if not df_year.empty:
    # Normalitzem tipus i filtrem anys
    df_year["year"] = pd.to_numeric(df_year["year"], errors="coerce").astype("Int64")
    df_year = df_year.dropna(subset=["year"])

    df_year = df_year[(df_year["year"] >= 2011) & (df_year["year"] <= 2025)]

    if not df_year.empty:

        # Ens assegurem que surtin tots els anys de 2011 a 2025 (encara que tinguin 0 docs)
        all_years = pd.DataFrame({"year": list(range(2011, 2025 + 1))})
        df_year = all_years.merge(df_year, on="year", how="left")
        df_year["n_docs"] = df_year["n_docs"].fillna(0)

        # ------------------------------
        # Construcció de la gràfica
        # ------------------------------
        fig_year = px.line(
            df_year,
            x="year",
            y="n_docs",
            markers=True,
        )

        # Personalització del hover amb etiquetes en català
        fig_year.update_traces(
            hovertemplate=(
                "<b>Any analitzat:</b> %{customdata[0]}<br>"
                "<b>Nombre d'articles analitzats:</b> %{customdata[1]}<extra></extra>"
            ),
            customdata=df_year[["year", "n_docs"]].values,
            line=dict(color=COLOR_PRIMARY_DARK),
        )

        # Aspecte visual: eix Y des de 0 i línia base visible
        fig_year.update_layout(
            xaxis_title="Any",
            yaxis_title="Documents",
            font=dict(color="#000078"),
            margin=dict(l=0, r=0, t=40, b=0),
            xaxis=dict(
                tickmode="array",
                tickvals=list(range(2011, 2026)),
            ),
            yaxis=dict(
                rangemode="tozero",                       # comença a 0
                range=[0, df_year["n_docs"].max() + 20],  # una mica de marge superior
                showline=True,
                linecolor="#000078",
                zeroline=True,
                zerolinecolor="#000078",
            ),
        )

        st.plotly_chart(fig_year, use_container_width=True)
    else:
        st.info("No hi ha dades dins del període 2011–2025 per mostrar la producció anual.")
else:
    st.info("No hi ha dades suficients per calcular l'evolució anual.")

st.markdown("---")
