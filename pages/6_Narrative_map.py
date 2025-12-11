# VisualAnalytics/pages/6_Narrative_map.py
# Narrative Map de clústers (UMAP + etiquetes + metadades)

import sys
import os  # el deixem per si es fa servir més avall

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# =============================================================================
# CONFIGURACIÓ BÀSICA DE LA PÀGINA
# =============================================================================
st.set_page_config(
    page_title="06 · Narrative map de clústers",
    layout="wide",
)

# Ajust tipogràfic lleu
st.markdown(
    """
    <style>
    h1 { font-size: 26px !important; }
    h2 { font-size: 20px !important; }
    h3 { font-size: 18px !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# RUTES DEL PROJECTE I BRANDING (VERSIÓ CENTRALITZADA VIA Paths_Project)
# =============================================================================
from Paths_Project import (
    DOCS_ENRICHED_FILE,   # app_data / df_docs_kw_enriched_with_labels.parquet
    NARRATIVE_FILE,       # app_data / narrative_map_docs.parquet
    TOOLS_DIR,            # VisualAnalytics / tools
)

# tools/ (plot_style.py)
if str(TOOLS_DIR) not in sys.path:
    sys.path.append(str(TOOLS_DIR))


try:
    from plot_style import (
        COLOR_PRIMARY,
        COLOR_PRIMARY_DARK,
        COLOR_COMP1,
        COLOR_NEUTRAL_1,
        COLOR_NEUTRAL_2,
        COLOR_NEUTRAL_3,
    )
except ImportError:
    COLOR_PRIMARY = "#73EDFF"
    COLOR_PRIMARY_DARK = "#000078"
    COLOR_COMP1 = "#D6FAFF"
    COLOR_NEUTRAL_1 = "#f0f4ff"
    COLOR_NEUTRAL_2 = "#cccccc"
    COLOR_NEUTRAL_3 = "#878787"

# Tema lleu de fons
st.markdown(
    f"""
    <style>
      html, body, [data-testid="stAppViewContainer"] * {{
          color: #000078 !important;
          font-size: 14px;
      }}
      section[data-testid="stSidebar"] * {{
          color: #000078 !important;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# HELPERS
# =============================================================================

def get_best_cluster_label(label: str) -> str:
    """
    Donada una etiqueta de clúster separada per ';',
    intenta retornar, per ordre de preferència:
      5a paraula → 4a → 3a → 2a → 1a.
    Si no n'hi ha o totes són buides, retorna string buit.
    """
    if label is None or (isinstance(label, float) and pd.isna(label)):
        return ""

    text = str(label).strip()
    if not text or text.lower() in {"nan", "none"}:
        return ""

    parts = [p.strip() for p in text.split(";")]
    parts = [p for p in parts if p]

    if not parts:
        return ""

    preferred_idxs = [4, 3, 2, 1, 0]
    for idx in preferred_idxs:
        if 0 <= idx < len(parts) and parts[idx]:
            return parts[idx]

    return ""


def build_cluster_display(cid: int, base_label: str) -> str:
    """
    Construeix 'Cxxx — etiqueta' a partir de:
      - cid: id numèric del clúster (HDBSCAN)
      - base_label: etiqueta textual (5a paraula preferentment)
    Cas especial soroll (-1): 'C-1 — Multidisciplinar'
    """
    if cid == -1:
        code = "C-1"
        label = "Multidisciplinar"
    else:
        code = f"C{cid:03d}"
        label = base_label.strip() if base_label and str(base_label).strip() else str(cid)

    return f"{code} — {label}"


# =============================================================================
# CÀRREGA DE DADES
# =============================================================================

@st.cache_data
def load_narrative_data():
    """
    Carrega narrative_map_docs.parquet (part6) i, si cal,
    recupera cluster_hdbscan des del corpus enriquit (part5).
    També garanteix:
      - any numèric 'year'
      - columna cluster_label_best
      - columna auxiliar cluster_display = 'Cxxx — etiqueta'
      - keywords en forma de string per hover (kw_top5_str)
    """
    if not os.path.exists(NARRATIVE_FILE):
        st.error(
            "No s'ha trobat l'artefacte del Narrative Map:\n"
            f"{NARRATIVE_FILE}"
        )
        return pd.DataFrame()

    df = pd.read_parquet(NARRATIVE_FILE, engine="fastparquet")
    cols = df.columns.tolist()

    # ---------------------------
    # 1) Recuperar cluster_hdbscan si falta
    # ---------------------------
    if "cluster_hdbscan" not in cols:
        if os.path.exists(DOCS_ENRICHED_FILE):
            try:
                df_docs = pd.read_parquet(DOCS_ENRICHED_FILE, engine="fastparquet")
            except Exception:
                df_docs = pd.read_parquet(DOCS_ENRICHED_FILE)  # fallback

            if "doc_id" in df_docs.columns and "doc_id" in df.columns:
                df = df.merge(
                    df_docs[["doc_id", "cluster_hdbscan"]],
                    on="doc_id",
                    how="left",
                    suffixes=("", "_from_docs"),
                )
        # actualitzem la llista de columnes
        cols = df.columns.tolist()

    # ---------------------------
    # 2) Any numèric 'year'
    # ---------------------------
    if "year" not in cols:
        if "AnyPubARPC" in cols:
            df["year"] = pd.to_numeric(df["AnyPubARPC"], errors="coerce")
        else:
            df["year"] = pd.NA

    df["year"] = pd.to_numeric(df["year"], errors="coerce")

    # ---------------------------
    # 3) Etiqueta "millor" de clúster
    # ---------------------------
    if "cluster_label_best" not in cols:
        if "cluster_label_auto" in cols:
            df["cluster_label_best"] = df["cluster_label_auto"].apply(
                get_best_cluster_label
            )
        else:
            df["cluster_label_best"] = ""

    df["cluster_label_best"] = df["cluster_label_best"].fillna("").astype(str)

    # Soroll sense etiqueta → Multidisciplinar
    if "cluster_hdbscan" in df.columns:
        mask_noise = (
            (df["cluster_hdbscan"] == -1)
            & df["cluster_label_best"].str.strip().eq("")
        )
        df.loc[mask_noise, "cluster_label_best"] = "Multidisciplinar"

    # ---------------------------
    # 4) Etiqueta de visualització 'Cxxx — etiqueta'
    # ---------------------------
    if "cluster_hdbscan" in df.columns:
        df["cluster_hdbscan"] = pd.to_numeric(df["cluster_hdbscan"], errors="coerce")
        df["cluster_hdbscan"] = df["cluster_hdbscan"].fillna(-1).astype(int)
        df["cluster_display"] = df.apply(
            lambda row: build_cluster_display(
                row["cluster_hdbscan"], row.get("cluster_label_best", "")
            ),
            axis=1,
        )
    else:
        # Si, per algun motiu, no tenim id numèric de clúster,
        # fem servir només l'etiqueta textual
        df["cluster_display"] = df["cluster_label_best"].replace(
            "", "(Sense clúster)"
        )

    # ---------------------------
    # 5) Keywords top5 per hover
    # ---------------------------
    def kw_to_str(kw_list):
        if isinstance(kw_list, list):
            return ", ".join(str(k) for k in kw_list[:5])
        # si ve serialitzat o en string, no toquemen massa
        return str(kw_list) if kw_list is not None else ""

    if "kw_list" in df.columns:
        df["kw_top5_str"] = df["kw_list"].apply(kw_to_str)
    else:
        df["kw_top5_str"] = ""

    # Probabilitat HDBSCAN
    if "prob_hdbscan" not in df.columns:
        df["prob_hdbscan"] = np.nan
    else:
        df["prob_hdbscan"] = pd.to_numeric(df["prob_hdbscan"], errors="coerce")

    return df


df = load_narrative_data()

if df.empty:
    st.title("Narrative map de clústers")
    st.warning("No s'han pogut carregar dades per construir el Narrative Map.")
    st.stop()

# =============================================================================
# CAPÇALERA
# =============================================================================

st.title("Narrative map de clústers temàtics")

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
      Aquest <strong>narrative map</strong> representa el paisatge temàtic del corpus
      a partir de:
      <ul>
        <li>una <strong>projecció UMAP 2D</strong> dels embeddings SBERT,</li>
        <li>els <strong>clústers HDBSCAN</strong> (codi Cxxx),</li>
        <li>les <strong>etiquetes automàtiques</strong> derivades de les paraules clau,</li>
        <li>i metadades com <strong>any de publicació</strong> i <strong>departament</strong>.</li>
      </ul>
      Cada punt és un document; la seva posició reflecteix la <strong>proximitat semàntica</strong>,
      el color indica el <strong>clúster temàtic</strong> i els filtres permeten explorar
      l'evolució temporal i institucional.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

# =============================================================================
# FILTRES
# =============================================================================

st.subheader("Filtres del Narrative Map")

df_filt = df.copy()

col_f1, col_f2, col_f3 = st.columns([1.2, 1.2, 1.6])

# -------- Any (2011–2025) + Probabilitat HDBSCAN --------
with col_f1:
    years = pd.to_numeric(df_filt["year"], errors="coerce").dropna().astype(int)
    if not years.empty:
        MIN_YEAR_DASHBOARD = 2011
        MAX_YEAR_DASHBOARD = 2025

        years_in_range = [
            y for y in years.unique()
            if MIN_YEAR_DASHBOARD <= y <= MAX_YEAR_DASHBOARD
        ]
        if years_in_range:
            default_from = min(years_in_range)
            default_to = max(years_in_range)
        else:
            default_from = MIN_YEAR_DASHBOARD
            default_to = MAX_YEAR_DASHBOARD

        year_range = st.slider(
            "Any de publicació",
            min_value=MIN_YEAR_DASHBOARD,
            max_value=MAX_YEAR_DASHBOARD,
            value=(default_from, default_to),
            step=1,
        )

        df_filt = df_filt[
            df_filt["year"].isna()
            | (
                (df_filt["year"] >= year_range[0])
                & (df_filt["year"] <= year_range[1])
            )
        ]

    # Llindar de probabilitat HDBSCAN
    if "prob_hdbscan" in df_filt.columns:
        prob_min = float(
            st.slider(
                "Probabilitat mínima HDBSCAN",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.05,
                help="Descarta punts amb assignació de clúster molt incerta.",
            )
        )
        df_filt = df_filt[
            df_filt["prob_hdbscan"].isna()
            | (df_filt["prob_hdbscan"] >= prob_min)
        ]

# -------- Departament principal --------
with col_f2:
    if "Dept_main" in df_filt.columns:
        all_depts = (
            df_filt["Dept_main"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        all_depts = sorted(all_depts)
        dept_options = ["(Tots els departaments)"] + all_depts

        selected_dept = st.selectbox(
            "Departament principal",
            options=dept_options,
            index=0,
        )

        if selected_dept != "(Tots els departaments)":
            df_filt = df_filt[df_filt["Dept_main"] == selected_dept]

# -------- Selecció de clústers (després d'any, prob i departament) --------
with col_f3:
    if "cluster_display" in df_filt.columns and "cluster_hdbscan" in df_filt.columns:
        cluster_counts = (
            df_filt.groupby(["cluster_hdbscan", "cluster_display"])
            .size()
            .reset_index(name="n_docs")
            .sort_values("n_docs", ascending=False)
        )
        cluster_labels_all = cluster_counts["cluster_display"].tolist()

        selected_clusters = st.multiselect(
            "Clústers temàtics (Cxxx — etiqueta)",
            options=cluster_labels_all,
            default=cluster_labels_all[:15] if len(cluster_labels_all) > 15 else cluster_labels_all,
            help="Selecciona quins clústers vols veure al mapa.",
        )

        if selected_clusters:
            df_filt = df_filt[df_filt["cluster_display"].isin(selected_clusters)]

# Si després de tots els filtres no queda res, sortim
if df_filt.empty:
    st.warning("Cap document compleix els filtres actuals.")
    st.stop()

st.markdown("---")

# =============================================================================
# KPIs RÀPIDS
# =============================================================================

col_k1, col_k2, col_k3 = st.columns(3)

with col_k1:
    st.metric("Documents visibles al mapa", f"{len(df_filt):,}")

with col_k2:
    n_clusters_visibles = df_filt["cluster_display"].nunique()
    st.metric("Clústers visibles", n_clusters_visibles)

with col_k3:
    anys_valids = pd.to_numeric(df_filt["year"], errors="coerce").dropna().astype(int)
    if not anys_valids.empty:
        st.metric(
            "Període cobert",
            f"{anys_valids.min()}–{anys_valids.max()}",
        )
    else:
        st.metric("Període cobert", "Sense any definit")

st.markdown("---")

# =============================================================================
# SCATTER UMAP (Narrative Map base)
# =============================================================================

st.subheader("Mapa UMAP dels clústers temàtics")

# Assegurem que hi ha coords
if not {"umap_x", "umap_y"}.issubset(df_filt.columns):
    st.error("L'artefacte del Narrative Map no conté coordenades UMAP (umap_x, umap_y).")
else:
    # Preparem dades per hover
    hover_data = {
        "Title": True,
        "year": True,
        "Dept_main": True,
        "cluster_display": True,
        "kw_top5_str": True,
        "handle_url": True,
        "prob_hdbscan": True,
    }
    for k in list(hover_data.keys()):
        if k not in df_filt.columns:
            hover_data.pop(k, None)

    fig_umap = px.scatter(
        df_filt,
        x="umap_x",
        y="umap_y",
        color="cluster_display",
        hover_name="Title" if "Title" in df_filt.columns else None,
        hover_data=hover_data if hover_data else None,
        labels={
            "umap_x": "UMAP 1",
            "umap_y": "UMAP 2",
            "cluster_display": "Clúster (Cxxx — etiqueta)",
        },
    )

    fig_umap.update_layout(
        margin=dict(l=0, r=0, t=40, b=0),
        font=dict(color="#000078"),
        legend_title_text="Clúster",
    )

    # Eixos en estil net
    fig_umap.update_xaxes(
        showline=False,
        zeroline=False,
        showgrid=False,
        visible=False,
    )
    fig_umap.update_yaxes(
        showline=False,
        zeroline=False,
        showgrid=False,
        visible=False,
    )

    st.plotly_chart(fig_umap, width="stretch")

st.markdown("---")

# =============================================================================
# TAULA RESUM DE CLÚSTERS VISIBLES
# =============================================================================

st.subheader("Resum de clústers visibles")

cluster_summary = (
    df_filt.groupby(["cluster_hdbscan", "cluster_display"])
    .agg(
        n_docs=("doc_id", "count"),
        any_min=("year", "min"),
        any_max=("year", "max"),
    )
    .reset_index()
    .sort_values("n_docs", ascending=False)
)

cluster_summary["any_min"] = cluster_summary["any_min"].fillna("").astype(str)
cluster_summary["any_max"] = cluster_summary["any_max"].fillna("").astype(str)

cluster_summary = cluster_summary.rename(
    columns={
        "cluster_hdbscan": "ID clúster",
        "cluster_display": "Clúster (Cxxx — etiqueta)",
        "n_docs": "Documents",
        "any_min": "Any mínim",
        "any_max": "Any màxim",
    }
)

st.dataframe(cluster_summary, width="stretch")

st.markdown(
    """
    <p style="margin-top:6px; font-size:14px;">
      <em>Cada fila correspon a un clúster temàtic visible al mapa, amb el nombre de documents
      i el període temporal que cobreixen dins dels filtres actuals.</em>
    </p>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")
