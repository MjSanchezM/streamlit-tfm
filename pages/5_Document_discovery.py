# VisualAnalytics/pages/5_Document_discovery.py
# Exploració detallada de documents (Document discovery)

import sys
import io
import os

import pandas as pd
import streamlit as st
import plotly.express as px
from st_aggrid import AgGrid, GridOptionsBuilder, ColumnsAutoSizeMode, JsCode

# =============================================================================
# CONFIGURACIÓ BÀSICA DE LA PÀGINA
# =============================================================================
st.set_page_config(
    page_title="05 · Document discovery",
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
    DOC_TABLE_MINIMAL_FILE,   # app_data / doc_table_minimal.parquet
    DOC_TABLE_ENRICHED_FILE,  # app_data / doc_table_enriched.parquet
    TOOLS_DIR,                # VisualAnalytics / tools
)

# tools/ (plot_style.py, etc.)
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

def get_best_cluster_label(label):
    """
    Donada una etiqueta de clúster separada per ';',
    intenta retornar, per ordre de preferència:
      - 5a paraula (índex 4)
      - 4a (3)
      - 3a (2)
      - 2a (1)
      - 1a (0)
    Si no n'hi ha o totes són buides, retorna string buit.
    """
    if label is None or (isinstance(label, float) and pd.isna(label)):
        return ""

    text = str(label).strip()
    if not text or text.lower() in {"nan", "none"}:
        return ""

    # Separem per ';' i netegem espais / buits
    parts = [p.strip() for p in text.split(";")]
    parts = [p for p in parts if p]  # només no buides

    if not parts:
        return ""

    # Índexs en ordre de preferència: 5a, 4a, 3a, 2a, 1a
    preferred_idxs = [4, 3, 2, 1, 0]
    for idx in preferred_idxs:
        if 0 <= idx < len(parts) and parts[idx]:
            return parts[idx]

    # Si per algun motiu no hem trobat res, retornem buit
    return ""



def search_text(row, query, fields):
    """
    Cerca un text (query) en una llista de camps del DataFrame.
    """
    q = query.lower().strip()
    if not q:
        return True
    for f in fields:
        if f in row and pd.notna(row[f]):
            if q in str(row[f]).lower():
                return True
    return False

# Renderer JS per convertir l'URL en un enllaç clicable a AgGrid
link_renderer = JsCode(
    """
    class UrlCellRenderer {
      init(params) {
        const url = params.value;
        this.eGui = document.createElement('a');
        if (!url || url === 'nan') {
          this.eGui.innerText = '';
          return;
        }
        this.eGui.innerText = '🔗 ' + url;
        this.eGui.setAttribute('href', url);
        this.eGui.setAttribute('target', '_blank');
        this.eGui.setAttribute('style', 'text-decoration:none; color:#000078;');
      }
      getGui() {
        return this.eGui;
      }
    }
    """
)

# =============================================================================
# CÀRREGA DE DADES
# =============================================================================

@st.cache_data
def load_doc_table():
    """
    Carrega:
      - doc_table_minimal.parquet (part6) -> info bàsica + cluster_hdbscan, cluster_label_auto
      - doc_table_enriched.parquet (part5) -> Dept_main, Dept_list, Dept_collab_list, cluster_label_best, kw_list_str, year...
    i les fusiona en un únic DataFrame preparat per al dashboard.
    """

    # ---------------------------
    # 1) Taula mínima (obligatòria)
    # ---------------------------
    if not os.path.exists(DOC_TABLE_MINIMAL_FILE):
        st.error(
            "No s'ha trobat el fitxer doc_table_minimal.parquet:\n"
            f"{DOC_TABLE_MINIMAL_FILE}"
        )
        return pd.DataFrame()

    df_min = None
    last_error = None
    for engine in ["fastparquet", "pyarrow"]:
        try:
            df_min = pd.read_parquet(DOC_TABLE_MINIMAL_FILE, engine=engine)
            break
        except Exception as e:
            last_error = e

    if df_min is None:
        st.error(
            "No s'ha pogut carregar doc_table_minimal.parquet "
            "amb fastparquet ni pyarrow."
        )
        st.info(f"Últim error reportat: {last_error}")
        return pd.DataFrame()

    # ---------------------------
    # 2) Taula enriquida (opcional)
    # ---------------------------
    df_enr = None
    if os.path.exists(DOC_TABLE_ENRICHED_FILE):
        try:
            df_enr = pd.read_parquet(DOC_TABLE_ENRICHED_FILE, engine="fastparquet")
        except Exception:
            try:
                df_enr = pd.read_parquet(DOC_TABLE_ENRICHED_FILE, engine="pyarrow")
            except Exception:
                df_enr = None

    # ---------------------------
    # 3) Fusió de les dues taules
    # ---------------------------
    df = df_min.copy()

    if df_enr is not None:
        # Clau de fusió robusta (títol + any + handle)
        on_cols = [
            c
            for c in ["Title", "AnyPubARPC", "handle_url"]
            if c in df_min.columns and c in df_enr.columns
        ]

        if on_cols:
            df = df_min.merge(
                df_enr,
                on=on_cols,
                how="left",
                suffixes=("", "_enr"),
            )
        # si no hi ha cap columna comuna, simplement ens quedem amb df_min

    cols = df.columns

    # ---------------------------
    # 4) Any de publicació -> 'year'
    # ---------------------------
    if "year" not in cols:
        if "AnyPubARPC" in cols:
            df["year"] = pd.to_numeric(df["AnyPubARPC"], errors="coerce")
        elif "any" in cols:
            df["year"] = pd.to_numeric(df["any"], errors="coerce")
        else:
            df["year"] = pd.NA
    df["year"] = pd.to_numeric(df["year"], errors="coerce")

    # ---------------------------
    # 5) Departament principal normalitzat -> 'Dept_main'
    # ---------------------------
    if "Dept_main" not in cols:
        if "Dept_normalized" in cols:
            df["Dept_main"] = df["Dept_normalized"]
        elif "DeptARPC" in cols:
            df["Dept_main"] = df["DeptARPC"]
        else:
            df["Dept_main"] = pd.NA

    # ---------------------------
    # 6) Etiqueta de clúster "millor" -> 'cluster_label_best'
    # ---------------------------
    if "cluster_label_best" not in df.columns:
        if "cluster_label_auto" in df.columns:
            df["cluster_label_best"] = df["cluster_label_auto"].apply(
                get_best_cluster_label
            )
        else:
            df["cluster_label_best"] = ""

    df["cluster_label_best"] = df["cluster_label_best"].fillna("").astype(str)

    # Reetiquetem soroll sense etiqueta com "Multidisciplinar"
    if "cluster_hdbscan" in df.columns:
        mask_buida = df["cluster_label_best"].str.strip().eq("") & (
            df["cluster_hdbscan"] == -1
        )
    else:
        mask_buida = df["cluster_label_best"].str.strip().eq("")
    df.loc[mask_buida, "cluster_label_best"] = "Multidisciplinar"

    return df




df = load_doc_table()

if df.empty:
    st.title("Exploració de documents")
    st.warning("No s'han pogut carregar dades des de doc_table_minimal.parquet.")
    st.stop()

# =============================================================================
# CAPÇALERA
# =============================================================================

st.title("Exploració detallada de documents")

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
      Aquesta pàgina funciona com un <strong>explorador de documents</strong> del corpus. 
      Permet combinar filtres per:
      <ul>
        <li><strong>any de publicació</strong> (2011–2025),</li>
        <li><strong>departament principal</strong>,</li>
        <li><strong>clúster temàtic</strong> (a partir de l'etiqueta automàtica),</li>
        <li>i una <strong>cerca per paraules</strong> en títol, paraules clau i resum.</li>
      </ul>
      El resultat és una taula interactiva amb els camps principals del document 
      i l'<strong>enllaç directe al repositori institucional</strong>, amb opció de 
      <strong>descarregar en CSV</strong> el conjunt filtrat.
      És una visió de <strong>"document discovery"</strong>: orientada a trobar, refinar 
      i exportar subconjunts de documents d'interès a partir dels resultats del model temàtic.
    </div>
    """,
    unsafe_allow_html=True,
)


st.markdown("---")

# =============================================================================
# FILTRES
# =============================================================================

st.subheader("Filtres de cerca")

df_filt = df.copy()

col_f1, col_f2, col_f3 = st.columns([1.2, 1.2, 1.2])

# -------- Any (2011–2025) --------
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

# -------- Departament principal + col·laboradors --------
with col_f2:
    selected_dept = "(Tots els departaments)"
    selected_collab = None

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

        # Filtre pel principal
        if selected_dept != "(Tots els departaments)":
            df_filt = df_filt[df_filt["Dept_main"] == selected_dept]

            # Si tenim info de col·laboradors, construïm el segon filtre
            if "Dept_collab_list" in df_filt.columns:
                # Explota la llista de col·laboradors per obtenir els únics
                collab_series = (
                    df_filt["Dept_collab_list"]
                    .explode()
                    .dropna()
                    .astype(str)
                    .str.strip()
                )

                collab_vals = sorted(
                    {d for d in collab_series if d and d != selected_dept}
                )

                if collab_vals:
                    collab_options = (
                        ["(Tots els departaments col·laboradors)"] + collab_vals
                    )

                    selected_collab = st.selectbox(
                        "Departaments col·laboradors",
                        options=collab_options,
                        index=0,
                        help=(
                            "Mostra només els articles on el departament "
                            "principal col·labora amb el departament seleccionat."
                        ),
                    )

                    if selected_collab != "(Tots els departaments col·laboradors)":
                        df_filt = df_filt[
                            df_filt["Dept_collab_list"].apply(
                                lambda lst: isinstance(lst, list)
                                and selected_collab in lst
                            )
                        ]


# -------- Clústers temàtics (Cxxx — etiqueta millor possible) --------
with col_f3:
    if "cluster_hdbscan" in df_filt.columns:
        cluster_col = "cluster_hdbscan"

        # Tots els clústers presents DESPRÉS dels altres filtres (any, dept, etc.)
        cluster_ids_for_ui = (
            df_filt[cluster_col]
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )
        cluster_ids_for_ui = sorted(cluster_ids_for_ui)

        # Construïm mapping id -> etiqueta "millor" a partir de cluster_label_auto
        mapping_best = {}
        if "cluster_label_auto" in df_filt.columns:
            tmp = (
                df_filt[[cluster_col, "cluster_label_auto"]]
                .dropna(subset=[cluster_col])
                .drop_duplicates(subset=[cluster_col])
            )
            tmp["best"] = tmp["cluster_label_auto"].apply(get_best_cluster_label)
            mapping_best = dict(
                zip(
                    tmp[cluster_col].astype(int),
                    tmp["best"].astype(str),
                )
            )

        cluster_labels_for_ui = []
        label_to_id = {}

        for cid in cluster_ids_for_ui:
            cid_int = int(cid)

            if cid_int == -1:
                # Soroll: sempre Multidisciplinar
                best = "Multidisciplinar"
            else:
                best = mapping_best.get(cid_int, "").strip()
                if not best:
                    # si no tenim etiqueta, fem servir el número com a fallback
                    best = str(cid_int)

            # Codi Cxxx (o C-1 per al soroll)
            code = "C-1" if cid_int < 0 else f"C{cid_int:03d}"

            label_ui = f"{code} — {best}"
            cluster_labels_for_ui.append(label_ui)
            label_to_id[label_ui] = cid_int

        selected_clusters_ui = st.multiselect(
            "Clústers temàtics (etiqueta)",
            options=cluster_labels_for_ui,
            default=[],
            help=(
                "Prefix Cxxx = ID de clúster; l'etiqueta s'obté prioritzant la 5a "
                "paraula de cluster_label_auto, i si no existeix es recorre cap a la 1a."
            ),
        )

        if selected_clusters_ui:
            selected_ids = [label_to_id[lbl] for lbl in selected_clusters_ui]
            df_filt = df_filt[df_filt[cluster_col].astype(int).isin(selected_ids)]




# -------- Cerca per paraules (a sota, ample) --------
query = st.text_input(
    "Cerca per paraules a títol, paraules clau i resum",
    value="",
    placeholder="ex. obesity, catalysis, qualitative study...",
).strip()

if query:
    search_fields = []
    for cand in ["Title", "ParClauARPC", "kw_list_str", "ResARPC"]:
        if cand in df_filt.columns:
            search_fields.append(cand)

    if search_fields:
        df_filt = df_filt[
            df_filt.apply(lambda row: search_text(row, query, search_fields), axis=1)
        ]

if df_filt.empty:
    st.warning("Cap document compleix els filtres actuals.")
    st.stop()

st.markdown("---")

# =============================================================================
# VISUAL BREU · TOP CLÚSTERS ALS RESULTATS
# =============================================================================

st.subheader("Distribució temàtica dels resultats filtrats")

if "cluster_label_best" in df_filt.columns:
    tmp = df_filt.copy()
    cluster_col = "cluster_hdbscan" if "cluster_hdbscan" in tmp.columns else None

    # Checkbox per incloure/excloure el soroll C-1
    if cluster_col is not None:
        show_multidisciplinari = st.checkbox(
            "Mostrar clúster Multidisciplinari (C-1)",
            value=False,
            help="Inclou el clúster de soroll (-1), etiquetat com «Multidisciplinari»."
        )

        if not show_multidisciplinari:
            tmp = tmp[tmp[cluster_col] != -1]

    if tmp.empty:
        st.info("No hi ha clústers per mostrar amb la configuració actual.")
    else:
        # Construïm una etiqueta de visualització: Cxxx — etiqueta
        def format_cluster_row(row):
            if cluster_col is None or pd.isna(row.get(cluster_col)):
                # Sense id de clúster: mostrem només l'etiqueta
                return row.get("cluster_label_best", "(sense clúster)")

            cid = int(row.get(cluster_col))
            base = str(row.get("cluster_label_best", "")).strip()

            if cid == -1:
                code = "C-1"
                label = "Multidisciplinari"
            else:
                code = f"C{cid:03d}"
                label = base if base else str(cid)

            return f"{code} — {label}"

        tmp["cluster_display"] = tmp.apply(format_cluster_row, axis=1)

        top_clusters = (
            tmp.groupby("cluster_display")
            .size()
            .reset_index(name="n_docs")
            .sort_values("n_docs", ascending=False)
            .head(15)
        )

        fig_bar = px.bar(
            top_clusters,
            x="n_docs",
            y="cluster_display",
            orientation="h",
            labels={
                "n_docs": "Documents",
                "cluster_display": "Clúster (Cxxx — etiqueta)",
            },
        )
        fig_bar.update_layout(
            margin=dict(l=0, r=10, t=100, b=40),
            font=dict(color="#000078"),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_bar, width="stretch")

        # Text resum sota el gràfic
        total_docs_in_chart = int(top_clusters["n_docs"].sum())
        st.markdown(
            f"""
            <p style="margin-top:6px; font-size:14px;">
              <em>Amb els filtres seleccionats, els clústers mostrats en el gràfic
              agrupen <strong>{total_docs_in_chart:,}</strong> documents del repositori.</em>
            </p>
            """,
            unsafe_allow_html=True,
        )

else:
    st.info(
        "No hi ha informació d'etiquetes automàtiques per mostrar la distribució de clústers."
    )

st.markdown("---")




# =============================================================================
# TAULA INTERACTIVA I DESCÀRREGA
# =============================================================================

st.subheader("Resultats de la cerca")

# Preparem les columnes a mostrar (sense doc_id)
cols_to_show = []
for c in ["Title", "year", "Dept_main", "cluster_display", "handle_url"]:
    if c in df_filt.columns:
        cols_to_show.append(c)

df_table = df_filt[cols_to_show].copy()

# Canvis de nom de columnes
rename_map = {
    "Title": "Article",
    "year": "Any",
    "Dept_main": "Departament",
    "cluster_display": "Clúster (Cxxx — etiqueta)",
    "handle_url": "Enllaç al Repositori",
}
df_table = df_table.rename(columns=rename_map)

# ---------- Taula interactiva amb AgGrid (enllaç clicable) ----------
gb = GridOptionsBuilder.from_dataframe(df_table)

# Configuració general de columnes
gb.configure_default_column(
    resizable=True,
    filter=True,
    sortable=True,
    wrapText=True,
    autoHeight=True,
)

# Columna d’article una mica més ampla
if "Article" in df_table.columns:
    gb.configure_column("Article", flex=3)

# Columna de clúster
if "Clúster (Cxxx — etiqueta)" in df_table.columns:
    gb.configure_column("Clúster (Cxxx — etiqueta)", flex=2)

# Columna d'enllaç amb renderer clicable
if "Enllaç al Repositori" in df_table.columns:
    gb.configure_column(
        "Enllaç al Repositori",
        cellRenderer=link_renderer,
        flex=2,
        filter=False,
        sortable=False,
    )

# Una mica de paginació perquè sigui més usable
gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=25)

grid_options = gb.build()

AgGrid(
    df_table,
    gridOptions=grid_options,
    theme="streamlit",
    height=650,  # ajusta l'alçada si vols
    fit_columns_on_grid_load=True,
    columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS,
    allow_unsafe_jscode=True,          # ✅ IMPORTANT per poder usar JsCode
)


# -------------------------------------------------------------------------
# Botó de descàrrega de CSV dels resultats filtrats (tots els camps originals)
# -------------------------------------------------------------------------
csv_buffer = io.StringIO()
df_filt.to_csv(csv_buffer, index=False)
csv_bytes = csv_buffer.getvalue().encode("utf-8-sig")

st.download_button(
    label="Descarregar resultats filtrats en CSV",
    data=csv_bytes,
    file_name="document_discovery_filtrat.csv",
    mime="text/csv",
)

st.markdown("---")

