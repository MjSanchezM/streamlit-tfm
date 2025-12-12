# pages/4_Institutional_evolution.py
# Anàlisi de l'evolució institucional en el temps

import os
import sys

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# -------------------------------------------------------------------
# CONFIGURACIÓ BÀSICA DE LA PÀGINA
# -------------------------------------------------------------------
st.set_page_config(
    page_title="04 · Evolució del Repositori Institucional (Període 2011 - 2025)",
    layout="wide",
)

# =============================================================================
# 0. RUTES BASE I BRANDING (VERSIÓ RELATIVA VIA Paths_Project)
# =============================================================================
from Paths_Project import (
    DOCS_ENRICHED_FILE,      # df_docs_kw_enriched_with_labels.parquet (app_data)
    CLUSTER_YEAR_FILE,  # cluster_year_counts.parquet (app_data)
    TOOLS_DIR,               # VisualAnalytics / tools
)

# Fitxer principal de documents
MAIN_PARQUET = DOCS_ENRICHED_FILE

# Fitxer principal per a aquesta pàgina (ja ve directament de app_data)
# (ja importat com CLUSTER_YEAR_COUNTS_FILE)

# tools/ (plot_style.py, etc.)
if str(TOOLS_DIR) not in sys.path:
    sys.path.append(str(TOOLS_DIR))

# Branding corporatiu
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

px.defaults.template = "plotly_white"
BRAND_CONTINUOUS_SCALE = [COLOR_COMP1, COLOR_PRIMARY, COLOR_PRIMARY_DARK]

# Estils globals coherents
st.markdown(
    """
    <style>
        html, body, [data-testid="stAppViewContainer"] * {
            color: #000078 !important;
            font-size: 14px;
        }
        h1, h2, h3, h4, h5, h6 { color: #000078 !important; }
        h1 { font-size: 24px !important; }
        h2 { font-size: 20px !important; }
        h3 { font-size: 18px !important; }
        .stMetric label { font-size: 12px !important; color: #000078 !important; }
        .stMetric div { font-size: 20px !important; color: #000078 !important; }
        section[data-testid="stSidebar"] * { color: #000078 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------------
# Mapa de colors per Dept_main
# -------------------------------------------------------------------
DEPT_COLOR_MAP = {
    "Antropologia, Filosofia i Treball Social": "#ACDA90",
    "Bioquímica i Biotecnologia": "#00A7E1",
    "Ciències Mèdiques Bàsiques": "#FF8F12",
    "Dret Privat, Processal i Financer": "#C2002F",
    "Dret Públic": "#C2002F",
    "Economia": "#BCBBBA",
    "Enginyeria Electrònica, Elèctrica i Automàtica": "#BC6123",
    "Enginyeria Informàtica i Matemàtiques": "#BC6123",
    "Enginyeria Mecànica": "#BC6123",
    "Enginyeria Química": "#BC6123",
    "Estudis Anglesos i Alemanys": "#00A7E1",
    "Estudis de Comunicació": "#D3441C",
    "Filologia Catalana": "#00A7E1",
    "Filologies Romàniques": "#00A7E1",
    "Geografia": "#BCBBBA",
    "Gestió d'Empreses": "#BCBBBA",
    "Història i Història de l'Art": "#ACDA90",
    "Infermeria": "#8A8A8D",
    "Medicina i Cirurgia": "#FF8F12",
    "Pedagogia": "#ACDA90",
    "Psicologia": "#BA80D0",
    "Química Analítica i Química Orgànica": "#00A7E1",
    "Química Física i Inorgànica": "#00A7E1",
    "Escola Tècnica Superior d'Arquitectura": "#BC6123",
    "Altres organs de gestió": "#8B1C40",
}

# Paleta contínua institucional en grisos per al heatmap
CLUSTER_TIME_SCALE = ["#F0F0F0", "#878787"]

# =============================================================================
# HELPERS
# =============================================================================
def get_5th_label(label):
    """
    Retorna el 5è terme d'una etiqueta de clúster separada per ';'.
    Si no hi ha prou termes, retorna l'etiqueta sencera.
    """
    if label is None or (isinstance(label, float) and pd.isna(label)):
        return ""
    parts = str(label).split(";")
    return parts[4].strip() if len(parts) >= 5 else str(label).strip()


# =============================================================================
# 1. FUNCIONS DE CÀRREGA D'ARTEFACTES
# =============================================================================

@st.cache_data
def load_cluster_year_counts_or_build():
    """
    1) Intenta carregar cluster_year_counts.parquet si existeix.
    2) Si no existeix o és buit, el reconstrueix a partir del parquet principal.
    """
    # 1) Intentem carregar l'artefacte, si hi és
    if os.path.exists(CLUSTER_YEAR_FILE):
        for engine in ["fastparquet", "pyarrow"]:
            try:
                df = pd.read_parquet(CLUSTER_YEAR_FILE, engine=engine)
                if not df.empty:
                    return df
            except Exception:
                pass  # provarem a reconstruir-lo

    # 2) Reconstruïm a partir del parquet principal
    df_main_local = load_main_df()
    if df_main_local.empty:
        return pd.DataFrame()

    df = df_main_local.copy()

    # Any numèric
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df[df["year"].notna()]
    df["year"] = df["year"].astype(int)

    # Ens assegurem que Dept_main existeix
    if "Dept_main" not in df.columns:
        if "Dept_list" in df.columns:
            def get_main(lst):
                if isinstance(lst, list) and len(lst) > 0:
                    return lst[0]
                return None
            df["Dept_main"] = df["Dept_list"].apply(get_main)
        elif "Dept_normalized" in df.columns:
            df["Dept_main"] = df["Dept_normalized"]
        else:
            df["Dept_main"] = pd.NA

    # Triem columna de clúster (si hi és)
    cluster_col = None
    for cand in ["cluster_hdbscan", "cluster_label", "cluster"]:
        if cand in df.columns:
            cluster_col = cand
            break

    # Definim columnes de grup
    group_cols = ["year"]
    if "Dept_main" in df.columns:
        group_cols.append("Dept_main")
    if cluster_col is not None:
        group_cols.append(cluster_col)

    df_cyc = (
        df[group_cols]
        .groupby(group_cols)
        .size()
        .reset_index(name="n_docs")
    )

    return df_cyc



@st.cache_data
def load_cluster_year_counts_or_build():
    """
    1) Intenta carregar cluster_year_counts.parquet si existeix.
    2) Si no existeix o és buit, el reconstrueix a partir del parquet principal.
    """
    # 1) Intentem carregar l'artefacte, si hi és
    if os.path.exists(CLUSTER_YEAR_COUNTS_FILE):
        for engine in ["fastparquet", "pyarrow"]:
            try:
                df = pd.read_parquet(CLUSTER_YEAR_COUNTS_FILE, engine=engine)
                if not df.empty:
                    return df
            except Exception:
                pass  # provarem a reconstruir-lo

    # 2) Reconstruïm a partir del parquet principal
    df_main_local = load_main_df()
    if df_main_local.empty:
        return pd.DataFrame()

    df = df_main_local.copy()

    # Any numèric
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df[df["year"].notna()]
    df["year"] = df["year"].astype(int)

    # Ens assegurem que Dept_main existeix
    if "Dept_main" not in df.columns:
        if "Dept_list" in df.columns:
            def get_main(lst):
                if isinstance(lst, list) and len(lst) > 0:
                    return lst[0]
                return None
            df["Dept_main"] = df["Dept_list"].apply(get_main)
        elif "Dept_normalized" in df.columns:
            df["Dept_main"] = df["Dept_normalized"]
        else:
            df["Dept_main"] = pd.NA

    # Triem columna de clúster (si hi és)
    cluster_col = None
    for cand in ["cluster_hdbscan", "cluster_label", "cluster"]:
        if cand in df.columns:
            cluster_col = cand
            break

    # Definim columnes de grup
    group_cols = ["year"]
    if "Dept_main" in df.columns:
        group_cols.append("Dept_main")
    if cluster_col is not None:
        group_cols.append(cluster_col)

    df_cyc = (
        df[group_cols]
        .groupby(group_cols)
        .size()
        .reset_index(name="n_docs")
    )

    return df_cyc


# Carreguem parquet principal i artefacte (o reconstruït)
df_main = load_main_df()
df_cyc = load_cluster_year_counts_or_build()

if df_cyc.empty:
    st.title("Evolució institucional de la producció científica")
    st.warning(
        "No s'han pogut obtenir dades per a l'evolució institucional.\n"
        "Ni `cluster_year_counts.parquet` ni el parquet principal han permès construir la taula."
    )
    st.stop()

# =============================================================================
# 2. DETECCIÓ DE COLUMNES CLAU
# =============================================================================

# Any
if "year" in df_cyc.columns:
    year_col = "year"
else:
    st.error("No s'ha trobat cap columna 'year' a cluster_year_counts.parquet.")
    st.stop()

# Nombre de documents
if "n_docs" in df_cyc.columns:
    count_col = "n_docs"
elif "count" in df_cyc.columns:
    count_col = "count"
else:
    num_cols = df_cyc.select_dtypes(include=["number"]).columns.tolist()
    if len(num_cols) == 1:
        count_col = num_cols[0]
    else:
        st.error(
            "No s'ha trobat cap columna de recompte (`n_docs` o `count`) "
            "a cluster_year_counts.parquet."
        )
        st.stop()

# Clúster (si hi és)
cluster_col = None
for cand in ["cluster_label", "cluster_hdbscan", "cluster"]:
    if cand in df_cyc.columns:
        cluster_col = cand
        break

# Departament (si hi és a l'artefacte agregat)
dept_col = None
for cand in ["dept_norm", "Dept_normalized", "Dept_main", "department", "dept"]:
    if cand in df_cyc.columns:
        dept_col = cand
        break

# Normalitzem la columna d'any a numèrica (per si ve com a string)
df_cyc[year_col] = pd.to_numeric(df_cyc[year_col], errors="coerce")

# Anys vàlids (> 0)
years_all = (
    df_cyc[year_col]
    .dropna()
    .astype(int)
    .loc[lambda s: s > 0]
    .unique()
)

years_all = sorted(years_all.tolist())

if not years_all:
    st.error(
        "No s'han trobat anys vàlids (> 0) a cluster_year_counts.parquet "
        "(després de convertir la columna a numèrica)."
    )
    st.stop()

# Límits temporals oficials del dashboard
MIN_YEAR_DASHBOARD = 2011
MAX_YEAR_DASHBOARD = 2025

# Intersecció entre el que hi ha a les dades i el rang oficial del dashboard
years_in_range = [y for y in years_all if MIN_YEAR_DASHBOARD <= y <= MAX_YEAR_DASHBOARD]

if years_in_range:
    min_year_global = min(years_in_range)
    max_year_global = max(years_in_range)
else:
    min_year_global = MIN_YEAR_DASHBOARD
    max_year_global = MAX_YEAR_DASHBOARD

# -------------------------------------------------------------------
# Llista de clústers i etiquetes "id · 5è terme"
# -------------------------------------------------------------------
# -------------------------------------------------------------------
# Llista de clústers i etiquetes "Cxxx — 5è terme"
# -------------------------------------------------------------------
cluster_ids_for_ui = []
cluster_labels_for_ui = []
label_to_id = {}
mapping_5th = {}

if cluster_col is not None:
    # Tots els clústers presents a l'artefacte agregat
    cluster_ids_for_ui = sorted(df_cyc[cluster_col].dropna().unique().tolist())

    # Construïm etiquetes a partir de df_main.cluster_label_auto si es pot
    if (
        not df_main.empty
        and "cluster_label_auto" in df_main.columns
        and cluster_col in df_main.columns
    ):
        tmp = (
            df_main[[cluster_col, "cluster_label_auto"]]
            .dropna()
            .drop_duplicates(subset=[cluster_col])
        )
        tmp["fifth"] = tmp["cluster_label_auto"].apply(get_5th_label)
        mapping_5th = dict(zip(tmp[cluster_col], tmp["fifth"]))

    for cid in cluster_ids_for_ui:
        # intentem tenir un enter
        try:
            cid_int = int(cid)
        except (TypeError, ValueError):
            cid_int = None

        # 5a paraula clau (o etiqueta sencera si n'hi ha menys)
        base = mapping_5th.get(cid, "")
        short = base if base else str(cid)

        if cid_int is not None:
            label_ui = f"C{cid_int:03d} — {short}"
        else:
            label_ui = short

        cluster_labels_for_ui.append(label_ui)
        label_to_id[label_ui] = cid



# =============================================================================
# 3. LAYOUT I FILTRES
# =============================================================================

st.title("Evolució institucional de la producció científica")
st.subheader("Anàlisi temporal per anys, departaments i clústers temàtics")
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
      Aquesta pàgina analitza l'<strong>evolució temporal</strong> de la producció científica 
      del repositori institucional durant el període <strong>2011–2025</strong>, a partir dels 
      resultats del model <strong>SBERT + UMAP + HDBSCAN</strong> i dels artefactes agregats 
      per any, clúster i departament.
      <br><br>
      La visualització es desplega en tres nivells:
      <ul>
        <li><strong>Evolució anual global</strong> del nombre de documents analitzats.</li>
        <li><strong>Evolució per departament</strong>, per identificar quines unitats 
            generen més producció en cada moment.</li>
        <li><strong>Evolució per clúster temàtic</strong>, amb un mapa de calor i
            una classificació dels clústers segons la seva 
            <strong>tendència temporal</strong> (emergent, estable o decreixent).</li>
      </ul>
      Els filtres d'any, departament i clúster permeten explorar <strong>patrons institucionals</strong> 
      i detectar línies de recerca que creixen, es mantenen o perden pes al llarg del temps.
    </div>
    """,
    unsafe_allow_html=True,
)



# Selector de clústers al cos principal
cluster_filter = None

if cluster_col is not None and cluster_ids_for_ui:
    st.subheader("Selecció de clústers temàtics")

    selected_labels = st.multiselect(
        "Clústers temàtics",
        options=cluster_labels_for_ui,
        default=[],  # cap clúster seleccionat per defecte
        help="Selecciona un o més clústers temàtics per filtrar l'evolució.",
    )

    # Mapegem de l'etiqueta visible al seu id real
    selected_ids = [label_to_id[lbl] for lbl in selected_labels]
    cluster_filter = selected_ids


# Filtres a la sidebar
st.sidebar.header("Filtres de l'evolució temporal")

year_range = st.sidebar.slider(
    "Període temporal",
    min_value=MIN_YEAR_DASHBOARD,
    max_value=MAX_YEAR_DASHBOARD,
    value=(min_year_global, max_year_global),
    step=1,
)

dept_filter = None
if dept_col is not None:
    all_depts = sorted(df_cyc[dept_col].dropna().unique().tolist())
    dept_selected = st.sidebar.multiselect(
        "Departaments",
        options=all_depts,
        default=all_depts,
    )
    dept_filter = dept_selected

# DataFrame filtrat global
df_filt = df_cyc.copy()
df_filt = df_filt[
    (df_filt[year_col] >= year_range[0]) & (df_filt[year_col] <= year_range[1])
]

if cluster_col is not None and cluster_filter:
    df_filt = df_filt[df_filt[cluster_col].isin(cluster_filter)]

if dept_col is not None and dept_filter:
    df_filt = df_filt[df_filt[dept_col].isin(dept_filter)]

# =============================================================================
# 4. EVOLUCIÓ GLOBAL ANUAL
# =============================================================================



st.header("1. Evolució anual global de la producció")

df_year_global = (
    df_filt.groupby(year_col)[count_col]
    .sum()
    .reset_index()
    .sort_values(year_col)
)

if df_year_global.empty:
    st.info("No hi ha dades per al rang d'anys i filtres seleccionats.")
else:
    colA, colB = st.columns(2)
    with colA:
        st.metric(
            "Documents en el període seleccionat",
            f"{int(df_year_global[count_col].sum()):,}",
        )
    with colB:
        st.metric(
            "Mitjana anual de documents",
            f"{df_year_global[count_col].mean():.1f}",
        )

    # ------------------------------
    # Gràfica global adaptativa
    # ------------------------------
    if cluster_filter:
        # Si hi ha clústers seleccionats → columnes
        fig_global = px.bar(
            df_year_global,
            x=year_col,
            y=count_col,
            labels={year_col: "Any", count_col: "Documents"},
        )
    else:
        # Vista global → línia
        fig_global = px.line(
            df_year_global,
            x=year_col,
            y=count_col,
            markers=True,
            labels={year_col: "Any", count_col: "Documents"},
        )
        fig_global.update_traces(line=dict(color=COLOR_PRIMARY_DARK))

    # ------------------------------
    # Ajust dinàmic del límit Y
    # ------------------------------
    y_max = df_year_global[count_col].max() if not df_year_global.empty else 0
    y_upper = y_max * 1.15 if y_max > 0 else 1  # lleu marge superior

    fig_global.update_layout(
        margin=dict(l=40, r=20, t=100, b=40),
        font=dict(color="#000078"),
        xaxis_title="Any",
        yaxis_title="Nombre de documents",
    )

    fig_global.update_xaxes(
        showline=True,
        linewidth=1,
        linecolor=COLOR_NEUTRAL_3,
        mirror=True,
        zeroline=False,
    )
    fig_global.update_yaxes(
        showline=True,
        linewidth=1,
        linecolor=COLOR_NEUTRAL_3,
        mirror=True,
        rangemode="tozero",
        range=[0, y_upper],
        zeroline=True,
        zerolinecolor=COLOR_NEUTRAL_3,
    )

    st.plotly_chart(fig_global, width="stretch")

st.markdown("---")

# =============================================================================
# 5. EVOLUCIÓ PER DEPARTAMENT
# =============================================================================

st.header("2. Evolució per departament")

# CAS A: el cluster_year_counts ja porta departament agregat
if dept_col is not None:
    df_dept_src = df_filt.copy()

    df_dept_src[year_col] = pd.to_numeric(df_dept_src[year_col], errors="coerce")
    df_dept_src = df_dept_src[df_dept_src[year_col].notna()]
    df_dept_src[year_col] = df_dept_src[year_col].astype(int)
    df_dept_src = df_dept_src[
        (df_dept_src[year_col] >= 2011) & (df_dept_src[year_col] <= 2025)
    ]

    df_dept = (
        df_dept_src.groupby([year_col, dept_col])[count_col]
        .sum()
        .reset_index()
        .sort_values([dept_col, year_col])
    )

    if df_dept.empty:
        st.info("No hi ha dades per als departaments i anys seleccionats.")
    else:
        topN = st.slider("Nombre màxim de departaments a mostrar", 3, 25, 8)
        total_per_dept = (
            df_dept.groupby(dept_col)[count_col]
            .sum()
            .reset_index()
            .sort_values(count_col, ascending=False)
        )
        top_depts = total_per_dept[dept_col].head(topN).tolist()
        df_dept_top = df_dept[df_dept[dept_col].isin(top_depts)]

        fig_dept = px.line(
            df_dept_top,
            x=year_col,
            y=count_col,
            color=dept_col,
            markers=True,
            labels={year_col: "Any", count_col: "Documents", dept_col: "Departament"},
            color_discrete_map=DEPT_COLOR_MAP,
        )
        fig_dept.update_layout(
            margin=dict(l=0, r=0, t=100, b=40),
            font=dict(color="#000078"),
            legend_title_text="Departament",
        )
        st.plotly_chart(fig_dept, width="stretch")

# CAS B: cluster_year_counts NO porta departament → fem servir el parquet principal
else:
    if df_main.empty or "Dept_main" not in df_main.columns:
        st.info(
            "No hi ha informació de departaments ni a `cluster_year_counts.parquet` "
            "ni al parquet principal. No es pot mostrar l'evolució per departament."
        )
    else:
        df_dep = df_main.copy()

        df_dep["year"] = pd.to_numeric(df_dep["year"], errors="coerce")
        df_dep = df_dep[df_dep["year"].notna()]
        df_dep["year"] = df_dep["year"].astype(int)
        df_dep = df_dep[(df_dep["year"] >= 2011) & (df_dep["year"] <= 2025)]

        if cluster_filter and "cluster_hdbscan" in df_dep.columns:
            df_dep = df_dep[df_dep["cluster_hdbscan"].isin(cluster_filter)]

        df_dep = df_dep[df_dep["Dept_main"].notna()]
        dept_col_effective = "Dept_main"

        df_dept = (
            df_dep.groupby(["year", dept_col_effective])
            .size()
            .reset_index(name="n_docs")
            .sort_values([dept_col_effective, "year"])
        )

        if df_dept.empty:
            st.info("No hi ha dades per als departaments i anys seleccionats.")
        else:
            topN = st.slider("Nombre màxim de departaments a mostrar", 3, 25, 8)
            total_per_dept = (
                df_dept.groupby(dept_col_effective)["n_docs"]
                .sum()
                .reset_index()
                .sort_values("n_docs", ascending=False)
            )
            top_depts = total_per_dept[dept_col_effective].head(topN).tolist()
            df_dept_top = df_dept[df_dept[dept_col_effective].isin(top_depts)]

            fig_dept = px.line(
                df_dept_top,
                x="year",
                y="n_docs",
                color=dept_col_effective,
                markers=True,
                labels={
                    "year": "Any",
                    "n_docs": "Documents",
                    dept_col_effective: "Departament",
                },
                color_discrete_map=DEPT_COLOR_MAP,
            )
            fig_dept.update_layout(
                margin=dict(l=0, r=0, t=100, b=40),
                font=dict(color="#000078"),
                legend_title_text="Departament",
            )
            st.plotly_chart(fig_dept, width="stretch")

st.markdown("---")

# =============================================================================
# 6. EVOLUCIÓ PER CLÚSTER (HEATMAP) I PATRONS
# =============================================================================

# =============================================================================
# 6. EVOLUCIÓ PER CLÚSTER (PATRONS TEMPORALS SENSE MAPA DE CALOR)
# =============================================================================

st.header("3. Evolució per clúster i patrons temporals")

if cluster_col is None:
    st.info(
        "L'artefacte `cluster_year_counts.parquet` no conté una columna de clúster "
        "(`cluster_label`, `cluster_hdbscan` o `cluster`). No es pot fer l'anàlisi per clúster."
    )
else:
    # Agrupem per clúster i any amb els filtres ja aplicats a df_filt
    df_cluster = (
        df_filt.groupby([cluster_col, year_col])[count_col]
        .sum()
        .reset_index()
    )

    if df_cluster.empty:
        st.info("No hi ha dades per als clústers i anys seleccionats.")
    else:
        # ---------- Patrons emergent / estable / decreixent ----------

        st.subheader("Tendència temàtica dels clústers (emergent, estable, decreixent)")

        patterns = []
        for cl in df_cluster[cluster_col].unique():
            df_c = df_cluster[df_cluster[cluster_col] == cl].sort_values(year_col)
            years = df_c[year_col].values
            counts = df_c[count_col].values

            if len(np.unique(years)) < 2:
                slope = 0.0
                pattern = "insuficient"
            else:
                x = np.arange(len(years))
                try:
                    slope, _ = np.polyfit(x, counts, 1)
                except Exception:
                    slope = 0.0

                if slope > 0.5:
                    pattern = "emergent"
                elif slope < -0.5:
                    pattern = "decreixent"
                else:
                    pattern = "estable"

            patterns.append(
                {
                    "cluster": cl,
                    "total_docs": int(df_c[count_col].sum()),
                    "slope": float(round(slope, 3)),
                    "pattern": pattern,
                }
            )

        df_patterns = pd.DataFrame(patterns)

        # KPIs de recompte de clústers per tipus de tendència
        colP1, colP2, colP3 = st.columns(3)
        with colP1:
            st.metric(
                "Clústers emergents",
                int((df_patterns["pattern"] == "emergent").sum()),
            )
        with colP2:
            st.metric(
                "Clústers estables",
                int((df_patterns["pattern"] == "estable").sum()),
            )
        with colP3:
            st.metric(
                "Clústers decreixents",
                int((df_patterns["pattern"] == "decreixent").sum()),
            )

        # ------------------------------------------------------------------
        # Taula de resum amb etiquetes semàntiques
        # ------------------------------------------------------------------
        df_patterns_display = df_patterns.copy()

        # Substituïm el número de clúster per la 5a etiqueta semàntica
        # Afegim número de clúster + etiqueta curta ("C006 — energy efficiency")
        def format_cluster_label(cid):
            cid_int = int(cid)
            base_label = mapping_5th.get(cid, str(cid))
            short = base_label.strip() if base_label else str(cid)
            return f"C{cid_int:03d} — {short}"

        if "mapping_5th" in globals() and isinstance(mapping_5th, dict):
            df_patterns_display["cluster"] = df_patterns_display["cluster"].apply(format_cluster_label)
        else:
            # Si per algun motiu no hi ha mapping, mostrem només el número
            df_patterns_display["cluster"] = df_patterns_display["cluster"].apply(lambda x: f"C{int(x):03d}")

        # Canvi de noms de columnes
        df_patterns_display = df_patterns_display.rename(
            columns={
                "cluster": "Clúster",
                "total_docs": "Documents totals",
                "slope": "Pendent",
                "pattern": "Tendència temàtica",
            }
        )

        # Ordenem columnes
        cols_order = ["Clúster", "Documents totals", "Pendent", "Tendència temàtica"]
        df_patterns_display = df_patterns_display[cols_order]

        st.markdown("**Taula resum de clústers per patró temporal**")
        st.dataframe(
            df_patterns_display.sort_values("Documents totals", ascending=False),
            width="stretch",
        )

st.markdown("---")

