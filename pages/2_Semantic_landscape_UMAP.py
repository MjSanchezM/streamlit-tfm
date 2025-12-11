# pages/2_Semantic_landscape_UMAP.py

import sys
import ast
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.colors as pc
import streamlit as st

# --------------------------------------------------------------
# INTENTEM FER SERVIR Paths_Project (si existeix)
# --------------------------------------------------------------
try:
    import Paths_Project as P
except ImportError:
    P = None

# --------------------------------------------------------------
# DIRECTORIS BASE (derivats de la ubicació d'aquest fitxer)
# --------------------------------------------------------------
# Aquest fitxer està a: .../GitHub/VisualAnalytics/pages/2_Semantic_landscape_UMAP.py

VISUAL_DIR = Path(__file__).resolve().parents[1]   # .../GitHub/VisualAnalytics
REPO_DIR   = VISUAL_DIR.parent                     # .../GitHub

APP_DATA_DIR = VISUAL_DIR / "app_data"             # .../GitHub/VisualAnalytics/app_data
TOOLS_DIR    = REPO_DIR / "tools"                  # .../GitHub/tools

# Si Paths_Project està disponible, prioritzem els seus paths
if P is not None:
    if hasattr(P, "APP_DATA_DIR"):
        APP_DATA_DIR = P.APP_DATA_DIR
    if hasattr(P, "TOOLS_DIR"):
        TOOLS_DIR = P.TOOLS_DIR

# --------------------------------------------------------------
# PATHS DELS ARXIUS DE DADES
# --------------------------------------------------------------
DOCS_FILE = APP_DATA_DIR / "df_docs_kw_enriched_with_labels.parquet"
UMAP_FILE = APP_DATA_DIR / "df_docs_full_umap_simple.parquet"

# --------------------------------------------------------------
# AFEGIM tools AL SYS.PATH ABANS D'IMPORTAR plot_style
# --------------------------------------------------------------
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))  # millor insert(0) que append

# (opcional, útil per comprovar a la consola)
print(">>> TOOLS_DIR:", TOOLS_DIR)
print(">>> Té plot_style.py?", (TOOLS_DIR / "plot_style.py").exists())

# --------------------------------------------------------------
# IMPORT DEL TEMA GRÀFIC CORPORATIU
# --------------------------------------------------------------
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


# -------------------------------------------------------------------
# CONFIGURACIÓ BÀSICA DE LA PÀGINA
# -------------------------------------------------------------------
st.set_page_config(
    page_title="02 · Semantic Landscape UMAP",
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


# -------------------------------------------------------------------
# MAPA DE COLORS PER Dept_main
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
DEPT_COLOR_MAP["Gestió d'empreses"] = DEPT_COLOR_MAP["Gestió d'Empreses"]

# -------------------------------------------------------------------
# MAPA DE COLORS PER PLOT UMAP
# -------------------------------------------------------------------

# Escala de colors reutilitzada del Treemap (01_Overview)
TREEMAP_COLOR_SCALE = ["#F0F0F0", "#878787"]  # degradat gris clar → gris fosc

# -------------------------------------------------------------------
# PER LA GRÀFICA UMAP - COLOR DEL SOROLL
# -------------------------------------------------------------------
NOISE_COLOR = "#73EDFF"   # color específic per al clúster -1 (No classificats)


def make_cluster_color_map(df: pd.DataFrame) -> dict:
    """
    Assigna un color discret per a cada clúster HDBSCAN (excloent -1),
    fent servir l'escala del Treemap. Afageix també un color fix
    per al clúster de soroll (-1).
    """
    if "cluster_hdbscan" not in df.columns:
        return {}

    # Clústers reals (sense el soroll)
    clusters = sorted(
        int(c)
        for c in df["cluster_hdbscan"].dropna().unique().tolist()
        if int(c) != -1
    )

    n = len(clusters)
    if n == 0:
        return {"-1": NOISE_COLOR}

    # Posicions normalitzades per repartir els colors
    if n == 1:
        positions = [0.5]
    else:
        positions = [i / (n - 1) for i in range(n)]

    colors = pc.sample_colorscale(TREEMAP_COLOR_SCALE, positions)

    # Diccionari final de colors (Plotly demana claus en string)
    color_map = {str(cid): colors[i] for i, cid in enumerate(clusters)}

    # Afegeix el soroll (-1) amb color neutre
    color_map["-1"] = NOISE_COLOR

    return color_map






# -------------------------------------------------------------------
# TEMA PLOTLY CORPORATIU
# -------------------------------------------------------------------
PLOTLY_CORPORATE_THEME = dict(
    layout=dict(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color=COLOR_PRIMARY_DARK, size=12),
        xaxis=dict(color=COLOR_NEUTRAL_3, showgrid=False),
        yaxis=dict(color=COLOR_NEUTRAL_3, showgrid=False),
        #title=dict(font=dict(color=COLOR_PRIMARY_DARK, size=18)),
    )
)

# -------------------------------------------------------------------
# CÀRREGA DE DADES
# -------------------------------------------------------------------
@st.cache_data
def load_data():
    df_docs = pd.read_parquet(DOCS_FILE, engine="fastparquet")
    df_umap = pd.read_parquet(UMAP_FILE, engine="fastparquet")

    if "doc_id" not in df_umap.columns:
        raise ValueError(
            "Al fitxer UMAP no hi ha la columna 'doc_id'. "
            f"Columnes disponibles: {list(df_umap.columns)}"
        )

    for c in ["umap_x", "umap_y"]:
        if c not in df_umap.columns:
            raise ValueError(
                "Al fitxer UMAP no hi ha les columnes esperades umap_x/umap_y. "
                f"Columnes disponibles: {list(df_umap.columns)}"
            )

    df_umap = df_umap[["doc_id", "umap_x", "umap_y"]].copy()
    df = df_docs.merge(df_umap, on="doc_id", how="inner")

    for base_col in ["cluster_hdbscan", "cluster_kmeans"]:
        col_x = f"{base_col}_x"
        col_y = f"{base_col}_y"
        if col_x in df.columns and col_y in df.columns:
            df[base_col] = df[col_x]
            df = df.drop(columns=[col_x, col_y])
        elif col_x in df.columns:
            df = df.rename(columns={col_x: base_col})
        elif col_y in df.columns:
            df = df.rename(columns={col_y: base_col})

    if "AnyPubARPC" in df.columns:
        df["year"] = pd.to_numeric(df["AnyPubARPC"], errors="coerce").astype("Int64")
    elif "year" not in df.columns:
        df["year"] = pd.NA

    if "kw_list" in df.columns:
        df["kw_list_str"] = df["kw_list"].astype(str)
    else:
        df["kw_list_str"] = pd.NA

    if "Dept_normalized" not in df.columns:
        if "DeptARPC" in df.columns:
            df["Dept_normalized"] = df["DeptARPC"]
        else:
            df["Dept_normalized"] = pd.NA

    if "Dept_list" not in df.columns:
        def split_departments(value):
            if pd.isna(value):
                return []
            return [v.strip() for v in str(value).split(";") if v.strip()]
        df["Dept_list"] = df["Dept_normalized"].apply(split_departments)

    if "Dept_main" not in df.columns:
        def get_main_department(dept_list):
            if isinstance(dept_list, list) and len(dept_list) > 0:
                return dept_list[0]
            return None
        df["Dept_main"] = df["Dept_list"].apply(get_main_department)

    return df


# -------------------------------------------------------------------
# CÀRREGA
# -------------------------------------------------------------------
try:
    df = load_data()
except Exception as e:
    st.error(f"S'ha produït un error carregant les dades: {e}")
    st.stop()

if df.empty:
    st.warning("El dataframe resultant està buit després del merge.")
    st.stop()

# -------------------------------------------------------------------
# CAPÇALERA
# -------------------------------------------------------------------
st.title("Mapa temàtic UMAP del corpus")

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
      Aquesta pàgina mostra el <strong>paisatge semàntic del corpus</strong> en 2D, 
      a partir del pipeline <strong>SBERT + UMAP + HDBSCAN/k-means</strong>. 
      Cada punt correspon a un document del repositori institucional, projectat 
      en un espai on:
      <ul>
        <li>documents <strong>semànticament similars</strong> apareixen pròxims,</li>
        <li>els <strong>clústers temàtics</strong> es poden identificar visualment,</li>
        <li>i és possible resseguir <strong>patrons per any i departament</strong>.</li>
      </ul>
      Els filtres superiors permeten restringir el mapa per <strong>període temporal</strong>, 
      <strong>departament principal</strong> i <strong>clúster HDBSCAN</strong>, mentre que 
      el <strong>mode de color</strong> ajuda a canviar la lectura (per clúster, per any 
      o per departament). És la vista més global de l'espai temàtic generat pel model.
    </div>
    """,
    unsafe_allow_html=True,
)


st.markdown("---")

# -------------------------------------------------------------------
# FILTRES AL COS
# -------------------------------------------------------------------
st.subheader("Filtres")

df_filter = df.copy()

# 1 = més estret, 2 = més ample
col_f1, col_f2, col_f3 = st.columns([1, 1.3, 1.7])


# Any
with col_f1:
    if df_filter["year"].notna().any():
        # Ens assegurem que year és numèric
        years_series = pd.to_numeric(df_filter["year"], errors="coerce")
        years_series = years_series.dropna().astype(int)

        if not years_series.empty:
            # Límits oficials del dashboard
            MIN_YEAR_DASHBOARD = 2011
            MAX_YEAR_DASHBOARD = 2025

            # Anys reals dins del rang 2011–2025
            years_in_range = [
                y for y in years_series.unique()
                if MIN_YEAR_DASHBOARD <= y <= MAX_YEAR_DASHBOARD
            ]

            if years_in_range:
                default_from = min(years_in_range)
                default_to = max(years_in_range)
            else:
                # Si per algun motiu no hi ha dades dins 2011–2025,
                # fem servir igualment els límits del dashboard
                default_from = MIN_YEAR_DASHBOARD
                default_to = MAX_YEAR_DASHBOARD

            year_range = st.slider(
                "Rang d'anys de publicació",
                min_value=MIN_YEAR_DASHBOARD,
                max_value=MAX_YEAR_DASHBOARD,
                value=(default_from, default_to),
                step=1,
            )

            df_filter = df_filter[
                df_filter["year"].isna()
                | (
                    (df_filter["year"] >= year_range[0])
                    & (df_filter["year"] <= year_range[1])
                )
            ]

# Departament principal
with col_f2:
    if "Dept_main" in df_filter.columns:
        all_depts = (
            df_filter["Dept_main"]
            .dropna()
            .unique()
            .tolist()
        )
        all_depts = sorted(all_depts)
        dept_options = ["(Tots els departaments)"] + all_depts

        selected_dept = st.selectbox(
            "Departament principal (Dept_main)",
            options=dept_options,
            index=0,
        )

        if selected_dept != "(Tots els departaments)":
            df_filter = df_filter[
                df_filter["Dept_list"].apply(
                    lambda lst: isinstance(lst, list) and selected_dept in lst
                )
            ]

# -------------------------------------------------------
# Clúster HDBSCAN (amb possibilitat de seleccionar soroll)
# -------------------------------------------------------
with col_f3:
    selected_cluster = None

    if "cluster_hdbscan" in df_filter.columns:

        # Helper: agafa la 5a keyword de l'etiqueta (o tot el text si n'hi ha menys)
        def get_5th_label_for_option(label):
            if label is None or (isinstance(label, float) and pd.isna(label)):
                return ""
            parts = str(label).split(";")
            return parts[4].strip() if len(parts) >= 5 else str(label).strip()

        # Associar etiqueta humana si existeix
        label_map = {}
        if "cluster_label_auto" in df_filter.columns:
            tmp = (
                df_filter[["cluster_hdbscan", "cluster_label_auto"]]
                .dropna()
                .drop_duplicates()
            )
            for _, row in tmp.iterrows():
                cid = int(row["cluster_hdbscan"])
                label_map[cid] = str(row["cluster_label_auto"])

        # Llista de clústers incloent el -1
        raw_clusters = sorted(
            int(c)
            for c in df_filter["cluster_hdbscan"].dropna().unique().tolist()
        )

        # Representació amable
        def format_cluster_option(cid):
            if cid == -1:
                # abans: "Soroll (-1)"
                return "No classificats (-1)"
            base = f"C{cid:03d}"
            if cid in label_map:
                short = get_5th_label_for_option(label_map[cid])
                if short:
                    return f"{base} — {short}"
                return base
            return base

        cluster_options = ["(Tots els clústers)"] + [
            format_cluster_option(cid) for cid in raw_clusters
        ]

        selection = st.selectbox(
            "Clúster (HDBSCAN)",
            options=cluster_options,
            index=0,
        )

        if selection != "(Tots els clústers)":
            # Convertir text → id
            if selection.startswith("No classificats"):
                selected_cluster = -1
            else:
                selected_cluster = int(
                    selection.split(" ")[0].replace("C", "")
                )

            df_filter = df_filter[df_filter["cluster_hdbscan"] == selected_cluster]

        else:
            # En la vista global, amaguem el -1 del mapa
            df_filter = df_filter[df_filter["cluster_hdbscan"] != -1]



if df_filter.empty:
    st.warning("Cap document compleix els filtres seleccionats.")
    st.stop()

st.markdown("---")

# -------------------------------------------------------------------
# MODE DE COLOR
# -------------------------------------------------------------------
color_mode = st.radio(
    "Mode de color al mapa UMAP:",
    options=[
        "Per clúster",
        "Per any de publicació",
        "Per departament principal (Dept_main)",
    ],
    horizontal=True,
)

color_arg = None
color_kwargs = {}

if color_mode == "Per clúster":
    if "cluster_hdbscan" in df_filter.columns:
        # Fem servir una versió string per poder aplicar color_discrete_map
        df_filter["cluster_hdbscan_str"] = (
            df_filter["cluster_hdbscan"]
            .astype("Int64")
            .astype(str)
        )
        color_arg = "cluster_hdbscan_str"

        # Mapa de colors discret basat en l'escala del Treemap
        cluster_color_map = make_cluster_color_map(df_filter)
        color_kwargs["color_discrete_map"] = cluster_color_map

    elif "cluster_kmeans" in df_filter.columns:
        # Si vols, aquí també podríem generar un color map similar per k-means
        color_arg = "cluster_kmeans"

    else:
        st.info(
            "No s'ha trobat cap columna de clúster; es farà servir el departament principal."
        )
        color_arg = "Dept_main"

elif color_mode == "Per any de publicació":
    color_arg = "year"

elif color_mode == "Per departament principal (Dept_main)":
    color_arg = "Dept_main"
    color_kwargs["color_discrete_map"] = DEPT_COLOR_MAP

# -------------------------------------------------------------------
# FIGURA UMAP INTERACTIVA
# -------------------------------------------------------------------

# Preparem columnes auxiliars per al tooltip
df_plot = df_filter.copy()

# 1) 5a paraula clau de cluster_label_auto
def get_5th_label(label):
    if pd.isna(label):
        return ""
    parts = str(label).split(";")
    return parts[4].strip() if len(parts) >= 5 else str(label).strip()

if "cluster_label_auto" in df_plot.columns:
    df_plot["cluster_label_5th"] = df_plot["cluster_label_auto"].apply(get_5th_label)
else:
    df_plot["cluster_label_5th"] = ""

# 2) Top 5 paraules clau en format HTML
def format_kw_top5(val):
    # intentem tractar-ho com a llista python
    kws = []
    if isinstance(val, list):
        kws = val
    else:
        try:
            parsed = ast.literal_eval(str(val))
            if isinstance(parsed, list):
                kws = parsed
            else:
                kws = str(val).split(",")
        except Exception:
            kws = str(val).split(",")

    kws = [str(k).strip().strip("'\"") for k in kws if str(k).strip()]
    kws = kws[:5]

    if not kws:
        return ""

    lines = "<br>".join(f"- {k}" for k in kws)
    return "Paraules clau:<br>" + lines

if "kw_list" in df_plot.columns:
    src_kw = df_plot["kw_list"]
else:
    src_kw = df_plot.get("kw_list_str", "")

df_plot["kw_top5_html"] = src_kw.apply(format_kw_top5)

# Ens assegurem que hi ha columna Dept_main
if "Dept_main" not in df_plot.columns:
    df_plot["Dept_main"] = df_plot.get("Dept_normalized", "")

# Scatter amb custom_data per controlar completament el tooltip
fig = px.scatter(
    df_plot,
    x="umap_x",
    y="umap_y",
    color=color_arg,
    height=650,
    custom_data=[
        "cluster_hdbscan",     # 0
        "umap_x",              # 1
        "umap_y",              # 2
        "doc_id",              # 3
        "Title",               # 4
        "year",                # 5
        "Dept_main",           # 6
        "cluster_label_5th",   # 7
        "kw_top5_html",        # 8 (HTML amb la llista)
    ],
    **color_kwargs,
)

fig.update_traces(
    hovertemplate=(
        "<b>Número de clúster:</b> C%{customdata[0]:03d}<br><br>"

        "<b>Coordenades UMAP:</b> "
        "(%{customdata[1]:.3f}, %{customdata[2]:.3f})<br><br>"

        "<b>Identificació de l’article:</b><br>"
        "%{customdata[3]} — %{customdata[4]} (%{customdata[5]})<br><br>"

        "<b>Departament:</b> %{customdata[6]}<br><br>"

        "<b>Etiqueta automàtica:</b> %{customdata[7]}<br><br>"

        #"<b>Paraules clau:</b><br>"
        "%{customdata[8]}"   # ja és HTML amb salts de línia
        "<extra></extra>"
    )
)


fig.update_layout(
    xaxis_title="UMAP 1",
    yaxis_title="UMAP 2",
    legend_title="",
    **PLOTLY_CORPORATE_THEME["layout"],
)

st.plotly_chart(fig, width="stretch")


# -------------------------------------------------------------------
# RESUM
# -------------------------------------------------------------------
st.markdown("### Resum dels filtres aplicats")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Documents filtrats",
    f"{len(df_filter):,}".replace(",", "."),
)

if "cluster_hdbscan" in df_filter.columns:
    n_clusters = df_filter.loc[
        df_filter["cluster_hdbscan"] != -1, "cluster_hdbscan"
    ].nunique()
    col2.metric(
        "Clústers HDBSCAN presents",
        str(n_clusters),
    )
elif "cluster_kmeans" in df_filter.columns:
    col2.metric(
        "Clústers k-means presents",
        str(df_filter["cluster_kmeans"].nunique()),
    )
else:
    col2.metric("Clústers presents", "n/d")

if "Dept_list" in df_filter.columns:
    depts_present = set()
    for lst in df_filter["Dept_list"]:
        if isinstance(lst, list):
            depts_present.update(lst)
    col3.metric(
        "Departaments presents",
        str(len(depts_present)),
    )
elif "Dept_normalized" in df_filter.columns:
    col3.metric(
        "Departaments presents",
        str(df_filter["Dept_normalized"].nunique()),
    )
else:
    col3.metric("Departaments presents", "n/d")

st.markdown("---")




