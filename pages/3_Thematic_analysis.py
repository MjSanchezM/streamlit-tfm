# pages/3_Thematic_analisys.py

import os
import sys
import ast
import pandas as pd
import plotly.express as px
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode

# ================================================================
# 0. RUTES BASE I BRANDING (ARA TOTALMENT RELATIU AL REPO)
# ================================================================
from Paths_Project import (
    DOCS_ENRICHED_FILE,  # df_docs_kw_enriched_with_labels.parquet
    APP_DATA_DIR,
    TOOLS_DIR,
)

# Fitxer principal d’aquest mòdul (corpus enriquit)
MAIN_PARQUET = DOCS_ENRICHED_FILE

# Afegim tools al path si cal
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
except Exception:
    COLOR_PRIMARY = "#73EDFF"
    COLOR_PRIMARY_DARK = "#000078"
    COLOR_COMP1 = "#D6FAFF"
    COLOR_NEUTRAL_1 = "#f0f4ff"
    COLOR_NEUTRAL_2 = "#cccccc"
    COLOR_NEUTRAL_3 = "#878787"


px.defaults.template = "plotly_white"

PLOTLY_CORPORATE_THEME = dict(
    layout=dict(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color=COLOR_PRIMARY_DARK, size=12),
        #xaxis=dict(color=COLOR_NEUTRAL_3, showgrid=False),
        #yaxis=dict(color=COLOR_NEUTRAL_3, showgrid=False),
    )
)


# =============================================================================
# 1. CONFIG STREAMLIT
# =============================================================================

st.set_page_config(
    page_title="03 · Thematic analysis",
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
# 2. FUNCIONS DE CÀRREGA I HELPERS
# =============================================================================

@st.cache_data
def load_main_df():
    """Carrega el parquet principal amb tots els documents i metadades."""
    if not os.path.exists(MAIN_PARQUET):
        st.error(f"No s'ha trobat el parquet principal:\n{MAIN_PARQUET}")
        return pd.DataFrame()

    for engine in ["fastparquet", "pyarrow"]:
        try:
            df = pd.read_parquet(MAIN_PARQUET, engine=engine)
            break
        except Exception:
            df = pd.DataFrame()

    if df.empty:
        st.error("Error carregant el parquet principal amb qualsevol motor.")
        return df

    # Any normalitzat
    if "year" not in df.columns:
        if "AnyPubARPC" in df.columns:
            df["year"] = pd.to_numeric(df["AnyPubARPC"], errors="coerce").astype("Int64")
        else:
            df["year"] = pd.NA

    # Dept_main (assumim que Dept_list ja hi és; si no, fem una versió simple)
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

    # Assegurem kw_list_str per si cal
    if "kw_list_str" not in df.columns:
        if "kw_list" in df.columns:
            df["kw_list_str"] = df["kw_list"].astype(str)
        else:
            df["kw_list_str"] = ""

    return df


def get_5th_label(label: str) -> str:
    """
    Retorna la 5a keyword de cluster_label_auto (separada per ';'),
    o el text original si no n'hi ha prou.
    """
    if label is None or (isinstance(label, float) and pd.isna(label)):
        return ""
    parts = str(label).split(";")
    return parts[4].strip() if len(parts) >= 5 else str(label).strip()


def parse_kw_list(value):
    """
    Intenta convertir kw_list/kw_list_str en una llista de paraules clau.
    Accepta:
      - llista Python
      - string amb representació de llista
      - llista separada per comes
    """
    if isinstance(value, list):
        return [str(v).strip().strip("'\"") for v in value if str(v).strip()]
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []

    text = str(value).strip()
    if not text:
        return []

    # Intentem primer literal_eval (p.ex. "['a', 'b', 'c']")
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(v).strip().strip("'\"") for v in parsed if str(v).strip()]
    except Exception:
        pass

    # Fallback: separació per comes
    parts = [p.strip().strip("'\"") for p in text.split(",") if p.strip()]
    return parts


def build_cluster_keyword_stats(df_cluster: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """
    A partir de df_cluster, construeix una taula amb les paraules clau més freqüents.
    """
    all_kws = []
    if "kw_list" in df_cluster.columns:
        src = df_cluster["kw_list"]
    else:
        src = df_cluster["kw_list_str"]

    for v in src:
        kws = parse_kw_list(v)
        all_kws.extend(kws)

    if not all_kws:
        return pd.DataFrame(columns=["keyword", "count"])

    s = pd.Series(all_kws)
    counts = s.value_counts().reset_index()
    counts.columns = ["keyword", "count"]
    return counts.head(top_n)


def build_cluster_dept_stats(df_cluster: pd.DataFrame) -> pd.DataFrame:
    """
    Taula amb nombre de documents per departament principal.
    """
    if "Dept_main" not in df_cluster.columns:
        return pd.DataFrame(columns=["Dept_main", "n_docs"])

    tmp = (
        df_cluster["Dept_main"]
        .dropna()
        .astype(str)
        .value_counts()
        .reset_index()
    )
    tmp.columns = ["Dept_main", "n_docs"]
    return tmp


def build_cluster_year_evolution(df_cluster: pd.DataFrame) -> pd.DataFrame:
    """
    Sèrie temporal de nombre de documents per any dins del clúster.
    """
    if "year" not in df_cluster.columns:
        return pd.DataFrame(columns=["year", "n_docs"])

    df_tmp = df_cluster.copy()
    df_tmp["year"] = pd.to_numeric(df_tmp["year"], errors="coerce")
    df_tmp = df_tmp[df_tmp["year"].notna()]

    if df_tmp.empty:
        return pd.DataFrame(columns=["year", "n_docs"])

    df_year = (
        df_tmp.groupby("year")
        .size()
        .reset_index(name="n_docs")
        .sort_values("year")
    )

    # Filtre 2011-2025 per coherència amb la resta del dashboard
    df_year = df_year[(df_year["year"] >= 2011) & (df_year["year"] <= 2025)]

    if df_year.empty:
        return df_year

    # Ens assegurem que surtin tots els anys, encara que tinguin 0 docs
    all_years = pd.DataFrame({"year": list(range(2011, 2025 + 1))})
    df_year = all_years.merge(df_year, on="year", how="left")
    df_year["n_docs"] = df_year["n_docs"].fillna(0)

    return df_year


# =============================================================================
# 3. CÀRREGA
# =============================================================================

df = load_main_df()

if df.empty:
    st.error("No s'ha pogut carregar el corpus principal.")
    st.stop()

if "cluster_hdbscan" not in df.columns:
    st.error("Aquesta pàgina requereix la columna 'cluster_hdbscan' al parquet principal.")
    st.stop()

df_non_noise = df[df["cluster_hdbscan"] != -1].copy()
if df_non_noise.empty:
    st.error("No hi ha clústers HDBSCAN sense soroll; no es pot fer l'anàlisi temàtica.")
    st.stop()


# =============================================================================
# 4. CAPÇALERA
# =============================================================================

st.title("Anàlisi temàtica dels clústers")

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
      Aquesta pàgina se centra en l'<strong>anàlisi detallada d'un clúster temàtic</strong> 
      identificat pel model <strong>SBERT + UMAP + HDBSCAN</strong>. A partir de la selecció 
      d'un clúster, es mostra:
      <ul>
        <li>la seva <strong>mida relativa</strong> dins del corpus,</li>
        <li>el <strong>període temporal</strong> en què es concentra la producció,</li>
        <li>els <strong>departaments principals</strong> que hi contribueixen,</li>
        <li>les <strong>paraules clau representatives</strong> que defineixen el tema,</li>
        <li>i la <strong>llista d’articles</strong> associats, amb accés directe al repositori.</li>
      </ul>
      És una visió de <strong>"lupa temàtica"</strong>: permet entendre en profunditat 
      què representa cada clúster com a possible línia o àrea de recerca institucional.
    </div>
    """,
    unsafe_allow_html=True,
)


st.markdown("---")


# =============================================================================
# 5. SELECCIÓ DE CLÚSTER I RESUM
# =============================================================================

st.subheader("Selecció de clúster")

# Map de clúster -> etiqueta automàtica (si existeix)
label_map = {}
if "cluster_label_auto" in df_non_noise.columns:
    tmp = (
        df_non_noise[["cluster_hdbscan", "cluster_label_auto"]]
        .dropna()
        .drop_duplicates()
    )
    for _, row in tmp.iterrows():
        cid = int(row["cluster_hdbscan"])
        label_map[cid] = str(row["cluster_label_auto"])

cluster_ids = sorted(
    int(c)
    for c in df_non_noise["cluster_hdbscan"].dropna().unique().tolist()
    if int(c) != -1
)

def format_cluster_option(cid: int) -> str:
    base = f"C{cid:03d}"
    if cid in label_map:
        # Mostrem la 5a keyword com a "resum"
        short = get_5th_label(label_map[cid])
        if short:
            return f"{base} — {short}"
        return f"{base} — {label_map[cid]}"
    return base

if not cluster_ids:
    st.error("No hi ha clústers HDBSCAN vàlids (sense soroll).")
    st.stop()

cluster_options = [format_cluster_option(cid) for cid in cluster_ids]

# Per defecte, prenem el primer clúster (normalment el més petit id, no necessàriament el més gran)
selected_label = st.selectbox(
    "Clúster (HDBSCAN)",
    options=cluster_options,
    index=0,
)

selected_cid = int(selected_label.split(" ")[0].replace("C", ""))

df_cluster = df_non_noise[df_non_noise["cluster_hdbscan"] == selected_cid].copy()

if df_cluster.empty:
    st.warning("El clúster seleccionat no conté documents.")
    st.stop()

# KPIs del clúster
n_docs_total = len(df_non_noise)
n_docs_cluster = len(df_cluster)
share_pct = (n_docs_cluster / n_docs_total * 100) if n_docs_total > 0 else 0.0

years = pd.to_numeric(df_cluster["year"], errors="coerce")
years_valid = years[years.notna()]
if not years_valid.empty:
    min_year = int(years_valid.min())
    max_year = int(years_valid.max())
else:
    min_year = None
    max_year = None

n_depts = df_cluster["Dept_main"].dropna().astype(str).nunique()

cluster_label_full = label_map.get(selected_cid, "")
cluster_label_5th = get_5th_label(cluster_label_full)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Documents al clúster",
        f"{n_docs_cluster:,}".replace(",", "."),
        delta=f"{share_pct:.1f}% del corpus (sense soroll)",
    )

with col2:
    if min_year is not None and max_year is not None:
        st.metric("Període temporal", f"{min_year} — {max_year}")
    else:
        st.metric("Període temporal", "n/d")

with col3:
    st.metric("Departaments implicats", str(n_depts))

with col4:
    if cluster_label_5th:
        st.metric("Etiqueta automàtica (5a keyword)", cluster_label_5th)
    elif cluster_label_full:
        st.metric("Etiqueta automàtica", cluster_label_full)
    else:
        st.metric("Etiqueta automàtica", "n/d")

st.markdown("---")


# =============================================================================
# 6. PARAULES CLAU DEL CLÚSTER
# =============================================================================

st.subheader("Perfil semàntic del clúster: paraules clau")

kw_stats = build_cluster_keyword_stats(df_cluster, top_n=20)

if kw_stats.empty:
    st.info("No s'han pogut calcular paraules clau representatives per aquest clúster.")
else:
    fig_kw = px.bar(
        kw_stats.sort_values("count", ascending=True),
        x="count",
        y="keyword",
        orientation="h",
    )
    fig_kw.update_layout(
        xaxis_title="Documents al clúster",
        yaxis_title="Paraula clau",
        margin=dict(l=0, r=10, t=30, b=0),
        **PLOTLY_CORPORATE_THEME["layout"],
    )
    st.plotly_chart(fig_kw, width="stretch")

st.markdown("---")


# =============================================================================
# 7. DEPARTAMENTS I EVOLUCIÓ TEMPORAL
# =============================================================================

st.subheader("Context institucional i temporal del clúster")

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

col_g1, col_g2 = st.columns(2)

# 7.1 Distribució per departament
with col_g1:
    st.markdown("**Distribució per departament principal (Dept_main)**")

    df_dept = (
        df_cluster.groupby("Dept_main")
        .size()
        .reset_index(name="n_docs")
        .sort_values("n_docs", ascending=True)
    )

    fig_dept = px.bar(
        df_dept,
        x="n_docs",
        y="Dept_main",
        orientation="h",
        color="Dept_main",                  # color per departament
        color_discrete_map=DEPT_COLOR_MAP,  # mapa de colors institucional
    )

    fig_dept.update_layout(
        xaxis_title="Documents al clúster",
        yaxis_title="Departament",
        showlegend=False,
        margin=dict(l=20, r=20, t=30, b=20),
        height=500,
        **PLOTLY_CORPORATE_THEME["layout"],
    )

    st.plotly_chart(fig_dept, width="stretch")

# 7.2 Evolució temporal
with col_g2:
    st.markdown("**Evolució anual de la producció del clúster**")

    df_year_cluster = build_cluster_year_evolution(df_cluster)

    if df_year_cluster.empty:
        st.info("No hi ha dades suficients per mostrar l'evolució temporal del clúster.")
    else:
        fig_year = px.line(
            df_year_cluster,
            x="year",
            y="n_docs",
            markers=True,
        )
        fig_year.update_traces(
            line=dict(color=COLOR_PRIMARY_DARK),
        )
        fig_year.update_layout(
            xaxis_title="Any",
            yaxis_title="Documents",
            margin=dict(l=0, r=10, t=30, b=0),
            xaxis=dict(
                tickmode="array",
                tickvals=list(range(2011, 2025 + 1)),
            ),
            yaxis=dict(
                rangemode="tozero",
            ),
            **PLOTLY_CORPORATE_THEME["layout"],
        )
        st.plotly_chart(fig_year, width="stretch")

st.markdown("---")



# =============================================================================
# 8. TAULA DE DOCUMENTS DEL CLÚSTER (AgGrid amb URL clicable)
# =============================================================================

st.subheader("Documents del clúster seleccionat")

cols_raw = ["Title", "year", "Dept_main", "handle_url"]
cols_available = [c for c in cols_raw if c in df_cluster.columns]

if not cols_available:
    st.info("No s'han trobat columnes adequades per mostrar la llista de documents.")
else:
    # 1) DataFrame de treball
    df_table = df_cluster[cols_available].copy()

    # Any numèric + ordenació
    if "year" in df_table.columns:
        df_table["year"] = pd.to_numeric(df_table["year"], errors="coerce")
        df_table = df_table.sort_values("year", ascending=False)

    # 2) Renombrar columnes visibles
    df_table = df_table.rename(columns={
        "Title": "Article",
        "year": "Any",
        "Dept_main": "Departament",
    })

    # 3) Columna de link neta: només URL plana
    df_table["LinkURL"] = df_table["handle_url"].astype(str)
    df_table = df_table.drop(columns=["handle_url"])

    # 4) Renderer JS tipus classe (solució del fòrum Streamlit)
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
            this.eGui.setAttribute('style', 'text-decoration:none;');
          }
          getGui() {
            return this.eGui;
          }
        }
        """
    )

    # 5) Configuració d'AgGrid
    gb = GridOptionsBuilder.from_dataframe(df_table)

    gb.configure_default_column(
        sortable=True,
        filter=True,
        resizable=True,
        autoHeight=True,
        wrapText=False,
    )

    if "Article" in df_table.columns:
        gb.configure_column("Article", width=280)
    if "Any" in df_table.columns:
        gb.configure_column("Any", width=80)
    if "Departament" in df_table.columns:
        gb.configure_column("Departament", width=160)

    gb.configure_column(
        "LinkURL",
        headerName="Enllaç",
        width=420,
        cellRenderer=link_renderer,
    )

    grid_options = gb.build()

    AgGrid(
        df_table,
        gridOptions=grid_options,
        enable_enterprise_modules=False,
        fit_columns_on_grid_load=False,
        update_mode=GridUpdateMode.NO_UPDATE,
        theme="streamlit",
        allow_unsafe_jscode=True,
    )

st.markdown("---")


