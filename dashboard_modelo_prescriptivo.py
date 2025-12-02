import streamlit as st
import pandas as pd
import numpy as np
import os
import pulp

# ==========================
# CONFIG
# ==========================
BASE_PATH = os.path.join(os.path.dirname(__file__), "data")


# ==========================
# CARGA DE PARQUETS
# ==========================
@st.cache_data
def cargar_datos(base_path: str):
    """
    Lee los datasets ya procesados (a nivel de modelo constelación)
    desde archivos Parquet.
    - df_presc: exportaciones MINCETUR (anio, region, destino, bloque, sector, valor, toneladas)
    - df_vias:  comercio exterior SUNAT agregado por año y via_transporte
    """
    df_presc = pd.read_parquet(os.path.join(base_path, "modelo_prescriptivo_dataset.parquet"))
    df_vias  = pd.read_parquet(os.path.join(base_path, "sunat_vias_dataset.parquet"))
    return df_presc, df_vias


# ==========================
# HELPERS DE FORMATO
# ==========================
def formatear_tabla_regiones(df: pd.DataFrame) -> pd.DataFrame:
    """Formatea valores de regiones con unidades (M, T, %) para la vista en tablas."""
    df_fmt = df.copy()

    # Valor total (ya viene en millones)
    if "valor_total" in df_fmt.columns:
        df_fmt["valor_total"] = df_fmt["valor_total"].apply(lambda x: f"{x:,.2f} M")

    # Millones por tonelada
    if "valor_por_ton" in df_fmt.columns:
        df_fmt["valor_por_ton"] = df_fmt["valor_por_ton"].apply(lambda x: f"{x:,.4f} M/T")

    # Toneladas óptimas
    if "toneladas_optimas" in df_fmt.columns:
        df_fmt["toneladas_optimas"] = df_fmt["toneladas_optimas"].apply(
            lambda x: f"{x:,.0f} T"
        )

    # Valor estimado en millones
    if "valor_millones_estimado" in df_fmt.columns:
        df_fmt["valor_millones_estimado"] = df_fmt["valor_millones_estimado"].apply(
            lambda x: f"{x:,.2f} M"
        )

    # Columnas de participación → porcentaje
    cols_pct = [c for c in df_fmt.columns if c.startswith("participacion_")]
    for c in cols_pct:
        df_fmt[c] = (df_fmt[c] * 100).round(1).astype(str) + " %"

    return df_fmt


def formatear_tabla_porcentajes(df: pd.DataFrame, cols_pct) -> pd.DataFrame:
    """Convierte proporciones (0–1) en porcentajes con 1 decimal."""
    df_fmt = df.copy()
    for c in cols_pct:
        df_fmt[c] = (df_fmt[c] * 100).round(1).astype(str) + " %"
    return df_fmt


# ==========================
# CONSTRUCCIÓN DE MODELOS
# ==========================
@st.cache_data
def construir_modelos(df_presc, df_vias, anio_desde, anio_hasta):
    """
    df_presc: dataset prescriptivo MINCETUR (Parquet)
        columnas mínimas: anio, region_mincetur, destino, bloque_comercial, sector,
                          valor_millones, toneladas_metricas
    df_vias: dataset de vías SUNAT (Parquet)
        columnas mínimas: anio, via_transporte, valor_millones
    """

    # ---- Filtrar rango de años ----
    df = df_presc.copy()
    df = df[(df["anio"] >= anio_desde) & (df["anio"] <= anio_hasta)].copy()
    df = df[df["region_mincetur"].notna()].copy()
    df = df[df["toneladas_metricas"] > 0].copy()

    # ---- Indicadores regionales básicos ----
    region_stats = (
        df.groupby("region_mincetur", as_index=False)
          .agg(
              valor_total=("valor_millones", "sum"),
              toneladas_totales=("toneladas_metricas", "sum"),
              anios_distintos=("anio", "nunique"),
              n_registros=("anio", "count")
          )
    )
    region_stats["valor_por_ton"] = region_stats["valor_total"] / region_stats["toneladas_totales"]
    region_stats["valor_prom_anual"] = region_stats["valor_total"] / region_stats["anios_distintos"]

    regiones = region_stats["region_mincetur"].tolist()

    # ======================================================
    # 1) MODELO OPERATIVO (TONELADAS + MILLONES)
    # ======================================================
    valor_por_ton_dict = dict(zip(region_stats["region_mincetur"], region_stats["valor_por_ton"]))

    op_model = pulp.LpProblem("Modelo_Operativo_Toneladas", pulp.LpMaximize)
    t = pulp.LpVariable.dicts("t", regiones, lowBound=0)

    # Objetivo: maximizar valor estimado
    op_model += pulp.lpSum(t[r] * valor_por_ton_dict[r] for r in regiones)

    # Mantener el total de toneladas históricas
    total_ton_hist = region_stats["toneladas_totales"].sum()
    op_model += pulp.lpSum(t[r] for r in regiones) == total_ton_hist

    # Equidad operativa: mínimo 1%, máximo 20% por región
    min_share = 0.01
    max_share = 0.20
    for r in regiones:
        op_model += t[r] >= min_share * total_ton_hist
        op_model += t[r] <= max_share * total_ton_hist

    op_model.solve(pulp.PULP_CBC_CMD(msg=False))

    op_region = region_stats[["region_mincetur", "valor_por_ton"]].copy()
    op_region["toneladas_optimas"] = op_region["region_mincetur"].apply(lambda r: t[r].value())
    op_region["valor_millones_estimado"] = op_region["toneladas_optimas"] * op_region["valor_por_ton"]

    total_ton_opt = op_region["toneladas_optimas"].sum()
    total_val_opt = op_region["valor_millones_estimado"].sum()

    op_region["participacion_ton_opt"] = op_region["toneladas_optimas"] / total_ton_opt
    op_region["participacion_valor_opt"] = op_region["valor_millones_estimado"] / total_val_opt

    op_region = op_region.sort_values("valor_millones_estimado", ascending=False)

    # ======================================================
    # 2) MODELO ESTRATÉGICO (PORCENTAJES DE ESFUERZO)
    # ======================================================
    def min_max_norm(s):
        if s.max() == s.min():
            return s * 0
        return (s - s.min()) / (s.max() - s.min())

    region_stats["score_valor"] = min_max_norm(region_stats["valor_prom_anual"])
    region_stats["score_rentabilidad"] = min_max_norm(region_stats["valor_por_ton"])
    region_stats["score_total"] = 0.6 * region_stats["score_valor"] + 0.4 * region_stats["score_rentabilidad"]

    scores = dict(zip(region_stats["region_mincetur"], region_stats["score_total"]))

    strat_model = pulp.LpProblem("Modelo_Prescriptivo_Regional", pulp.LpMaximize)
    w = pulp.LpVariable.dicts("w", regiones, lowBound=0, upBound=1)

    strat_model += pulp.lpSum(w[r] * scores[r] for r in regiones)
    strat_model += pulp.lpSum(w[r] for r in regiones) == 1

    epsilon = 0.005
    max_region = 0.15
    for r in regiones:
        strat_model += w[r] >= epsilon
        strat_model += w[r] <= max_region

    strat_model.solve(pulp.PULP_CBC_CMD(msg=False))

    region_stats["participacion_optima"] = region_stats["region_mincetur"].apply(lambda r: w[r].value())

    # Histórico por región (%)
    region_hist = df.groupby("region_mincetur")["valor_millones"].sum()
    region_hist_share = region_hist / region_hist.sum()

    comparacion_regiones = region_stats.merge(
        region_hist_share.rename("participacion_historica"),
        left_on="region_mincetur",
        right_index=True,
        how="left"
    )

    # Diccionarios de pesos por región
    region_opt_strat = dict(zip(region_stats["region_mincetur"], region_stats["participacion_optima"]))
    region_opt_oper  = dict(zip(op_region["region_mincetur"], op_region["participacion_valor_opt"]))

    # ==========================================
    # Helper para proyectar a bloque / sector / país
    # ==========================================
    def distribucion_por_dimension(df_local, dimension, region_weights_dict,
                                   region_col="region_mincetur", valor_col="valor_millones"):
        tmp = (
            df_local.groupby([region_col, dimension])[valor_col]
                    .sum()
                    .reset_index()
                    .rename(columns={valor_col: "valor_hist"})
        )

        total_region = (
            df_local.groupby(region_col)[valor_col]
                    .sum()
                    .reset_index()
                    .rename(columns={valor_col: "valor_region"})
        )

        tmp = tmp.merge(total_region, on=region_col, how="left")
        tmp["share_dim_en_region"] = tmp["valor_hist"] / tmp["valor_region"]
        tmp["peso_region"] = tmp[region_col].map(region_weights_dict)
        tmp["contribucion_dim"] = tmp["peso_region"] * tmp["share_dim_en_region"]

        return tmp.groupby(dimension)["contribucion_dim"].sum().sort_values(ascending=False)

    # ---- Histórico por dimensión (MINCETUR) ----
    bloque_hist = df.groupby("bloque_comercial")["valor_millones"].sum()
    bloque_hist_share = bloque_hist / bloque_hist.sum()

    sector_hist = df.groupby("sector")["valor_millones"].sum()
    sector_hist_share = sector_hist / sector_hist.sum()

    pais_hist = df.groupby("destino")["valor_millones"].sum()
    pais_hist_share = pais_hist / pais_hist.sum()

    # ---- Estrategia vs Operativo proyectado a cada dimensión ----
    bloque_strat_share = distribucion_por_dimension(df, "bloque_comercial", region_opt_strat)
    bloque_oper_share  = distribucion_por_dimension(df, "bloque_comercial", region_opt_oper)

    sector_strat_share = distribucion_por_dimension(df, "sector", region_opt_strat)
    sector_oper_share  = distribucion_por_dimension(df, "sector", region_opt_oper)

    pais_strat_share = distribucion_por_dimension(df, "destino", region_opt_strat)
    pais_oper_share  = distribucion_por_dimension(df, "destino", region_opt_oper)

    # --------------------------------------------
    # Vías históricas (SUNAT) usando df_vias parquet
    # --------------------------------------------
    vias_filtrado = df_vias[
        (df_vias["anio"] >= anio_desde) & (df_vias["anio"] <= anio_hasta)
    ].copy()

    via_hist = vias_filtrado.groupby("via_transporte")["valor_millones"].sum()
    via_hist_share = via_hist / via_hist.sum()

    return (
        df,
        comparacion_regiones,
        op_region,
        bloque_hist_share,
        bloque_strat_share,
        bloque_oper_share,
        sector_hist_share,
        sector_strat_share,
        sector_oper_share,
        pais_hist_share,
        pais_strat_share,
        pais_oper_share,
        via_hist_share,
    )


# ==========================
# UI STREAMLIT
# ==========================
st.set_page_config(page_title="Modelo Prescriptivo Comercio Exterior", layout="wide")

st.title("Modelo Prescriptivo de Comercio Exterior del Perú")
st.caption("Basado en modelo constelación SUNAT–MINCETUR, con visión estratégica y operativa (toneladas y millones).")

# Cargamos datos desde Parquet
df_presc, df_vias = cargar_datos(BASE_PATH)

# ---- Filtros globales ----
anios = sorted(df_presc["anio"].unique())
anio_min, anio_max = int(anios[0]), int(anios[-1])

anio_desde, anio_hasta = st.sidebar.slider(
    "Rango de años a analizar",
    min_value=anio_min,
    max_value=anio_max,
    value=(anio_min, anio_max),
    step=1,
)

vista = st.sidebar.selectbox(
    "Elige vista",
    [
        "Resumen general",
        "Regiones",
        "Bloques comerciales",
        "Sectores",
        "Países destino",
        "Vías de transporte (histórico)",
    ],
)

top_n = st.sidebar.slider("Top N categorías a mostrar", 5, 30, 10)

(
    df_filtrado,
    comparacion_regiones,
    op_region,
    bloque_hist_share,
    bloque_strat_share,
    bloque_oper_share,
    sector_hist_share,
    sector_strat_share,
    sector_oper_share,
    pais_hist_share,
    pais_strat_share,
    pais_oper_share,
    via_hist_share,
) = construir_modelos(df_presc, df_vias, anio_desde, anio_hasta)


# =============== VISTAS ===============

if vista == "Resumen general":
    st.subheader("Resumen del modelo")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("N° regiones", comparacion_regiones["region_mincetur"].nunique())
    with col2:
        st.metric("Años analizados", df_filtrado["anio"].nunique())
    with col3:
        st.metric("Países destino", df_filtrado["destino"].nunique())

    st.markdown("### Top regiones según modelos histórico, estratégico y operativo")

    reg_full = comparacion_regiones.merge(
        op_region[
            [
                "region_mincetur",
                "toneladas_optimas",
                "valor_millones_estimado",
                "participacion_ton_opt",
                "participacion_valor_opt",
            ]
        ],
        on="region_mincetur",
        how="left",
    )

    reg_full = reg_full.sort_values("participacion_optima", ascending=False).head(top_n)

    chart_data = reg_full.set_index("region_mincetur")[
        ["participacion_historica", "participacion_optima", "participacion_valor_opt"]
    ]
    st.bar_chart(chart_data)

    reg_full_fmt = formatear_tabla_regiones(
        reg_full[
            [
                "region_mincetur",
                "valor_total",
                "valor_por_ton",
                "participacion_historica",
                "participacion_optima",
                "participacion_ton_opt",
                "participacion_valor_opt",
            ]
        ]
    )
    st.dataframe(reg_full_fmt)


elif vista == "Regiones":
    st.subheader("Regiones: histórico vs prescriptivo (estratégico y operativo)")

    reg_full = comparacion_regiones.merge(
        op_region[
            [
                "region_mincetur",
                "toneladas_optimas",
                "valor_millones_estimado",
                "participacion_ton_opt",
                "participacion_valor_opt",
            ]
        ],
        on="region_mincetur",
        how="left",
    )

    reg_full = reg_full.sort_values("participacion_optima", ascending=False)

    regiones_disp = reg_full["region_mincetur"].tolist()
    regiones_sel = st.multiselect(
        "Filtrar regiones",
        regiones_disp,
        default=regiones_disp[:min(top_n, len(regiones_disp))],
    )

    if regiones_sel:
        reg_view = reg_full[reg_full["region_mincetur"].isin(regiones_sel)]
    else:
        reg_view = reg_full

    chart_data = reg_view.set_index("region_mincetur")[
        ["participacion_historica", "participacion_optima", "participacion_valor_opt"]
    ]
    st.bar_chart(chart_data)

    reg_view_fmt = formatear_tabla_regiones(
        reg_view[
            [
                "region_mincetur",
                "valor_total",
                "valor_por_ton",
                "participacion_historica",
                "participacion_optima",
                "toneladas_optimas",
                "valor_millones_estimado",
                "participacion_ton_opt",
                "participacion_valor_opt",
            ]
        ]
    )
    st.dataframe(reg_view_fmt)


elif vista == "Bloques comerciales":
    st.subheader("Bloques comerciales: histórico vs prescriptivo (estratégico y operativo)")

    bloques = list(bloque_hist_share.index)
    df_b = pd.DataFrame(
        {
            "bloque_comercial": bloques,
            "historico": bloque_hist_share.values,
            "estrategico": bloque_strat_share.reindex(bloques).fillna(0).values,
            "operativo": bloque_oper_share.reindex(bloques).fillna(0).values,
        }
    ).sort_values("estrategico", ascending=False)

    df_b_top = df_b.head(top_n)

    st.bar_chart(df_b_top.set_index("bloque_comercial")[["historico", "estrategico", "operativo"]])

    df_b_fmt = formatear_tabla_porcentajes(df_b_top, ["historico", "estrategico", "operativo"])
    st.dataframe(df_b_fmt)


elif vista == "Sectores":
    st.subheader("Sectores: histórico vs prescriptivo (estratégico y operativo)")

    sectores = list(sector_hist_share.index)
    df_s = pd.DataFrame(
        {
            "sector": sectores,
            "historico": sector_hist_share.values,
            "estrategico": sector_strat_share.reindex(sectores).fillna(0).values,
            "operativo": sector_oper_share.reindex(sectores).fillna(0).values,
        }
    ).sort_values("estrategico", ascending=False)

    df_s_top = df_s.head(top_n)

    st.bar_chart(df_s_top.set_index("sector")[["historico", "estrategico", "operativo"]])

    df_s_fmt = formatear_tabla_porcentajes(df_s_top, ["historico", "estrategico", "operativo"])
    st.dataframe(df_s_fmt)


elif vista == "Países destino":
    st.subheader("Países destino: histórico vs prescriptivo (estratégico y operativo)")

    paises = list(pais_hist_share.index)
    df_p = pd.DataFrame(
        {
            "destino": paises,
            "historico": pais_hist_share.values,
            "estrategico": pais_strat_share.reindex(paises).fillna(0).values,
            "operativo": pais_oper_share.reindex(paises).fillna(0).values,
        }
    ).sort_values("estrategico", ascending=False)

    df_p_top = df_p.head(top_n)

    st.bar_chart(df_p_top.set_index("destino")[["historico", "estrategico", "operativo"]])

    df_p_fmt = formatear_tabla_porcentajes(df_p_top, ["historico", "estrategico", "operativo"])
    st.dataframe(df_p_fmt)


elif vista == "Vías de transporte (histórico)":
    st.subheader("Vías de transporte (histórico – SUNAT)")

    df_v = pd.DataFrame(
        {
            "via_transporte": via_hist_share.index,
            "participacion_historica": via_hist_share.values,
        }
    ).sort_values("participacion_historica", ascending=False)

    st.bar_chart(df_v.set_index("via_transporte"))

    df_v_fmt = formatear_tabla_porcentajes(df_v, ["participacion_historica"])
    st.dataframe(df_v_fmt)
