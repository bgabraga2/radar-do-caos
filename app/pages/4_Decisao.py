from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib import (
    PLOT_LAYOUT,
    action_card,
    forecast_horizon,
    in_range,
    load_clusters,
    load_curated,
    load_forecast,
    load_importance,
    last_7_days,
    page_setup,
    require_artifacts,
)

page_setup("Decisão operacional")
st.title("04 · Apoiar decisão operacional")
st.caption("Onde agir de forma preventiva: picos previstos, agrupamentos críticos, causas recorrentes e fatores de risco.")

if not require_artifacts():
    st.stop()

inicio, fim = last_7_days()
df = in_range(load_curated(), "aberto", inicio, fim)
fc = load_forecast()
clusters = load_clusters()
importance = load_importance()

kpi = df[df["eh_kpi"]].copy()

extra_total = forecast_horizon(fc, "total")
if extra_total.empty:
    action_card("Previsão indisponível", "Horizonte D+1…D+7 ainda não disponível.")
else:
    peak = extra_total.loc[extra_total["yhat"].idxmax()]
    hist_mean = float(fc[(fc["serie"] == "total") & fc["y"].notna()]["y"].tail(90).mean())
    peak_val = float(peak["yhat"])
    if peak_val > hist_mean:
        acao = "Antecipar escala no plantão nesse dia."
    else:
        acao = "Volume previsto abaixo da média recente — sem reforço extraordinário."
    action_card(
        "Capacidade — reforçar operação",
        f"Pico previsto em <strong>{pd.Timestamp(peak['ds']).date()}</strong> "
        f"(~{peak_val:.0f} incidentes, média recente {hist_mean:.0f}/dia). {acao}",
    )

ola = kpi[kpi["kpi_violado_flag"]]
if not ola.empty:
    top_grupo = ola["grupo_designado"].value_counts().idxmax()
    n_grupo = int(ola["grupo_designado"].value_counts().iloc[0])
    action_card(
        "Risco operacional — time com mais perda de OLA",
        f"<strong>{top_grupo}</strong> concentra {n_grupo} violações de KPI. "
        "Revisar fila, plantão e handoff desse grupo.",
    )

recorrentes = (
    df[df["template"].astype(str).str.len() >= 8]
    .groupby(["item_config", "template"])
    .size()
    .reset_index(name="n")
    .sort_values("n", ascending=False)
)
if not recorrentes.empty:
    r0 = recorrentes.iloc[0]
    ic = r0["item_config"] if pd.notna(r0["item_config"]) else "IC sem cadastro"
    action_card(
        "Incidente recorrente — abrir problema",
        f"O template <em>{r0['template'][:80]}</em> no item <strong>{ic}</strong> "
        f"repetiu {int(r0['n'])} vezes. Candidato a problema / ajuste de monitoramento.",
    )

st.subheader("Agrupamentos críticos (produto + categoria + prioridade)")
grp = (
    kpi[kpi["produto"].notna() & kpi["categoria"].notna()]
    .groupby(["produto", "categoria", "prioridade"], dropna=False)
    .agg(n=("numero", "size"), ola=("kpi_violado_flag", "sum"))
    .reset_index()
)
if grp.empty:
    st.info("Sem produto/categoria preenchidos no universo KPI.")
else:
    grp["taxa_ola_%"] = (grp["ola"] / grp["n"] * 100).round(2)
    grp = grp.sort_values(["ola", "n"], ascending=False).head(15)
    grp = grp.rename(
        columns={
            "produto": "Produto",
            "categoria": "Categoria",
            "prioridade": "Prioridade",
            "n": "Incidentes",
            "ola": "OLA perdidos",
            "taxa_ola_%": "Taxa OLA %",
        }
    )
    st.dataframe(grp, hide_index=True, width="stretch")
    st.caption("Só registros com produto e categoria preenchidos (~37% da base).")

st.subheader("Causas recorrentes (clusters de descrição)")
view = clusters.copy()
view = view.rename(
    columns={
        "cluster": "Cluster",
        "n": "Incidentes",
        "n_templates": "Templates",
        "termos": "Termos",
        "pct_ola": "OLA %",
        "top_grupo": "Grupo",
        "top_produto": "Produto",
        "top_template": "Template mais frequente",
    }
)
st.dataframe(view, hide_index=True, width="stretch")

st.subheader("Quais fatores mais influenciam a perda de OLA?")
fig = go.Figure(
    go.Bar(
        x=importance["importance"],
        y=importance["label"],
        orientation="h",
        marker_color="#22d3ee",
    )
)
fig.update_layout(**PLOT_LAYOUT, height=280, xaxis_title="Importância", yaxis=dict(autorange="reversed"))
st.plotly_chart(fig, width="stretch")

st.subheader("Onde o volume sobe")
c1, c2 = st.columns(2)
p23 = df[df["prioridade_num"].isin({2, 3})]
if p23.empty:
    st.info("Sem incidentes P2/P3 neste período.")
    st.stop()
with c1:
    por_hora = p23.groupby("hora").size().reset_index(name="n")
    pico_h = int(por_hora.loc[por_hora["n"].idxmax(), "hora"])
    st.markdown(f"Horário de maior abertura P2/P3: **{pico_h:02d}h**.")
    fig_h = go.Figure(go.Bar(x=por_hora["hora"], y=por_hora["n"], marker_color="#ff2e93"))
    fig_h.update_layout(**PLOT_LAYOUT, height=220, xaxis_title="Hora", yaxis_title="Incidentes")
    st.plotly_chart(fig_h, width="stretch")
with c2:
    por_grupo = p23["grupo_designado"].value_counts().head(8)
    st.markdown(f"Grupo com mais volume P2/P3: **{por_grupo.index[0]}** ({int(por_grupo.iloc[0])}).")
    fig_g = go.Figure(go.Bar(x=por_grupo.values, y=por_grupo.index, orientation="h", marker_color="#22d3ee"))
    fig_g.update_layout(**PLOT_LAYOUT, height=220, xaxis_title="Incidentes", yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_g, width="stretch")
