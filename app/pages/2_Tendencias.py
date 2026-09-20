from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import plotly.graph_objects as go
import streamlit as st

from lib import (
    DOW_PT,
    PLOT_LAYOUT,
    add_forecast_trace,
    in_range,
    last_7_days,
    load_curated,
    load_daily,
    load_forecast,
    page_setup,
    render_rec_recorrente,
    render_rec_sazonalidade,
    require_artifacts,
)

page_setup("Tendências")
st.title("02 · Identificar tendências")
st.caption("Últimos 7 dias da base e previsão D+1…D+7. P2 e P3 obrigatórias; quebra por categoria, produto ou IC.")

if not require_artifacts():
    st.stop()

inicio, fim = last_7_days()
df = in_range(load_curated(), "aberto", inicio, fim)
daily = in_range(load_daily(), "ds", inicio, fim)
fc = load_forecast()

st.subheader("Recomendação inteligente")
render_rec_recorrente(df)

st.subheader("Volume diário por prioridade")
if daily.empty:
    st.info("Sem dados nos últimos 7 dias.")
else:
    fig = go.Figure()
    for serie, cor, nome in (("p2", "#ff2e93", "P2 — Alta"), ("p3", "#22d3ee", "P3 — Média")):
        s = daily[daily["serie"] == serie].sort_values("ds")
        fig.add_trace(go.Bar(x=s["ds"], y=s["y"], name=nome, marker_color=cor))
        add_forecast_trace(fig, fc, serie, f"{nome} previsto", cor)
    fig.update_layout(
        **PLOT_LAYOUT,
        height=340,
        yaxis_title="Incidentes / dia",
        xaxis_title="Data",
        barmode="group",
        bargap=0.15,
    )
    st.plotly_chart(fig, width="stretch")
    st.caption("Barras sólidas: últimos 7 dias · barras hachuradas: previsão D+1…D+7.")

st.subheader("Quebra por categoria, produto ou IC")
dim_map = {"Categoria": "categoria", "Produto": "produto", "Item de configuração": "item_config"}
dim_label = st.selectbox("Dimensão", list(dim_map.keys()))
col = dim_map[dim_label]
prios = st.multiselect("Prioridade", ["2 Alta", "3 Média"], default=["2 Alta", "3 Média"])
prio_nums = {int(p[0]) for p in prios} or {2, 3}

sub = df[df["prioridade_num"].isin(prio_nums) & df[col].notna()].copy()
if sub.empty:
    st.info("Sem registros preenchidos nesta combinação.")
else:
    top = sub[col].value_counts().head(8)
    escolha = st.selectbox(f"Top {dim_label.lower()}", top.index.tolist())
    recorte = sub[sub[col] == escolha]
    serie_dim = recorte.set_index("aberto").resample("D").size().rename("y").reset_index()
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=serie_dim["aberto"], y=serie_dim["y"], name=escolha, marker_color="#22d3ee"))
    fig2.update_layout(**PLOT_LAYOUT, height=300, yaxis_title="Incidentes / dia", xaxis_title="Data", bargap=0.15)
    st.plotly_chart(fig2, width="stretch")
    st.caption(f"{len(recorte):,} incidentes P2/P3 com {dim_label.lower()} preenchido.".replace(",", "."))

st.subheader("Sazonalidade")
base = df[df["prioridade_num"].isin({2, 3})]
if base.empty:
    st.info("Sem incidentes P2/P3 nos últimos 7 dias.")
    st.stop()

render_rec_sazonalidade(df)

c1, c2 = st.columns(2)
with c1:
    dow = base.groupby("dow").size()
    fig_d = go.Figure(go.Bar(x=[DOW_PT[i] for i in dow.index], y=dow.values, marker_color="#22d3ee"))
    fig_d.update_layout(**PLOT_LAYOUT, height=240, title="Dia da semana", yaxis_title="Incidentes")
    st.plotly_chart(fig_d, width="stretch")
with c2:
    hora = base.groupby("hora").size()
    fig_h = go.Figure(go.Bar(x=hora.index, y=hora.values, marker_color="#ff2e93"))
    fig_h.update_layout(**PLOT_LAYOUT, height=240, title="Horário", yaxis_title="Incidentes")
    st.plotly_chart(fig_h, width="stretch")
