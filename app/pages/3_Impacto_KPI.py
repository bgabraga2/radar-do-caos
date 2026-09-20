from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib import (
    PLOT_LAYOUT,
    forecast_figure,
    forecast_horizon,
    add_forecast_trace,
    in_range,
    last_7_days,
    load_curated,
    load_daily,
    load_forecast,
    page_setup,
    render_rec_ola_grupo,
    require_artifacts,
)

page_setup("Impacto nos KPIs")
st.title("03 · Projetar impacto nos KPIs")
st.caption("Últimos 7 dias e previsão D+1…D+7: volume de incidentes e perda de OLA (P2 e P3).")

if not require_artifacts():
    st.stop()

inicio, fim = last_7_days()
daily = in_range(load_daily(), "ds", inicio, fim)
df = in_range(load_curated(), "aberto", inicio, fim)
fc = load_forecast()

vol = daily[daily["serie"] == "total"]
ola_p2 = daily[daily["serie"] == "ola_p2"]
ola_p3 = daily[daily["serie"] == "ola_p3"]
ola_last7 = float(ola_p2.tail(7)["y"].sum() + ola_p3.tail(7)["y"].sum())
vol_last7 = float(vol.tail(7)["y"].mean()) if len(vol) else 0

d1_total = forecast_horizon(fc, "total")
d1_ola = forecast_horizon(fc, "ola_p2")
d1_ola3 = forecast_horizon(fc, "ola_p3")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Média volume 7d", f"{vol_last7:.0f}")
c2.metric("OLA perdidos 7d", f"{ola_last7:.0f}")
c3.metric(
    "OLA previsto D+1 (P2+P3)",
    f"{(float(d1_ola.iloc[0]['yhat']) + float(d1_ola3.iloc[0]['yhat'])):.1f}" if not d1_ola.empty and not d1_ola3.empty else "–",
)
c4.metric(
    "Volume previsto D+1",
    f"{float(d1_total.iloc[0]['yhat']):.0f}" if not d1_total.empty else "–",
)

# Recorte de 30 dias para a recomendação ter massa estatística (gráficos seguem em 7d).
df_rec = in_range(load_curated(), "aberto", fim - pd.Timedelta(days=29), fim)

st.subheader("Recomendação inteligente")
st.caption("Baseada nos últimos 30 dias de OLA + previsão D+1…D+7.")
render_rec_ola_grupo(df_rec, fc)

st.subheader("Volume diário de incidentes")
if vol.empty:
    st.info("Sem dados nos últimos 7 dias.")
else:
    fig_v = go.Figure()
    fig_v.add_trace(go.Bar(x=vol["ds"], y=vol["y"], name="Últimos 7 dias", marker_color="#22d3ee"))
    add_forecast_trace(fig_v, fc, "total", "Previsão 7 dias", "#ff2e93")
    fig_v.update_layout(
        **PLOT_LAYOUT,
        height=320,
        yaxis_title="Incidentes / dia",
        xaxis_title="Data",
        barmode="group",
        bargap=0.15,
    )
    st.plotly_chart(fig_v, width="stretch")
    st.caption("Barras sólidas: histórico · barras hachuradas: previsão.")

st.subheader("Perda de OLA por dia")
if ola_p2.empty and ola_p3.empty:
    st.info("Sem perda de OLA neste período.")
else:
    fig_o = go.Figure()
    fig_o.add_trace(go.Bar(x=ola_p2["ds"], y=ola_p2["y"], name="OLA P2", marker_color="#ff2e93"))
    fig_o.add_trace(go.Bar(x=ola_p3["ds"], y=ola_p3["y"], name="OLA P3", marker_color="#f97316"))
    add_forecast_trace(fig_o, fc, "ola_p2", "OLA P2 previsto", "#ff2e93")
    add_forecast_trace(fig_o, fc, "ola_p3", "OLA P3 previsto", "#f97316")
    fig_o.update_layout(
        **PLOT_LAYOUT,
        height=320,
        yaxis_title="Violações / dia",
        xaxis_title="Data",
        barmode="group",
        bargap=0.15,
    )
    st.plotly_chart(fig_o, width="stretch")
    st.caption("Barras sólidas: histórico · barras hachuradas: previsão.")

st.subheader("Previsão de perda de OLA (D+1 … D+7)")
aba = st.radio("Série", ["P2 — Alta", "P3 — Média"], horizontal=True)
serie = "ola_p2" if aba.startswith("P2") else "ola_p3"
st.plotly_chart(forecast_figure(fc, serie, "Violações / dia"), width="stretch")

extra = forecast_horizon(fc, serie)
if not extra.empty:
    st.markdown(
        """
**Legenda:** **Previsto** = violações de OLA estimadas no dia ·
**IC inf./sup.** = faixa de incerteza (cenário baixo / alto).
"""
    )
    tab = extra[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    tab["ds"] = pd.to_datetime(tab["ds"]).dt.date
    tab = tab.rename(columns={"ds": "Data", "yhat": "Previsto", "yhat_lower": "IC inf.", "yhat_upper": "IC sup."})
    for c in ("Previsto", "IC inf.", "IC sup."):
        tab[c] = tab[c].round(2)
    st.dataframe(tab, hide_index=True, width="stretch")
