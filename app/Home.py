from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import streamlit as st

from lib import (
    SERIE_LABEL,
    forecast_figure,
    forecast_horizon,
    load_forecast,
    page_setup,
    render_rec_capacidade,
    require_artifacts,
)

page_setup("Antecipar incidentes")
st.title("01 · Antecipar incidentes")
st.caption("Últimos 7 dias da base e previsão do próximo dia (D+1) e da próxima semana (D+7). P2 e P3 obrigatórias.")

if not require_artifacts():
    st.stop()

fc = load_forecast()

opcoes = {"Total": "total", "P2 — Alta": "p2", "P3 — Média": "p3"}
escolha = st.radio("Série", list(opcoes.keys()), horizontal=True)
serie = opcoes[escolha]
extra = forecast_horizon(fc, serie)

if extra.empty:
    st.warning("Sem horizonte de previsão disponível.")
    st.stop()

d1 = extra.iloc[0]
d7_sum = float(extra["yhat"].sum())
peak = extra.loc[extra["yhat"].idxmax()]

c1, c2, c3 = st.columns(3)
c1.metric("D+1", f"{float(d1['yhat']):.0f}", f"{pd.Timestamp(d1['ds']).date()}", delta_color="off")
c2.metric("D+7 (soma)", f"{d7_sum:.0f}", "próximos 7 dias", delta_color="off")
c3.metric("Pico da semana", f"{float(peak['yhat']):.0f}", f"{pd.Timestamp(peak['ds']).date()}", delta_color="off")

st.subheader("Recomendação inteligente")
render_rec_capacidade(fc)

st.subheader(f"Histórico e previsão · Filtrado por: {SERIE_LABEL[serie]}")
st.plotly_chart(forecast_figure(fc, serie, "Incidentes / dia", show_ci=True), width="stretch")
st.caption("Barras sólidas: histórico · barras hachuradas: previsão · traços verticais: IC.")

st.subheader("Previsão diária (D+1 … D+7)")
st.markdown(
    """
**Legenda da tabela**
- **Data** — dia da previsão (após o último registro da base: 31/12/2025)
- **Previsto** — volume estimado de incidentes naquele dia
- **IC inf.** — limite inferior do intervalo de confiança (cenário mais baixo)
- **IC sup.** — limite superior do intervalo de confiança (cenário mais alto)

O valor real tende a cair entre IC inf. e IC sup.; quanto mais estreita a faixa, maior a confiança naquele dia.
"""
)
tabela = extra[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
tabela["ds"] = pd.to_datetime(tabela["ds"]).dt.date
tabela = tabela.rename(
    columns={"ds": "Data", "yhat": "Previsto", "yhat_lower": "IC inf.", "yhat_upper": "IC sup."}
)
tabela["Previsto"] = tabela["Previsto"].round(1)
tabela["IC inf."] = tabela["IC inf."].round(1)
tabela["IC sup."] = tabela["IC sup."].round(1)
st.dataframe(tabela, hide_index=True, width="stretch")
st.caption("Horizonte de 7 dias · IC = faixa de incerteza em torno do Previsto.")
