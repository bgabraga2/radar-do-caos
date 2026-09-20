from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
LOGO = ROOT / "logo.png"
LOGO_SIDEBAR = ROOT / "logo_sidebar.png"
FAVICON = ROOT / "favicon.png"
CURATED = ROOT / "data" / "curated.parquet"
DAILY = ROOT / "outputs" / "daily.parquet"
FORECAST = ROOT / "outputs" / "forecast.parquet"
RISK = ROOT / "outputs" / "risk.parquet"
IMPORTANCE = ROOT / "outputs" / "feature_importance.parquet"
CLUSTERS = ROOT / "outputs" / "clusters.parquet"
METRICS = ROOT / "outputs" / "metrics.json"

DOW_PT = {0: "Seg", 1: "Ter", 2: "Qua", 3: "Qui", 4: "Sex", 5: "Sáb", 6: "Dom"}
SERIE_LABEL = {
    "total": "Total",
    "p2": "P2 — Alta",
    "p3": "P3 — Média",
    "ola_p2": "Perda de OLA · P2",
    "ola_p3": "Perda de OLA · P3",
}

PLOT_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#0a0e1a",
    plot_bgcolor="#111827",
    legend=dict(orientation="h"),
    margin=dict(l=40, r=20, t=20, b=40),
)

CSS = """
<style>
  .stApp { background-color: #0a0e1a; color: #e2e8f0; }
  h1, h2, h3 { color: #e2e8f0 !important; }
  [data-testid="stSidebar"] { background-color: #111827; }
  [data-testid="stSidebarHeader"] {
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    padding: 0.75rem 0.5rem 0.35rem 0.5rem !important;
  }
  [data-testid="stSidebarHeader"] [data-testid="stLogo"] ,
  [data-testid="stSidebar"] a[data-testid="stLogoLink"] ,
  [data-testid="stSidebar"] img[data-testid="stLogo"] {
    height: 4.5rem !important;
    max-width: 180px !important;
    width: auto !important;
    margin: 0 auto !important;
    display: block !important;
  }
  [data-testid="stSidebar"] [data-testid="stLogoSpacer"] {
    display: none !important;
  }
  .metric-chip {
    display: inline-block;
    background: rgba(34, 211, 238, 0.12);
    border: 1px solid rgba(34, 211, 238, 0.35);
    color: #22d3ee;
    padding: 0.25rem 0.6rem;
    border-radius: 4px;
    font-size: 0.8rem;
    margin: 0 0.4rem 0.4rem 0;
  }
  .action-card {
    background: #111827;
    border: 1px solid #ff2e93;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin: 0 0 0.8rem 0;
  }
  .action-card h3 { color: #ff2e93 !important; margin: 0 0 0.4rem 0; font-size: 0.95rem; }
  .action-card p { margin: 0; color: #cbd5e1; font-size: 0.9rem; }
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def page_setup(title: str) -> None:
    logo_path = LOGO_SIDEBAR if LOGO_SIDEBAR.exists() else LOGO
    page_icon = str(FAVICON) if FAVICON.exists() else (str(logo_path) if logo_path.exists() else "📡")
    st.set_page_config(page_title=title, page_icon=page_icon, layout="wide")
    inject_css()
    if logo_path.exists():
        st.logo(str(logo_path), size="large")


def chips(*items: str) -> None:
    html = "".join(f'<span class="metric-chip">{x}</span>' for x in items if x)
    st.markdown(html, unsafe_allow_html=True)


def action_card(title: str, body: str) -> None:
    st.markdown(f"<div class='action-card'><h3>{title}</h3><p>{body}</p></div>", unsafe_allow_html=True)


def render_rec_capacidade(fc: pd.DataFrame) -> None:
    """Recomendação de capacidade / pico — alinhada à tela Antecipar."""
    extra = forecast_horizon(fc, "total")
    if extra.empty:
        action_card("Previsão indisponível", "Horizonte D+1…D+7 ainda não disponível.")
        return
    peak = extra.loc[extra["yhat"].idxmax()]
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


def render_rec_ola_grupo(df: pd.DataFrame, fc: pd.DataFrame | None = None) -> None:
    """Recomendações de OLA — alinhadas à tela Impacto nos KPIs."""
    kpi = df[df["eh_kpi"]].copy()
    ola = kpi[kpi["kpi_violado_flag"]]
    if ola.empty:
        action_card(
            "OLA sob controle",
            "Sem violações de KPI no recorte. Manter monitoramento de fila e plantão.",
        )
        return

    por_grupo = ola["grupo_designado"].value_counts()
    top_grupo = por_grupo.index[0]
    n_grupo = int(por_grupo.iloc[0])
    total_ola = int(len(ola))
    pct_grupo = 100.0 * n_grupo / total_ola

    kpi_g = kpi[kpi["grupo_designado"] == top_grupo]
    taxa_g = 100.0 * n_grupo / len(kpi_g) if len(kpi_g) else 0.0

    ola_g = ola[ola["grupo_designado"] == top_grupo]
    n_p2 = int((ola_g["prioridade_num"] == 2).sum())
    n_p3 = int((ola_g["prioridade_num"] == 3).sum())

    partes = [
        f"<strong>{top_grupo}</strong> concentra <strong>{n_grupo}</strong> de {total_ola} "
        f"violações ({pct_grupo:.0f}% do OLA do período), "
        f"com taxa de {taxa_g:.1f}% nos tickets KPI do próprio time "
        f"({n_grupo} OLA em {len(kpi_g)} tickets).",
        f"Quebra: P2 = {n_p2} · P3 = {n_p3}.",
    ]

    if ola_g["hora"].notna().any():
        pico_h = int(ola_g.groupby("hora").size().idxmax())
        partes.append(f"Horário com mais OLA nesse time: <strong>{pico_h:02d}h</strong>.")

    prod = ola_g["produto"].dropna()
    if not prod.empty:
        top_prod = prod.value_counts().index[0]
        n_prod = int(prod.value_counts().iloc[0])
        partes.append(f"Produto mais afetado: <strong>{top_prod}</strong> ({n_prod} OLA).")

    cat = ola_g["categoria"].dropna()
    if not cat.empty:
        top_cat = cat.value_counts().index[0]
        n_cat = int(cat.value_counts().iloc[0])
        partes.append(f"Categoria mais afetada: <strong>{top_cat}</strong> ({n_cat} OLA).")

    if len(por_grupo) >= 2:
        segundo = por_grupo.index[1]
        n_seg = int(por_grupo.iloc[1])
        partes.append(f"Segundo foco: {segundo} ({n_seg} OLA).")

    partes.append(
        "Ação: revisar fila e handoff desse grupo no horário de pico; "
        "priorizar P2 e o produto/categoria acima antes de redistribuir carga."
    )
    action_card("Risco operacional — time com mais perda de OLA", " ".join(partes))

    if fc is None:
        return
    fut_p2 = forecast_horizon(fc, "ola_p2")
    fut_p3 = forecast_horizon(fc, "ola_p3")
    if fut_p2.empty or fut_p3.empty:
        return

    d1 = float(fut_p2.iloc[0]["yhat"]) + float(fut_p3.iloc[0]["yhat"])
    d7 = float(fut_p2["yhat"].sum() + fut_p3["yhat"].sum())
    hist_ola = fc[fc["serie"].isin(["ola_p2", "ola_p3"]) & fc["y"].notna()]
    media_dia = float(hist_ola.groupby("ds")["y"].sum().tail(14).mean()) if not hist_ola.empty else 0.0

    if media_dia > 0 and d1 > media_dia * 1.15:
        pressao = (
            f"D+1 previsto (~{d1:.1f}) fica <strong>acima</strong> da média recente "
            f"({media_dia:.1f}/dia). Antecipar cobertura em {top_grupo}."
        )
    elif media_dia > 0:
        pressao = (
            f"D+1 previsto (~{d1:.1f}) está alinhado ou abaixo da média recente "
            f"({media_dia:.1f}/dia). Manter escala atual e vigiar {top_grupo}."
        )
    else:
        pressao = f"D+1 previsto ~{d1:.1f} violações (P2+P3)."

    action_card(
        "Pressão de OLA — próximos dias",
        f"{pressao} Soma D+1…D+7: ~{d7:.0f} violações estimadas. "
        "Usar o pico de volume da semana para reforçar plantão nos dias mais carregados.",
    )


def render_rec_recorrente(df: pd.DataFrame) -> None:
    """Recomendação de incidente recorrente — alinhada à tela Tendências."""
    recorrentes = (
        df[df["template"].astype(str).str.len() >= 8]
        .groupby(["item_config", "template"])
        .size()
        .reset_index(name="n")
        .sort_values("n", ascending=False)
    )
    if recorrentes.empty:
        return
    r0 = recorrentes.iloc[0]
    ic = r0["item_config"] if pd.notna(r0["item_config"]) else "IC sem cadastro"
    action_card(
        "Incidente recorrente — abrir problema",
        f"O template <em>{r0['template'][:80]}</em> no item <strong>{ic}</strong> "
        f"repetiu {int(r0['n'])} vezes. Candidato a problema / ajuste de monitoramento.",
    )


def render_rec_sazonalidade(df: pd.DataFrame) -> None:
    """Recomendação de escala por dia/hora — alinhada ao bloco de sazonalidade."""
    base = df[df["prioridade_num"].isin({2, 3})]
    if base.empty:
        return
    dow = base.groupby("dow").size()
    hora = base.groupby("hora").size()
    if dow.empty or hora.empty:
        return
    pico_dow = int(dow.idxmax())
    n_dow = int(dow.max())
    pico_h = int(hora.idxmax())
    n_h = int(hora.max())
    pct_h = 100.0 * n_h / len(base)
    action_card(
        "Sazonalidade — reforçar plantão no pico",
        f"Maior volume P2/P3 em <strong>{DOW_PT[pico_dow]}</strong> ({n_dow} aberturas) "
        f"e no horário <strong>{pico_h:02d}h</strong> ({n_h} aberturas, {pct_h:.0f}% do período). "
        "Antecipar cobertura nesses slots e revisar handoff pré-pico.",
    )


def load_metrics() -> dict:
    if not METRICS.exists():
        return {}
    return json.loads(METRICS.read_text(encoding="utf-8"))


def require_artifacts() -> bool:
    needed = (CURATED, DAILY, FORECAST, RISK, IMPORTANCE, CLUSTERS)
    missing = [p.name for p in needed if not p.exists()]
    if missing:
        st.error("Artefatos ausentes: **" + ", ".join(missing) + "**.")
        return False
    return True


@st.cache_data
def load_curated() -> pd.DataFrame:
    df = pd.read_parquet(CURATED)
    df["aberto"] = pd.to_datetime(df["aberto"])
    df["data"] = pd.to_datetime(df["data"])
    return df


@st.cache_data
def load_daily() -> pd.DataFrame:
    d = pd.read_parquet(DAILY)
    d["ds"] = pd.to_datetime(d["ds"])
    return d


@st.cache_data
def load_forecast() -> pd.DataFrame:
    fc = pd.read_parquet(FORECAST)
    fc["ds"] = pd.to_datetime(fc["ds"])
    return fc


@st.cache_data
def load_risk() -> pd.DataFrame:
    sc = pd.read_parquet(RISK)
    sc["aberto"] = pd.to_datetime(sc["aberto"])
    return sc


@st.cache_data
def load_importance() -> pd.DataFrame:
    return pd.read_parquet(IMPORTANCE)


@st.cache_data
def load_clusters() -> pd.DataFrame:
    return pd.read_parquet(CLUSTERS)


def hist_bounds() -> tuple[pd.Timestamp, pd.Timestamp]:
    daily = load_daily()
    hist = daily[daily["serie"] == "total"]
    return pd.Timestamp(hist["ds"].min()).normalize(), pd.Timestamp(hist["ds"].max()).normalize()


def last_7_days() -> tuple[pd.Timestamp, pd.Timestamp]:
    """Últimos 7 dias da base (inclusive)."""
    hist_min, hist_max = hist_bounds()
    start = max(hist_min, hist_max - pd.Timedelta(days=6))
    return start, hist_max


def in_range(df: pd.DataFrame, col: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    ts = pd.to_datetime(df[col])
    return df[(ts >= start) & (ts < end + pd.Timedelta(days=1))].copy()


def forecast_horizon(fc: pd.DataFrame, serie: str) -> pd.DataFrame:
    extra = fc[(fc["serie"] == serie) & fc["is_forecast"]].sort_values("ds")
    extra = extra.head(7).copy()
    if not extra.empty:
        extra["yhat"] = extra["yhat"].clip(lower=0)
        extra["yhat_lower"] = extra["yhat_lower"].clip(lower=0)
        extra["yhat_upper"] = extra["yhat_upper"].clip(lower=0)
    return extra


def history_last_7(fc: pd.DataFrame, serie: str) -> pd.DataFrame:
    start, end = last_7_days()
    hist = fc[(fc["serie"] == serie) & fc["y"].notna()].sort_values("ds")
    return hist[(hist["ds"] >= start) & (hist["ds"] <= end)]


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return hex_color
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def forecast_bar_marker(color: str) -> dict:
    """Marcador de barra prevista: fill hachurado + borda reforçada."""
    return dict(
        color=_hex_to_rgba(color, 0.2),
        line=dict(color=color, width=2.5),
        pattern=dict(
            shape="/",
            fgcolor=color,
            bgcolor=_hex_to_rgba(color, 0.1),
            size=7,
            solidity=0.45,
        ),
    )


def add_forecast_trace(fig: go.Figure, fc: pd.DataFrame, serie: str, name: str, color: str) -> None:
    fut = forecast_horizon(fc, serie)
    if fut.empty:
        return
    fig.add_trace(go.Bar(x=fut["ds"], y=fut["yhat"], name=name, marker=forecast_bar_marker(color)))


def forecast_figure(fc: pd.DataFrame, serie: str, y_title: str, *, show_ci: bool = False) -> go.Figure:
    hist = history_last_7(fc, serie)
    fut = forecast_horizon(fc, serie)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=hist["ds"], y=hist["y"], name="Últimos 7 dias", marker_color="#22d3ee"))
    if not fut.empty:
        bar_kwargs: dict = dict(
            x=fut["ds"],
            y=fut["yhat"],
            name="Previsão 7 dias",
            marker=forecast_bar_marker("#ff2e93"),
        )
        if show_ci:
            bar_kwargs["error_y"] = dict(
                type="data",
                symmetric=False,
                array=(fut["yhat_upper"] - fut["yhat"]).clip(lower=0),
                arrayminus=(fut["yhat"] - fut["yhat_lower"]).clip(lower=0),
                color="rgba(226, 232, 240, 0.85)",
                thickness=1.5,
                width=5,
            )
        fig.add_trace(go.Bar(**bar_kwargs))
    fig.update_layout(
        **PLOT_LAYOUT,
        height=380,
        yaxis_title=y_title,
        xaxis_title="Data",
        barmode="group",
        bargap=0.15,
    )
    return fig
