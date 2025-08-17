# app.py (v8) — EMA9/EMA21 + Confirmação por Volume (EMA50) + Retorno por ciclo até o próximo cruzamento
# Requisitos: pip install streamlit yfinance plotly pandas python-dateutil
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# ---------- Utils ----------
def normalize_ticker_b3(t: str) -> str:
    t = (t or "").strip().upper()
    if not t:
        return ""
    if t.startswith("^") or t.endswith(".SA"):
        return t
    return f"{t}.SA"

def fetch_last_12m(ticker_raw: str) -> pd.DataFrame:
    today = datetime.utcnow().date()
    start = today - relativedelta(years=1)
    end = today + timedelta(days=1)

    ticker = normalize_ticker_b3(ticker_raw)
    if not ticker:
        return pd.DataFrame()

    df = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        interval="1d",
        group_by="column",
    )

    if df.empty:
        return df

    # Achatar MultiIndex se necessário
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df.columns = df.columns.get_level_values(0)
        except Exception:
            df = df.droplevel(-1, axis=1)

    # Padronizar colunas e dtypes
    df = df.rename(columns=lambda c: str(c).title())
    cols = ["Open", "High", "Low", "Close", "Volume"]
    df = df[[c for c in cols if c in df.columns]]
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = pd.to_numeric(out["Close"], errors="coerce")
    vol = pd.to_numeric(out["Volume"], errors="coerce")
    out["EMA9"] = close.ewm(span=9, adjust=False).mean()
    out["EMA21"] = close.ewm(span=21, adjust=False).mean()
    out["VolEMA50"] = vol.ewm(span=50, adjust=False).mean()

    # Cruzamentos
    prev_up = out["EMA9"].shift(1) <= out["EMA21"].shift(1)
    prev_dn = out["EMA9"].shift(1) >= out["EMA21"].shift(1)
    out["BullCross"] = (out["EMA9"] > out["EMA21"]) & prev_up
    out["BearCross"] = (out["EMA9"] < out["EMA21"]) & prev_dn

    # Confirmação por volume: "tocando ou acima" => >=
    out["VolConfirm"] = out["Volume"] >= out["VolEMA50"]

    # Sinais + confirmação
    out["BullConfirmed"] = out["BullCross"] & out["VolConfirm"]
    out["BearConfirmed"] = out["BearCross"] & out["VolConfirm"]
    return out

def compute_confirmation_rates(df: pd.DataFrame):
    total_bull = int(df["BullCross"].sum())
    total_bear = int(df["BearCross"].sum())
    conf_bull = int(df["BullConfirmed"].sum())
    conf_bear = int(df["BearConfirmed"].sum())
    total_cross = total_bull + total_bear
    total_conf = conf_bull + conf_bear

    rate_bull = round(100 * conf_bull / total_bull, 2) if total_bull > 0 else None
    rate_bear = round(100 * conf_bear / total_bear, 2) if total_bear > 0 else None
    rate_total = round(100 * total_conf / total_cross, 2) if total_cross > 0 else None

    return {
        "total_bull": total_bull,
        "total_bear": total_bear,
        "conf_bull": conf_bull,
        "conf_bear": conf_bear,
        "rate_bull_%": rate_bull,
        "rate_bear_%": rate_bear,
        "rate_total_%": rate_total,
        "total_cross": total_cross,
        "total_conf": total_conf
    }

def compute_confirmed_cycles_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constrói ciclos a partir de cruzamentos CONFIRMADOS (Volume >= EMA50 no candle do cruzamento)
    e encerra no PRÓXIMO cruzamento (não exige confirmação na saída).
    Retorno do ciclo:
      - ALTA (bull): (Close_saida / Close_entrada - 1) * 100
      - BAIXA (bear): (Close_entrada / Close_saida - 1) * 100
    """
    events = []
    for ts, row in df.iterrows():
        if bool(row["BullConfirmed"]):
            events.append({"date": ts, "type": "BULL"})
        elif bool(row["BearConfirmed"]):
            events.append({"date": ts, "type": "BEAR"})

    if len(events) == 0:
        return pd.DataFrame()

    # Encontrar todos os cruzamentos (confirmados ou não) para determinar saídas
    cross_mask = df["BullCross"] | df["BearCross"]
    cross_dates = df.index[cross_mask]

    cycles = []
    for i in range(len(events)):
        entry = events[i]
        # próxima data de cruzamento após a entrada
        later_crosses = [d for d in cross_dates if d > entry["date"]]
        if len(later_crosses) == 0:
            # não há cruzamento seguinte — ciclo incompleto
            continue
        exit_dt = later_crosses[0]

        entry_close = float(df.loc[entry["date"], "Close"])
        exit_close = float(df.loc[exit_dt, "Close"])
        direction = "ALTA" if entry["type"] == "BULL" else "BAIXA"

        if direction == "ALTA":
            ret = (exit_close / entry_close - 1.0) * 100.0
        else:
            ret = (entry_close / exit_close - 1.0) * 100.0

        # duração em barras
        try:
            dur = int(df.index.get_loc(exit_dt) - df.index.get_loc(entry["date"]))
        except Exception:
            dur = None

        cycles.append({
            "Entrada": entry["date"],
            "Direcao": direction,
            "Preco_Entrada": round(entry_close, 6),
            "Saida": exit_dt,
            "Preco_Saida": round(exit_close, 6),
            "Barras_No_Ciclo": dur,
            "Retorno_%": round(ret, 2),
        })

    return pd.DataFrame(cycles)

def summarize_cycles(cycles: pd.DataFrame) -> dict:
    if cycles is None or cycles.empty:
        return {"num_cycles": 0}
    n = len(cycles)
    avg = round(cycles["Retorno_%"].mean(), 2)
    med = round(cycles["Retorno_%"].median(), 2)
    long = cycles[cycles["Direcao"] == "ALTA"]
    short = cycles[cycles["Direcao"] == "BAIXA"]
    avg_long = round(long["Retorno_%"].mean(), 2) if not long.empty else None
    avg_short = round(short["Retorno_%"].mean(), 2) if not short.empty else None
    return {
        "num_cycles": n,
        "avg_return_%": avg,
        "median_return_%": med,
        "avg_return_alta_%": avg_long,
        "avg_return_baixa_%": avg_short,
    }

def plot_chart(df: pd.DataFrame, cycles: pd.DataFrame, ticker_in: str):
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
        row_heights=[0.68, 0.32]
    )

    # Candlestick com cores definidas: verde para alta, vermelho para baixa
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="Candles",
        increasing_line_color="green", increasing_fillcolor="green",
        decreasing_line_color="red", decreasing_fillcolor="red"
    ), row=1, col=1)

    # EMAs
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA9"], mode="lines", name="EMA 9"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA21"], mode="lines", name="EMA 21"), row=1, col=1)

    # Marcadores de cruzamentos confirmados
    bulls = df[df["BullConfirmed"]]
    bears = df[df["BearConfirmed"]]
    if not bulls.empty:
        fig.add_trace(go.Scatter(
            x=bulls.index, y=bulls["Close"], mode="markers",
            name="Alta CONFIRMADA", marker_symbol="triangle-up", marker_size=10
        ), row=1, col=1)
    if not bears.empty:
        fig.add_trace(go.Scatter(
            x=bears.index, y=bears["Close"], mode="markers",
            name="Baixa CONFIRMADA", marker_symbol="triangle-down", marker_size=10
        ), row=1, col=1)

    # Volume + EMA50
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["VolEMA50"], mode="lines", name="EMA Vol 50"), row=2, col=1)

    # Faixas dos ciclos confirmados
    if cycles is not None and not cycles.empty:
        for _, tr in cycles.iterrows():
            color = "rgba(46, 204, 113, 0.18)" if tr["Direcao"] == "ALTA" else "rgba(231, 76, 60, 0.18)"
            fig.add_vrect(x0=tr["Entrada"], x1=tr["Saida"], fillcolor=color, line_width=0, row="all", col=1)

    fig.update_layout(
        title=f"{normalize_ticker_b3(ticker_in)} — EMA9/21 + Volume (EMA50) | Ciclos confirmados e retornos",
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        margin=dict(l=20, r=20, t=50, b=20),
        height=800
    )
    fig.update_yaxes(title_text="Preço", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    return fig

# ---------- UI ----------
st.set_page_config(page_title="EMA9/21 + Volume (B3) — Ciclos", layout="wide")
st.title("EMA9/EMA21 + Volume (EMA50) — Ciclos confirmados e retornos (12 meses, diário)")

with st.sidebar:
    st.markdown("### Regras")
    st.write("""
- **Alta**: EMA9 cruza **para cima** EMA21.
- **Baixa**: EMA9 cruza **para baixo** EMA21.
- **Confirmação**: `Volume >= EMA50(Volume)` **no mesmo candle**.
- **Ciclo**: começa no cruzamento **confirmado** e termina no **próximo cruzamento** (não exige confirmação na saída).
    """)
    st.caption("⚠️ Uso educacional. Não é recomendação de investimento.")

ticker_in = st.text_input("Ticker (ex.: ITUB4, PETR4, VALE3)", value="ITUB4")

if st.button("Analisar", type="primary"):
    with st.spinner("Baixando dados e calculando..."):
        df_raw = fetch_last_12m(ticker_in)
        if df_raw.empty:
            st.error("Não encontrei dados para esse ticker no período.")
        else:
            df = add_indicators(df_raw)

            # Métricas de confirmação
            rates = compute_confirmation_rates(df)

            # Ciclos confirmados e retornos
            cycles = compute_confirmed_cycles_returns(df)
            summary = summarize_cycles(cycles)

            # Gráfico
            fig = plot_chart(df, cycles, ticker_in)
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Taxas de confirmação (no candle do cruzamento)")
            col1, col2, col3 = st.columns(3)
            col1.metric("Alta — confirmados/total", f"{rates['conf_bull']}/{rates['total_bull']}" if rates['total_bull']>0 else "0/0")
            col1.metric("Taxa de confirmação (ALTA) %", rates["rate_bull_%"] if rates["rate_bull_%"] is not None else 0.0)
            col2.metric("Baixa — confirmados/total", f"{rates['conf_bear']}/{rates['total_bear']}" if rates['total_bear']>0 else "0/0")
            col2.metric("Taxa de confirmação (BAIXA) %", rates["rate_bear_%"] if rates["rate_bear_%"] is not None else 0.0)
            col3.metric("Total — confirmados/total", f"{rates['total_conf']}/{rates['total_cross']}" if rates['total_cross']>0 else "0/0")
            col3.metric("Taxa de confirmação (TOTAL) %", rates["rate_total_%"] if rates["rate_total_%"] is not None else 0.0)

            st.subheader("Retorno entre o início e o fim de cada ciclo (apenas entradas confirmadas)")
            if cycles.empty:
                st.info("Nenhum ciclo confirmado encontrado (poucas confirmações por volume).")
            else:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Nº de ciclos", summary.get("num_cycles", 0))
                c2.metric("Retorno médio (%)", summary.get("avg_return_%", 0.0))
                c3.metric("Retorno mediano (%)", summary.get("median_return_%", 0.0))
                c4.metric("Ret. médio ALTA / BAIXA (%)",
                          f"{summary.get('avg_return_alta_%', 0.0)} / {summary.get('avg_return_baixa_%', 0.0)}")

                st.dataframe(cycles.sort_values("Entrada"), use_container_width=True)

st.caption("⚠️ Base diária, últimos 12 meses. Retorno do ciclo usa preço de **fechamento** na entrada e na saída.")