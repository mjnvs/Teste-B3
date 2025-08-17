# app.py (v7) — EMA9/EMA21 + Confirmação por Volume>=EMA50 | 12 meses | Diário
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

def extract_events_table(df: pd.DataFrame) -> pd.DataFrame:
    events = []
    for ts, row in df.iterrows():
        if bool(row["BullCross"]) or bool(row["BearCross"]):
            events.append({
                "Data": ts,
                "Tipo": "ALTA" if bool(row["BullCross"]) else "BAIXA",
                "Fechamento": round(float(row["Close"]), 6),
                "Volume": int(row["Volume"]),
                "VolEMA50": round(float(row["VolEMA50"]), 2) if not np.isnan(row["VolEMA50"]) else None,
                "Confirmado": bool(row["VolConfirm"])
            })
    return pd.DataFrame(events)

def plot_chart(df: pd.DataFrame, ticker_in: str):
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

    # Destaque dos cruzamentos confirmados
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

    fig.update_layout(
        title=f"{normalize_ticker_b3(ticker_in)} — Diário (12 meses) | Candles + EMA9/21 | Volume + EMA50",
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        margin=dict(l=20, r=20, t=50, b=20),
        height=760
    )
    fig.update_yaxes(title_text="Preço", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    return fig

# ---------- UI ----------
st.set_page_config(page_title="EMA9/21 + Volume (B3)", layout="wide")
st.title("Análise: EMA9/EMA21 + Confirmação por Volume (EMA50) — B3 (12 meses)")

with st.sidebar:
    st.markdown("### Instruções")
    st.write("""
- **Entrada de alta**: EMA9 cruza **para cima** a EMA21.
- **Entrada de baixa**: EMA9 cruza **para baixo** a EMA21.
- **Confirmação**: no mesmo candle, `Volume >= EMA50(Volume)` (tocou ou passou).
- Gráfico diário, últimos **12 meses** a partir da data atual.
- Candles: **verde** (alta), **vermelho** (baixa).
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
            rates = compute_confirmation_rates(df)
            fig = plot_chart(df, ticker_in)
            st.plotly_chart(fig, use_container_width=True)

            # Métricas de confirmação
            st.subheader("Taxas de confirmação")
            col1, col2, col3 = st.columns(3)
            col1.metric("Alta — confirmados/total", f"{rates['conf_bull']}/{rates['total_bull']}" if rates['total_bull']>0 else "0/0")
            col1.metric("Taxa de confirmação (ALTA) %", rates["rate_bull_%"] if rates["rate_bull_%"] is not None else 0.0)
            col2.metric("Baixa — confirmados/total", f"{rates['conf_bear']}/{rates['total_bear']}" if rates['total_bear']>0 else "0/0")
            col2.metric("Taxa de confirmação (BAIXA) %", rates["rate_bear_%"] if rates["rate_bear_%"] is not None else 0.0)
            col3.metric("Total — confirmados/total", f"{rates['total_conf']}/{rates['total_cross']}" if rates['total_cross']>0 else "0/0")
            col3.metric("Taxa de confirmação (TOTAL) %", rates["rate_total_%"] if rates["rate_total_%"] is not None else 0.0)

            # Tabela de eventos
            st.subheader("Cruzamentos e confirmações (últimos 12 meses)")
            events = extract_events_table(df)
            if events.empty:
                st.info("Nenhum cruzamento detectado no período.")
            else:
                st.dataframe(events.sort_values("Data"), use_container_width=True)