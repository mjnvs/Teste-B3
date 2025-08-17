# app.py (v2) — B3 EMA9/EMA21 + Volume EMA50 com correções para MultiIndex e dtypes
# Requisitos: pip install streamlit yfinance plotly pandas python-dateutil
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# ---------- Utilidades ----------
def normalize_ticker_b3(t: str) -> str:
    t = (t or "").strip().upper()
    if not t:
        return ""
    if t.startswith("^") or t.endswith(".SA"):
        return t
    return f"{t}.SA"

def fetch_last_12m(ticker_raw: str) -> pd.DataFrame:
    """Baixa últimos 12 meses e ACHATA possíveis colunas MultiIndex, garantindo dtypes numéricos."""
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
        group_by="column",  # ajuda a evitar MultiIndex, mas ainda tratamos se vier
    )

    if df.empty:
        return df

    # Se vier MultiIndex (ex.: ('Open','ITUB4.SA')), achata para um nível
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df.columns = df.columns.get_level_values(0)
        except Exception:
            df = df.droplevel(-1, axis=1)

    # Padroniza colunas
    df = df.rename(columns=lambda c: str(c).title())
    cols = ["Open", "High", "Low", "Close", "Volume"]
    df = df[[c for c in cols if c in df.columns]]

    # Garante dtype numérico
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Remove linhas sem dados essenciais
    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])

    # Índice datetime sem timezone
    df.index = pd.to_datetime(df.index).tz_localize(None)

    return df

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula EMAs e sinais de cruzamento com confirmação por volume."""
    out = df.copy()

    # Garante Series numéricas
    close = pd.to_numeric(out["Close"], errors="coerce")
    vol = pd.to_numeric(out["Volume"], errors="coerce")

    out["EMA9"]      = close.ewm(span=9, adjust=False).mean()
    out["EMA21"]     = close.ewm(span=21, adjust=False).mean()
    out["VolEMA50"]  = vol.ewm(span=50, adjust=False).mean()

    # Cruzamentos
    prev_up = out["EMA9"].shift(1) <= out["EMA21"].shift(1)
    prev_dn = out["EMA9"].shift(1) >= out["EMA21"].shift(1)
    bull_cross = (out["EMA9"] > out["EMA21"]) & prev_up
    bear_cross = (out["EMA9"] < out["EMA21"]) & prev_dn

    # Confirmação por volume no MESMO candle
    vol_conf = vol > out["VolEMA50"]

    out["BullCross"]     = bull_cross
    out["BearCross"]     = bear_cross
    out["BullConfirmed"] = bull_cross & vol_conf
    out["BearConfirmed"] = bear_cross & vol_conf

    out["Signal"] = np.where(out["BullConfirmed"], "ALTA_CONFIRMADA",
                     np.where(out["BearConfirmed"], "BAIXA_CONFIRMADA",
                     np.where(out["BullCross"], "ALTA_NAO_CONFIRMADA",
                     np.where(out["BearCross"], "BAIXA_NAO_CONFIRMADA", ""))))
    return out

def extract_signals(df: pd.DataFrame) -> pd.DataFrame:
    sigs = df[df["Signal"] != ""].copy()
    if sigs.empty:
        return sigs
    sigs = sigs[["Signal", "Close", "Volume", "VolEMA50"]]
    sigs = sigs.rename(columns={"Close":"Fechamento"})
    return sigs

def summarize(df: pd.DataFrame, ticker_in: str) -> str:
    if df.empty:
        return "Sem dados para o período consultado."

    last = df.iloc[-1]
    trend = "ALTA" if last["EMA9"] > last["EMA21"] else "BAIXA"
    vol_conf_now = "SIM" if last["Volume"] > last["VolEMA50"] else "NÃO"

    # acha último sinal confirmado (se existir)
    confirmed = df[(df["BullConfirmed"]) | (df["BearConfirmed"])]
    if not confirmed.empty:
        last_sig_dt = confirmed.index[-1].date()
        last_sig = "ALTA" if confirmed.iloc[-1]["BullConfirmed"] else "BAIXA"
        last_sig_price = round(float(confirmed.iloc[-1]["Close"]), 4)
        conf_txt = f"Último sinal CONFIRMADO: {last_sig} em {last_sig_dt} (fechamento ≈ {last_sig_price})."
    else:
        conf_txt = "Sem sinal confirmado nos últimos 12 meses."

    return (
        f"Ticker: {normalize_ticker_b3(ticker_in)} | Tendência atual (EMA9 vs EMA21): {trend}. "
        f"Volume hoje acima da EMA50? {vol_conf_now}. {conf_txt}"
    )

def plot_chart(df: pd.DataFrame, ticker_in: str):
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04,
        row_heights=[0.68, 0.32]
    )

    # Candle
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="Candles"
    ), row=1, col=1)

    # EMAs
    fig.add_trace(go.Scatter(
        x=df.index, y=df["EMA9"], mode="lines", name="EMA 9"
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["EMA21"], mode="lines", name="EMA 21"
    ), row=1, col=1)

    # Sinais confirmados (marcadores no preço de fechamento)
    bull = df[df["BullConfirmed"]]
    bear = df[df["BearConfirmed"]]
    if not bull.empty:
        fig.add_trace(go.Scatter(
            x=bull.index, y=bull["Close"], mode="markers",
            name="Alta confirmada", marker_symbol="triangle-up", marker_size=10
        ), row=1, col=1)
    if not bear.empty:
        fig.add_trace(go.Scatter(
            x=bear.index, y=bear["Close"], mode="markers",
            name="Baixa confirmada", marker_symbol="triangle-down", marker_size=10
        ), row=1, col=1)

    # Volume
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"], name="Volume"
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["VolEMA50"], mode="lines", name="EMA Vol 50"
    ), row=2, col=1)

    fig.update_layout(
        title=f"{normalize_ticker_b3(ticker_in)} — 12 meses (Candles + EMA9/21) | Volume + EMA50",
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        margin=dict(l=20, r=20, t=50, b=20),
        height=720
    )
    fig.update_yaxes(title_text="Preço", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    return fig

# ---------- UI ----------
st.set_page_config(page_title="Análise Técnica B3 (EMA+Volume)", layout="wide")
st.title("Análise Técnica B3: EMA9/EMA21 + Confirmação por Volume")

with st.sidebar:
    st.markdown("### Como usar")
    st.write("Digite o **ticker** (ex.: ITUB4, PETR4, VALE3). O app adiciona automaticamente o sufixo **.SA**.")
    st.caption("⚠️ Uso educacional. Não é recomendação de investimento.")

ticker_in = st.text_input("Ticker (ex.: ITUB4, PETR4, VALE3)", value="ITUB4")

if st.button("Analisar", type="primary"):
    with st.spinner("Baixando dados e calculando indicadores..."):
        df_raw = fetch_last_12m(ticker_in)
        if df_raw.empty:
            st.error("Não encontrei dados para esse ticker no período. Verifique o código (ex.: ITUB4, PETR4...).")
        else:
            df = add_indicators(df_raw)
            st.success("Análise concluída!")
            st.write(summarize(df, ticker_in))

            # Gráfico
            fig = plot_chart(df, ticker_in)
            st.plotly_chart(fig, use_container_width=True)

            # Tabela de sinais
            sigs = extract_signals(df)
            st.subheader("Sinais detectados (últimos 12 meses)")
            if sigs.empty:
                st.info("Nenhum cruzamento detectado no período.")
            else:
                st.dataframe(
                    sigs.reset_index(names=["Data"]),
                    use_container_width=True
                )
