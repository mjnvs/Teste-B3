# app.py (v3) — B3 EMA9/EMA21 + Volume EMA50 + Índice de Acerto
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
        group_by="column",
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

def score_signals(df: pd.DataFrame, horizon: int = 10) -> pd.DataFrame:
    """
    Avalia o 'índice de acerto' usando uma regra simples:
    - Para ALTA_CONFIRMADA: acerto se Close(t+h) > Close(t)
    - Para BAIXA_CONFIRMADA: acerto se Close(t+h) < Close(t)
    Retorno direcional (%): 
      - bull: (Close_future / Close_entry - 1)
      - bear: (Close_entry / Close_future - 1)  # positivo se cair
    """
    conf = df[(df["BullConfirmed"]) | (df["BearConfirmed"])].copy()
    if conf.empty:
        return pd.DataFrame()

    conf["Entrada"] = conf["Close"]
    conf["Tipo"] = np.where(conf["BullConfirmed"], "ALTA_CONFIRMADA", "BAIXA_CONFIRMADA")

    rows = []
    idx_list = conf.index.tolist()
    for ts in idx_list:
        if ts not in df.index:
            continue
        pos = df.index.get_loc(ts)
        future_pos = pos + horizon
        if future_pos >= len(df.index):
            # sem dados suficientes para avaliar
            continue
        future_ts = df.index[future_pos]
        close_entry = float(df.loc[ts, "Close"])
        close_future = float(df.loc[future_ts, "Close"])
        tipo = "ALTA_CONFIRMADA" if bool(df.loc[ts, "BullConfirmed"]) else "BAIXA_CONFIRMADA"

        if tipo == "ALTA_CONFIRMADA":
            acerto = close_future > close_entry
            ret_dir = (close_future / close_entry) - 1.0
        else:
            acerto = close_future < close_entry
            # retorno direcional positivo quando o preço cai (benefício de venda)
            ret_dir = (close_entry / close_future) - 1.0

        rows.append({
            "Data_Sinal": ts,
            "Tipo": tipo,
            "Entrada": close_entry,
            "Data_Avaliacao": future_ts,
            "Close_Avaliacao": close_future,
            f"Retorno_{horizon}d_%": round(ret_dir * 100.0, 2),
            "Acertou": bool(acerto),
        })

    return pd.DataFrame(rows).sort_values("Data_Sinal")

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
        height=760
    )
    fig.update_yaxes(title_text="Preço", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    return fig

# ---------- UI ----------
st.set_page_config(page_title="Análise Técnica B3 (EMA+Volume)", layout="wide")
st.title("Análise Técnica B3: EMA9/EMA21 + Volume + Índice de Acerto")

with st.sidebar:
    st.markdown("### Como usar")
    st.write("Digite o **ticker** (ex.: ITUB4, PETR4, VALE3). O app adiciona automaticamente o sufixo **.SA**.")
    horizon = st.selectbox("Janela de avaliação do índice de acerto (dias úteis):", [5, 10, 20], index=1)
    st.caption("⚠️ Uso educacional. Backtest simples (sem custos, slippage, nem gestão de risco).")

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

            # Tabela de sinais (todos)
            sigs = extract_signals(df)
            st.subheader("Sinais detectados (últimos 12 meses)")
            if sigs.empty:
                st.info("Nenhum cruzamento detectado no período.")
            else:
                st.dataframe(
                    sigs.reset_index(names=["Data"]),
                    use_container_width=True
                )

            # Índice de acerto (apenas sinais confirmados)
            st.subheader(f"Índice de acerto — avaliação em {horizon} dias úteis (sinais confirmados)")
            scored = score_signals(df, horizon=horizon)
            if scored.empty:
                st.info("Não há sinais confirmados suficientes para calcular o índice de acerto.")
            else:
                # Métricas
                total = len(scored)
                acertos = int(scored["Acertou"].sum())
                hit_rate = round(100 * acertos / total, 2)

                bull_scored = scored[scored["Tipo"] == "ALTA_CONFIRMADA"]
                bear_scored = scored[scored["Tipo"] == "BAIXA_CONFIRMADA"]
                hit_bull = round(100 * bull_scored["Acertou"].mean(), 2) if not bull_scored.empty else None
                hit_bear = round(100 * bear_scored["Acertou"].mean(), 2) if not bear_scored.empty else None

                mean_ret = round(scored[f"Retorno_{horizon}d_%"].mean(), 2)
                mean_ret_bull = round(bull_scored[f"Retorno_{horizon}d_%"].mean(), 2) if not bull_scored.empty else None
                mean_ret_bear = round(bear_scored[f"Retorno_{horizon}d_%"].mean(), 2) if not bear_scored.empty else None

                colA, colB, colC = st.columns(3)
                colA.metric("Hit rate total (%)", hit_rate)
                colB.metric("Hit rate ALTA (%)", hit_bull if hit_bull is not None else 0.0)
                colC.metric("Hit rate BAIXA (%)", hit_bear if hit_bear is not None else 0.0)

                colD, colE, colF = st.columns(3)
                colD.metric(f"Retorno médio {horizon}d (%)", mean_ret)
                colE.metric(f"Ret. médio ALTA {horizon}d (%)", mean_ret_bull if mean_ret_bull is not None else 0.0)
                colF.metric(f"Ret. médio BAIXA {horizon}d (%)", mean_ret_bear if mean_ret_bear is not None else 0.0)

                st.dataframe(scored, use_container_width=True)
