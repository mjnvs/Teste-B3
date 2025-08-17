# app.py (v10) — Somente cruzamentos EMA9/EMA21 (sem volume) | Ciclos + Horizontes
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
    cols = ["Open", "High", "Low", "Close"]
    # mantemos só o que precisamos (sem Volume)
    df = df[[c for c in cols if c in df.columns]]
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = pd.to_numeric(out["Close"], errors="coerce")
    out["EMA9"] = close.ewm(span=9, adjust=False).mean()
    out["EMA21"] = close.ewm(span=21, adjust=False).mean()

    # Cruzamentos
    prev_up = out["EMA9"].shift(1) <= out["EMA21"].shift(1)
    prev_dn = out["EMA9"].shift(1) >= out["EMA21"].shift(1)
    out["BullCross"] = (out["EMA9"] > out["EMA21"]) & prev_up
    out["BearCross"] = (out["EMA9"] < out["EMA21"]) & prev_dn
    return out

# ---------- Métricas ----------
def compute_cross_counts(df: pd.DataFrame) -> dict:
    total_bull = int(df["BullCross"].sum())
    total_bear = int(df["BearCross"].sum())
    total_cross = total_bull + total_bear
    return {"bull": total_bull, "bear": total_bear, "total": total_cross}

def compute_cross_cycles_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constrói ciclos a partir de QUALQUER cruzamento (sem volume) e encerra no PRÓXIMO cruzamento.
    Retorno do ciclo:
      - ALTA (bull): (Close_saida / Close_entrada - 1) * 100
      - BAIXA (bear): (Close_entrada / Close_saida - 1) * 100
    """
    events = []
    for ts, row in df.iterrows():
        if bool(row["BullCross"]):
            events.append({"date": ts, "type": "BULL"})
        elif bool(row["BearCross"]):
            events.append({"date": ts, "type": "BEAR"})

    if len(events) == 0:
        return pd.DataFrame()

    cycles = []
    for i in range(len(events) - 1):
        entry = events[i]
        exit_ = events[i+1]
        entry_dt = entry["date"]
        exit_dt = exit_["date"]
        entry_close = float(df.loc[entry_dt, "Close"])
        exit_close = float(df.loc[exit_dt, "Close"])
        direction = "ALTA" if entry["type"] == "BULL" else "BAIXA"

        if direction == "ALTA":
            ret = (exit_close / entry_close - 1.0) * 100.0
        else:
            ret = (entry_close / exit_close - 1.0) * 100.0

        try:
            dur = int(df.index.get_loc(exit_dt) - df.index.get_loc(entry_dt))
        except Exception:
            dur = None

        cycles.append({
            "Entrada": entry_dt,
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

def evaluate_horizon_from_crosses(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    Avalia retorno em H dias úteis a partir de TODO cruzamento (sem volume).
    - ALTA: (Close_{t+H}/Close_t - 1)*100
    - BAIXA: (Close_t/Close_{t+H} - 1)*100
    """
    rows = []
    mask = df["BullCross"] | df["BearCross"]
    idxs = df.index[mask]
    for ts in idxs:
        pos = df.index.get_loc(ts)
        fpos = pos + horizon
        if fpos >= len(df.index):
            continue  # sem dados suficientes
        ts_future = df.index[fpos]
        entry = float(df.loc[ts, "Close"])
        future = float(df.loc[ts_future, "Close"])
        direcao = "ALTA" if bool(df.loc[ts, "BullCross"]) else "BAIXA"
        if direcao == "ALTA":
            ret = (future / entry - 1.0) * 100.0
            acertou = future > entry
        else:
            ret = (entry / future - 1.0) * 100.0
            acertou = future < entry
        rows.append({
            "Data_Sinal": ts,
            "Direcao": direcao,
            "Entrada_Close": round(entry, 6),
            "Data_Avaliacao": ts_future,
            "Close_Avaliacao": round(future, 6),
            f"Ret_{horizon}d_%": round(ret, 2),
            "Acertou": bool(acertou),
        })
    return pd.DataFrame(rows)

def summarize_horizon(df_eval: pd.DataFrame, horizon: int) -> dict:
    if df_eval is None or df_eval.empty:
        return {"n": 0}
    n = len(df_eval)
    hit = int(df_eval["Acertou"].sum())
    hit_rate = round(100 * hit / n, 2)
    avg_ret = round(df_eval[f"Ret_{horizon}d_%"].mean(), 2)
    bull = df_eval[df_eval["Direcao"] == "ALTA"]
    bear = df_eval[df_eval["Direcao"] == "BAIXA"]
    hit_bull = round(100 * bull["Acertou"].mean(), 2) if not bull.empty else None
    hit_bear = round(100 * bear["Acertou"].mean(), 2) if not bear.empty else None
    avg_bull = round(bull[f"Ret_{horizon}d_%"].mean(), 2) if not bull.empty else None
    avg_bear = round(bear[f"Ret_{horizon}d_%"].mean(), 2) if not bear.empty else None
    return {
        "n": n, "hit_rate_%": hit_rate, "avg_ret_%": avg_ret,
        "hit_rate_bull_%": hit_bull, "hit_rate_bear_%": hit_bear,
        "avg_ret_bull_%": avg_bull, "avg_ret_bear_%": avg_bear
    }

# ---------- Plot ----------
def plot_chart(df: pd.DataFrame, cycles: pd.DataFrame, ticker_in: str):
    fig = make_subplots(rows=1, cols=1)

    # Candlestick com cores: verde (alta), vermelho (baixa)
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="Candles",
        increasing_line_color="green", increasing_fillcolor="green",
        decreasing_line_color="red", decreasing_fillcolor="red"
    ), row=1, col=1)

    # EMAs
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA9"], mode="lines", name="EMA 9"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA21"], mode="lines", name="EMA 21"), row=1, col=1)

    # Marcadores de cruzamentos
    bulls = df[df["BullCross"]]
    bears = df[df["BearCross"]]
    if not bulls.empty:
        fig.add_trace(go.Scatter(
            x=bulls.index, y=bulls["Close"], mode="markers",
            name="Cruz. ALTA", marker_symbol="triangle-up", marker_size=10
        ), row=1, col=1)
    if not bears.empty:
        fig.add_trace(go.Scatter(
            x=bears.index, y=bears["Close"], mode="markers",
            name="Cruz. BAIXA", marker_symbol="triangle-down", marker_size=10
        ), row=1, col=1)

    # Faixas dos ciclos (entre cruzamentos)
    if cycles is not None and not cycles.empty:
        for _, tr in cycles.iterrows():
            color = "rgba(46, 204, 113, 0.18)" if tr["Direcao"] == "ALTA" else "rgba(231, 76, 60, 0.18)"
            fig.add_vrect(x0=tr["Entrada"], x1=tr["Saida"], fillcolor=color, line_width=0)

    fig.update_layout(
        title=f"{normalize_ticker_b3(ticker_in)} — Cruzamentos EMA9/EMA21 (sem volume) | 12 meses, diário",
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        margin=dict(l=20, r=20, t=50, b=20),
        height=720
    )
    fig.update_yaxes(title_text="Preço", row=1, col=1)
    return fig

# ---------- UI ----------
st.set_page_config(page_title="EMA9/21 (sem volume) — B3", layout="wide")
st.title("Somente cruzamentos EMA9/EMA21 — B3 (12 meses, diário)")

with st.sidebar:
    st.markdown("### Regras")
    st.write("""
- **Alta**: EMA9 cruza **para cima** EMA21.
- **Baixa**: EMA9 cruza **para baixo** EMA21.
- **Sem confirmação por volume** (não usamos Volume nem EMA50).
- **Ciclo**: começa no cruzamento e termina no **próximo** cruzamento.
    """)
    horizon = st.selectbox("Janela de avaliação (dias úteis):", [5, 10, 15, 20], index=0)
    st.caption("A avaliação por horizonte usa o **fechamento** de D e D+H.")

ticker_in = st.text_input("Ticker (ex.: ITUB4, PETR4, VALE3)", value="ITUB4")

if st.button("Analisar", type="primary"):
    with st.spinner("Baixando dados e calculando..."):
        df_raw = fetch_last_12m(ticker_in)
        if df_raw.empty:
            st.error("Não encontrei dados para esse ticker no período.")
        else:
            df = add_indicators(df_raw)

            # Contagem de cruzamentos
            counts = compute_cross_counts(df)

            # Ciclos entre cruzamentos
            cycles = compute_cross_cycles_returns(df)
            summary = summarize_cycles(cycles)

            # Avaliação por horizonte (a partir de todos os cruzamentos)
            eval_h = evaluate_horizon_from_crosses(df, horizon=horizon)
            summ_h = summarize_horizon(eval_h, horizon=horizon)

            # Gráfico
            fig = plot_chart(df, cycles, ticker_in)
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Cruzamentos detectados (últimos 12 meses)")
            c1, c2, c3 = st.columns(3)
            c1.metric("Cruz. de ALTA", counts["bull"])
            c2.metric("Cruz. de BAIXA", counts["bear"])
            c3.metric("Total de cruzamentos", counts["total"])

            st.subheader("Retorno entre o início e o fim de cada ciclo (entre cruzamentos)")
            if cycles.empty:
                st.info("Nenhum ciclo formado (poucos cruzamentos).")
            else:
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Nº de ciclos", summary.get("num_cycles", 0))
                k2.metric("Retorno médio (%)", summary.get("avg_return_%", 0.0))
                k3.metric("Retorno mediano (%)", summary.get("median_return_%", 0.0))
                k4.metric("Ret. médio ALTA / BAIXA (%)",
                          f"{summary.get('avg_return_alta_%', 0.0)} / {summary.get('avg_return_baixa_%', 0.0)}")
                st.dataframe(cycles.sort_values("Entrada"), use_container_width=True)

            st.subheader(f"Avaliação por horizonte de {horizon} dias úteis (a partir de cada cruzamento)")
            if eval_h.empty:
                st.info("Sem cruzamentos suficientes para avaliar nesse horizonte (ou muito perto do fim da série).")
            else:
                h1, h2, h3, h4 = st.columns(4)
                h1.metric("Sinais avaliados", summ_h.get("n", 0))
                h2.metric("Hit rate total (%)", summ_h.get("hit_rate_%", 0.0))
                h3.metric("Retorno médio (%)", summ_h.get("avg_ret_%", 0.0))
                h4.metric("Hit ALTA / BAIXA (%)", f"{summ_h.get('hit_rate_bull_%', 0.0)} / {summ_h.get('hit_rate_bear_%', 0.0)}")

                h5, h6 = st.columns(2)
                h5.metric("Ret. médio ALTA (%)", summ_h.get("avg_ret_bull_%", 0.0) if summ_h.get("avg_ret_bull_%") is not None else 0.0)
                h6.metric("Ret. médio BAIXA (%)", summ_h.get("avg_ret_bear_%", 0.0) if summ_h.get("avg_ret_bear_%") is not None else 0.0)

                st.dataframe(eval_h.sort_values("Data_Sinal"), use_container_width=True)

st.caption("⚠️ Base diária, últimos 12 meses. Sem uso de Volume/EMA50. Avaliação por horizonte usa fechamentos de D e D+H.")