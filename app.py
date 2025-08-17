# app.py (v11) — EMA9/EMA21 + Regra de 3% em H dias (5/10/15/20) — sem volume
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

    # Padronizar colunas e dtypes (sem Volume)
    df = df.rename(columns=lambda c: str(c).title())
    cols = ["Open", "High", "Low", "Close"]
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

# ---------- Avaliação 3% ----------
def evaluate_three_percent_rule(df: pd.DataFrame, horizon: int, threshold_pct: float = 3.0) -> pd.DataFrame:
    """
    Para cada cruzamento:
      - Direção ALTA (BullCross): sucesso se, dentro de H dias, o **HIGH** tocar >= Entry * (1+3%).
      - Direção BAIXA (BearCross): sucesso se, dentro de H dias, o **LOW** tocar <= Entry * (1-3%).
      - Saída: na primeira barra que tocar o alvo (3%).
      - Se não tocar até H dias, marcar como 'Não atingiu 3%' e usar D+H como data de avaliação.
    Retornos reportados:
      - 'Ret_%_no_evento': 3.00 se atingiu; senão o retorno direcional em D→D+H.
    """
    rows = []
    mask = df["BullCross"] | df["BearCross"]
    idxs = df.index[mask]
    thr = threshold_pct / 100.0

    for ts in idxs:
        pos = df.index.get_loc(ts)
        entry = float(df.loc[ts, "Close"])
        direcao = "ALTA" if bool(df.loc[ts, "BullCross"]) else "BAIXA"

        # Janela de avaliação: dias (1..H) após o sinal
        fpos = pos + horizon
        if fpos >= len(df.index):
            # janela incompleta no fim da série -> ignora
            continue

        window = df.iloc[pos+1:fpos+1]  # inclui D+H
        hit = False
        exit_dt = None
        exit_price = None

        if direcao == "ALTA":
            target = entry * (1.0 + thr)
            # primeira barra onde HIGH >= target
            hit_idx = window.index[window["High"] >= target]
            if len(hit_idx) > 0:
                hit = True
                exit_dt = hit_idx[0]
                exit_price = target  # considera preço alvo atingido
            else:
                exit_dt = df.index[fpos]
                exit_price = float(df.loc[exit_dt, "Close"])
        else:  # BAIXA
            target = entry * (1.0 - thr)
            hit_idx = window.index[window["Low"] <= target]
            if len(hit_idx) > 0:
                hit = True
                exit_dt = hit_idx[0]
                exit_price = target
            else:
                exit_dt = df.index[fpos]
                exit_price = float(df.loc[exit_dt, "Close"])

        # retorno
        if hit:
            ret = threshold_pct  # por definição: alvo de 3%
            bars_to = int(df.index.get_loc(exit_dt) - pos)
        else:
            if direcao == "ALTA":
                ret = (exit_price / entry - 1.0) * 100.0
            else:
                ret = (entry / exit_price - 1.0) * 100.0
            bars_to = int(horizon)

        rows.append({
            "Data_Sinal": ts,
            "Direcao": direcao,
            "Entrada_Close": round(entry, 6),
            "H_dias": horizon,
            "Atingiu_3pct": bool(hit),
            "Data_Fechamento_Operacao": exit_dt,
            "Preco_Saida": round(exit_price, 6),
            "Barras_ate_evento": bars_to,
            "Ret_%_no_evento": round(ret, 2),
        })

    return pd.DataFrame(rows)

def summarize_hits(df_eval: pd.DataFrame) -> dict:
    if df_eval is None or df_eval.empty:
        return {"n": 0}
    n = len(df_eval)
    hits = int(df_eval["Atingiu_3pct"].sum())
    misses = n - hits
    hit_pct = round(100.0 * hits / n, 2)
    miss_pct = round(100.0 * misses / n, 2)
    # por direção (opcional)
    bull = df_eval[df_eval["Direcao"] == "ALTA"]
    bear = df_eval[df_eval["Direcao"] == "BAIXA"]
    bh = int(bull["Atingiu_3pct"].sum())
    sh = int(bear["Atingiu_3pct"].sum())
    return {
        "n": n,
        "hits": hits,
        "misses": misses,
        "hit_pct_%": hit_pct,
        "miss_pct_%": miss_pct,
        "bull_hits": bh,
        "bear_hits": sh,
        "bull_total": len(bull),
        "bear_total": len(bear),
    }

# ---------- Plot ----------
def plot_chart(df: pd.DataFrame, eval_df: pd.DataFrame, ticker_in: str):
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

    # Marcadores: sinal de cruzamento colorido por sucesso (atingiu 3% no H) vs não
    if eval_df is not None and not eval_df.empty:
        succ = eval_df[eval_df["Atingiu_3pct"] == True]
        fail = eval_df[eval_df["Atingiu_3pct"] == False]
        if not succ.empty:
            fig.add_trace(go.Scatter(
                x=succ["Data_Sinal"], y=[df.loc[d, "Close"] for d in succ["Data_Sinal"]],
                mode="markers", name="Sinal (atingiu ≥3%)",
                marker_symbol="star", marker_size=11
            ), row=1, col=1)
        if not fail.empty:
            fig.add_trace(go.Scatter(
                x=fail["Data_Sinal"], y=[df.loc[d, "Close"] for d in fail["Data_Sinal"]],
                mode="markers", name="Sinal (<3% no H)",
                marker_symbol="x", marker_size=10
            ), row=1, col=1)

    fig.update_layout(
        title=f"{normalize_ticker_b3(ticker_in)} — Cruzamentos EMA9/EMA21 e regra de 3% em H dias",
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        margin=dict(l=20, r=20, t=50, b=20),
        height=720
    )
    fig.update_yaxes(title_text="Preço", row=1, col=1)
    return fig

# ---------- UI ----------
st.set_page_config(page_title="EMA9/21 — Regra 3% (B3)", layout="wide")
st.title("Cruzamentos EMA9/EMA21 — Sucesso se atingir 3% em H dias (5/10/15/20) — Últimos 12 meses")

with st.sidebar:
    st.markdown("### Parâmetros")
    st.write("""
- **Sinal**: cruzamento EMA9×EMA21 em qualquer direção.
- **Sucesso**: dentro de **H** dias úteis após o sinal, o preço **atinge** ±3% a favor da direção (usa **HIGH** para ALTA e **LOW** para BAIXA).
- **Saída**: na **primeira** barra que toca o alvo de 3%; caso contrário, usa D+H para avaliar.
- **Sem** volume e **sem** EMA50.
    """)
    horizon = st.selectbox("H (dias úteis):", [5, 10, 15, 20], index=0)
    st.caption("Atingimento usa HIGH/LOW intradiário; retornos mostram 3% quando bate o alvo.")

ticker_in = st.text_input("Ticker (ex.: ITUB4, PETR4, VALE3)", value="ITUB4")

if st.button("Analisar", type="primary"):
    with st.spinner("Baixando dados e calculando..."):
        df_raw = fetch_last_12m(ticker_in)
        if df_raw.empty:
            st.error("Não encontrei dados para esse ticker no período.")
        else:
            df = add_indicators(df_raw)

            # Avaliar a regra de 3% no horizonte escolhido
            eval_df = evaluate_three_percent_rule(df, horizon=horizon, threshold_pct=3.0)
            summary = summarize_hits(eval_df)

            # Gráfico
            fig = plot_chart(df, eval_df, ticker_in)
            st.plotly_chart(fig, use_container_width=True)

            st.subheader(f"Resultados no horizonte de {horizon} dias úteis")
            if eval_df.empty:
                st.info("Sem sinais suficientes (ou muito perto do fim da série para avaliar esse horizonte).")
            else:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total de operações", summary.get("n", 0))
                c2.metric("Acertivas (≥3%)", summary.get("hits", 0))
                c3.metric("Não atingiram 3%", summary.get("misses", 0))
                c4.metric("Taxa de acerto (≥3%)", f"{summary.get('hit_pct_%', 0.0)}%")

                d1, d2 = st.columns(2)
                d1.metric("Percentual ≥3%", f"{summary.get('hit_pct_%', 0.0)}%")
                d2.metric("Percentual <3%", f"{summary.get('miss_pct_%', 0.0)}%")

                st.caption("Quebra por direção (informativo)")
                e1, e2 = st.columns(2)
                e1.metric("ALTA — acertos/total", f"{summary.get('bull_hits', 0)}/{summary.get('bull_total', 0)}")
                e2.metric("BAIXA — acertos/total", f"{summary.get('bear_hits', 0)}/{summary.get('bear_total', 0)}")

                st.subheader("Operações (detalhe)")
                st.dataframe(eval_df.sort_values("Data_Sinal"), use_container_width=True)

st.caption("⚠️ Base diária, últimos 12 meses. Sinais pelo fechamento de D; alvo de 3% avaliado por HIGH/LOW até D+H.")