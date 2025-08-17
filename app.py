# app.py (v14) — EMA9/EMA21 | Sucesso se atingir ±1% em 1 semana (5 pregões) | Saída semanal e retorno 4 semanas
# Requisitos: pip install streamlit yfinance plotly pandas python-dateutil

import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from plotly.subplots import make_subplots
import plotly.graph_objects as go

H_WEEK = 5           # 1 semana de pregão (aprox.)
THRESHOLD_PCT = 1.0  # alvo de 1%

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

# ---------- Avaliação semanal (5 pregões) ----------
def start_of_week_monday(ts: pd.Timestamp) -> pd.Timestamp:
    d = pd.Timestamp(ts).normalize()
    return d - pd.Timedelta(days=d.weekday())

def evaluate_weekly_threshold(df: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada cruzamento:
      - ALTA: sucesso se, em até 5 pregões, HIGH >= Entry * (1 + 1%)
      - BAIXA: sucesso se, em até 5 pregões, LOW  <= Entry * (1 - 1%)
      - Saída no primeiro toque; senão usa D+5 (fechamento) como referência.
    """
    rows = []
    mask = df["BullCross"] | df["BearCross"]
    idxs = df.index[mask]
    thr = THRESHOLD_PCT / 100.0

    for ts in idxs:
        pos = df.index.get_loc(ts)
        entry = float(df.loc[ts, "Close"])
        direcao = "ALTA" if bool(df.loc[ts, "BullCross"]) else "BAIXA"
        fpos = pos + H_WEEK
        if fpos >= len(df.index):
            continue

        window = df.iloc[pos+1:fpos+1]  # inclui D+5
        hit = False
        exit_dt = None
        exit_price = None

        if direcao == "ALTA":
            target = entry * (1.0 + thr)
            hit_idx = window.index[window["High"] >= target]
            if len(hit_idx) > 0:
                hit = True
                exit_dt = hit_idx[0]
                exit_price = target
            else:
                exit_dt = df.index[fpos]
                exit_price = float(df.loc[exit_dt, "Close"])
        else:
            target = entry * (1.0 - thr)
            hit_idx = window.index[window["Low"] <= target]
            if len(hit_idx) > 0:
                hit = True
                exit_dt = hit_idx[0]
                exit_price = target
            else:
                exit_dt = df.index[fpos]
                exit_price = float(df.loc[exit_dt, "Close"])

        # Retornos auxiliares
        ret_long = (exit_price / entry - 1.0) * 100.0
        ret_short = (entry / exit_price - 1.0) * 100.0
        if hit:
            ret_dir = THRESHOLD_PCT
            bars_to = int(df.index.get_loc(exit_dt) - pos)
        else:
            ret_dir = ret_long if direcao == "ALTA" else ret_short
            bars_to = H_WEEK

        rows.append({
            "Data_Sinal": ts,
            "Semana": start_of_week_monday(ts),
            "Direcao": direcao,
            "Entrada_Close": round(entry, 6),
            "Atingiu_1pct": bool(hit),
            "Data_Fechamento_Operacao": exit_dt,
            "Preco_Saida": round(exit_price, 6),
            "Barras_ate_evento": bars_to,
            "Ret_%_direcao": round(ret_dir, 2),
            "Ret_long_%": round(ret_long, 2),
            "Ret_short_%": round(ret_short, 2),
        })

    return pd.DataFrame(rows)

def weekly_summary(eval_df: pd.DataFrame) -> pd.DataFrame:
    if eval_df is None or eval_df.empty:
        return pd.DataFrame(columns=[
            "Semana","Sinais","Acertos","Taxa_Sucesso_%",
            "Sinais_Alta","Acertos_Alta","Sinais_Baixa","Acertos_Baixa"
        ])
    grp = eval_df.groupby("Semana")
    out = grp.agg(
        Sinais=("Atingiu_1pct","count"),
        Acertos=("Atingiu_1pct","sum"),
    ).reset_index()
    out["Taxa_Sucesso_%"] = (out["Acertos"] / out["Sinais"] * 100).round(2)

    # Quebra por direção
    by_dir = eval_df.groupby(["Semana","Direcao"])["Atingiu_1pct"].agg(["count","sum"]).reset_index()
    piv_cnt = by_dir.pivot(index="Semana", columns="Direcao", values="count").fillna(0).rename(
        columns={"ALTA":"Sinais_Alta","BAIXA":"Sinais_Baixa"}
    )
    piv_hit = by_dir.pivot(index="Semana", columns="Direcao", values="sum").fillna(0).rename(
        columns={"ALTA":"Acertos_Alta","BAIXA":"Acertos_Baixa"}
    )
    out = out.set_index("Semana").join(piv_cnt).join(piv_hit).fillna(0).reset_index()
    for c in ["Sinais","Acertos","Sinais_Alta","Acertos_Alta","Sinais_Baixa","Acertos_Baixa"]:
        out[c] = out[c].astype(int)
    return out.sort_values("Semana")

def overall_counts(eval_df: pd.DataFrame) -> dict:
    if eval_df is None or eval_df.empty:
        return {"total":0}
    n = len(eval_df)
    hits = int(eval_df["Atingiu_1pct"].sum())
    misses = n - hits
    hit_pct = round(100.0 * hits / n, 2)
    miss_pct = round(100.0 * misses / n, 2)
    bull = eval_df[eval_df["Direcao"]=="ALTA"]
    bear = eval_df[eval_df["Direcao"]=="BAIXA"]
    return {
        "total": n,
        "hits": hits,
        "misses": misses,
        "hit_pct_%": hit_pct,
        "miss_pct_%": miss_pct,
        "bull_total": len(bull),
        "bull_hits": int(bull["Atingiu_1pct"].sum()),
        "bear_total": len(bear),
        "bear_hits": int(bear["Atingiu_1pct"].sum()),
    }

def asset_return_last_4_weeks(df: pd.DataFrame) -> float | None:
    if len(df) < 21:
        return None
    last = float(df["Close"].iloc[-1])
    prev = float(df["Close"].shift(20).iloc[-1])
    if np.isnan(prev) or prev == 0.0:
        return None
    return round((last / prev - 1.0) * 100.0, 2)

# ---------- Plot ----------
def plot_chart(df: pd.DataFrame, eval_df: pd.DataFrame, ticker_in: str):
    fig = make_subplots(rows=1, cols=1)

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="Candles",
        increasing_line_color="green", increasing_fillcolor="green",
        decreasing_line_color="red", decreasing_fillcolor="red"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["EMA9"], mode="lines", name="EMA 9"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA21"], mode="lines", name="EMA 21"), row=1, col=1)

    if eval_df is not None and not eval_df.empty:
        succ = eval_df[eval_df["Atingiu_1pct"] == True]
        fail = eval_df[eval_df["Atingiu_1pct"] == False]
        if not succ.empty:
            fig.add_trace(go.Scatter(
                x=succ["Data_Sinal"], y=[df.loc[d, "Close"] for d in succ["Data_Sinal"]],
                mode="markers", name="Sinal (atingiu ≥1% na semana)",
                marker_symbol="star", marker_size=11
            ), row=1, col=1)
        if not fail.empty:
            fig.add_trace(go.Scatter(
                x=fail["Data_Sinal"], y=[df.loc[d, "Close"] for d in fail["Data_Sinal"]],
                mode="markers", name="Sinal (<1% na semana)",
                marker_symbol="x", marker_size=10
            ), row=1, col=1)

    fig.update_layout(
        title=f"{normalize_ticker_b3(ticker_in)} — Cruzamentos EMA9/EMA21 | Semana (5 pregões) alvo 1%",
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        margin=dict(l=20, r=20, t=50, b=20),
        height=720
    )
    fig.update_yaxes(title_text="Preço", row=1, col=1)
    return fig

def plot_weekly_success(weekly_df: pd.DataFrame):
    if weekly_df is None or weekly_df.empty:
        return None
    import plotly.express as px
    fig = px.bar(weekly_df, x="Semana", y="Taxa_Sucesso_%", hover_data=["Sinais","Acertos"],
                 title="Taxa de sucesso semanal (≥1% em 5 pregões)")
    return fig

# ---------- UI ----------
st.set_page_config(page_title="EMA9/21 — Semana 1% (B3)", layout="wide")
st.title("Cruzamentos EMA9/EMA21 — Sucesso se atingir 1% em 1 semana (5 pregões) — Últimos 12 meses")

with st.sidebar:
    st.markdown("### Regras fixas")
    st.write(f"""
- **Alta**: EMA9 cruza **para cima** EMA21 → sucesso se subir **≥ {THRESHOLD_PCT}%** em até **5 pregões**.
- **Baixa**: EMA9 cruza **para baixo** EMA21 → sucesso se cair **≥ {THRESHOLD_PCT}%** em até **5 pregões**.
- Saída no **primeiro toque** do alvo; se não tocar até D+5, marca como **não atingiu**.
- Avaliação e agrupamento **semanal** (sem opção de horizonte).
    """)
    st.caption("Sem volume / sem EMA50. Base diária, 12 meses.")

ticker_in = st.text_input("Ticker (ex.: ITUB4, PETR4, VALE3)", value="ITUB4")

if st.button("Analisar", type="primary"):
    with st.spinner("Baixando dados e calculando..."):
        df_raw = fetch_last_12m(ticker_in)
        if df_raw.empty:
            st.error("Não encontrei dados para esse ticker no período.")
        else:
            df = add_indicators(df_raw)

            eval_df = evaluate_weekly_threshold(df)
            weekly = weekly_summary(eval_df)
            overall = overall_counts(eval_df)
            ret4w = asset_return_last_4_weeks(df)

            fig = plot_chart(df, eval_df, ticker_in)
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Resumo geral (12 meses)")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total de operações", overall.get("total", 0))
            c2.metric("Acertos (≥1% na semana)", overall.get("hits", 0))
            c3.metric("Não atingiram 1%", overall.get("misses", 0))
            c4.metric("Taxa de sucesso geral (%)", overall.get("hit_pct_%", 0.0))

            d1, d2 = st.columns(2)
            d1.metric("ALTA — acertos/total", f"{overall.get('bull_hits',0)}/{overall.get('bull_total',0)}")
            d2.metric("BAIXA — acertos/total", f"{overall.get('bear_hits',0)}/{overall.get('bear_total',0)}")

            st.subheader("Percentual **semanal** de sucesso (sinais da semana)")
            if weekly.empty:
                st.info("Sem sinais suficientes para montar a série semanal.")
            else:
                figw = plot_weekly_success(weekly)
                if figw is not None:
                    st.plotly_chart(figw, use_container_width=True)
                st.dataframe(weekly, use_container_width=True)

            st.subheader("Rendimento total do ativo em 4 semanas (~20 pregões)")
            if ret4w is None:
                st.info("Série insuficiente para calcular 4 semanas.")
            else:
                st.metric("Retorno 4 semanas (%)", ret4w)

            st.subheader("Operações (detalhe)")
            if eval_df.empty:
                st.info("Nenhuma operação avaliada.")
            else:
                st.dataframe(eval_df.sort_values("Data_Sinal"), use_container_width=True)

st.caption("⚠️ Sinais pelo fechamento de D; alvo de 1% avaliado por HIGH/LOW até D+5. Retorno do ativo em 4 semanas usa 20 pregões.")