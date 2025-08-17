def fetch_last_12m(ticker_raw: str) -> pd.DataFrame:
    from datetime import datetime, timedelta
    from dateutil.relativedelta import relativedelta
    import pandas as pd
    import yfinance as yf

    def normalize_ticker_b3(t: str) -> str:
        t = (t or "").strip().upper()
        if not t:
            return ""
        if t.startswith("^") or t.endswith(".SA"):
            return t
        return f"{t}.SA"

    today = datetime.utcnow().date()
    start = today - relativedelta(years=1)
    end = today + timedelta(days=1)

    ticker = normalize_ticker_b3(ticker_raw)
    if not ticker:
        return pd.DataFrame()

    # group_by='column' tenta manter colunas simples,
    # mas em alguns casos ainda vem MultiIndex.
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

    # Se vier MultiIndex (ex.: ('Open','ITUB4.SA')), achata para um nível só.
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df.columns = df.columns.get_level_values(0)
        except Exception:
            df = df.droplevel(-1, axis=1)

    # Padroniza e mantém apenas as colunas necessárias
    df = df.rename(columns=lambda c: str(c).title())
    cols = ["Open", "High", "Low", "Close", "Volume"]
    df = df[[c for c in cols if c in df.columns]]

    # Garante dtype numérico nas colunas usadas
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Remove linhas sem dados essenciais
    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])

    # Índice datetime sem timezone
    df.index = pd.to_datetime(df.index).tz_localize(None)

    return df
