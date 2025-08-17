# EMA9/EMA21 + Confirmação por Volume (EMA50) — v7

**Regras**
- **Alta**: EMA9 cruza **para cima** a EMA21.
- **Baixa**: EMA9 cruza **para baixo** a EMA21.
- **Confirmação**: `Volume >= EMA50(Volume)` **no mesmo candle** (tocou ou passou).
- Período: **últimos 12 meses**, base diária.
- Candlestick com **verde** (alta) e **vermelho** (baixa).

**Saída**
- Candlestick + EMAs
- Gráfico de **Volume** + **EMA50(Volume)**
- **Taxas de confirmação** (alta/baixa/total) e tabela de eventos.

## Como rodar
```bash
pip install -r requirements.txt
streamlit run app.py
```