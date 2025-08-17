# EMA9/EMA21 + Volume (EMA50) — v9
- **Confirmação** no candle do cruzamento: `Volume ≥ EMA50(Volume)`
- **Ciclos**: de entrada confirmada até o **próximo cruzamento**
- **Avaliação por horizonte**: selecione **5/10/15/20** dias úteis para medir retornos e hit rate das **entradas confirmadas**
- Gráficos: **Candlestick** diário (12 meses) com **EMA9/EMA21**, marcadores dos sinais confirmados, **Volume + EMA50**

## Rodar
```bash
pip install -r requirements.txt
streamlit run app.py
```

> Uso educacional. Fechamento como preço de avaliação; sem custos/slippage.