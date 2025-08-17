# App de Análise Técnica (B3) — EMA9/EMA21 + Volume EMA50

Este mini-app em **Streamlit** permite digitar um **ticker da B3** (ex.: `ITUB4`, `PETR4`, `VALE3`) e analisa os **últimos 12 meses** com:
- **Candlestick** + **EMA 9** e **EMA 21**;
- **Volume** em barras + **EMA(Volume, 50)**;
- Detecção de **cruzamentos** (EMA9×EMA21) com **confirmação de tendência** quando **Volume > EMA50** no mesmo candle.

> Uso educacional. **Não é recomendação de investimento.**

## Como rodar localmente
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Como publicar 100% online (Streamlit Community Cloud)
1. Suba estes 3 arquivos (`app.py`, `requirements.txt`, `README.md`) em um **repositório no GitHub**.
2. Vá para [streamlit.io](https://streamlit.io) → **Community Cloud** → **New app** e selecione seu repositório/branch e o arquivo `app.py`.
3. Pronto! Você obterá um **link público** para compartilhar e usar no navegador.

## Observações
- Os dados são obtidos via **yfinance** (Yahoo Finance). Use tickers como `ITUB4`, `PETR4`, `VALE3` (o app adiciona `.SA` automaticamente).
- A janela de análise é fixada em **12 meses** a partir da data atual.