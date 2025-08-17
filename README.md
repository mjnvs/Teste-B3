# EMA9/EMA21 — Só cruzamentos (sem volume) — v10

**Regras**
- **Alta**: EMA9 cruza **para cima** EMA21.
- **Baixa**: EMA9 cruza **para baixo** EMA21.
- **Sem confirmação por volume** (não usa Volume nem EMA50).
- **Ciclo**: de um cruzamento ao próximo (saída no próximo cruzamento).

**Análises**
- **Ciclos**: retorno (%) do início ao fim do ciclo e duração (barras).
- **Horizonte**: escolha **5/10/15/20** dias úteis para medir retorno & hit rate a partir de cada cruzamento.

**Gráfico**
- Candlestick diário (12 meses) com **EMA9/EMA21**, marcadores de cruzamentos, candles **verdes** (alta) e **vermelhos** (baixa).

## Rodar
```bash
pip install -r requirements.txt
streamlit run app.py
```