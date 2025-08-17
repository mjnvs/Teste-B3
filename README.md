# EMA9/EMA21 + Volume (EMA50) — Ciclos confirmados e retornos (v8)

**Regras**
- **Alta**: EMA9 cruza **para cima** EMA21.
- **Baixa**: EMA9 cruza **para baixo** EMA21.
- **Confirmação**: `Volume >= EMA50(Volume)` **no mesmo candle**.
- **Ciclo**: começa no cruzamento **confirmado** e termina no **próximo cruzamento** (saída não exige confirmação).

**Saídas do app**
- Candlestick (diário, 12 meses) com **EMA9/EMA21**, marcadores de cruzamentos **confirmados**.
- Gráfico de **Volume** + **EMA50(Volume)**.
- **Taxas de confirmação** (alta/baixa/total).
- Tabela de **ciclos confirmados** com **retorno (%)** do início ao fim e duração (barras).
- Resumo com nº de ciclos, retorno médio/mediano e por direção.

## Como rodar
```bash
pip install -r requirements.txt
streamlit run app.py
```