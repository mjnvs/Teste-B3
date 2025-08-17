# EMA9/EMA21 — Regra de 3% em H dias — v12

**Novidade**
- Colunas **Ret_long_%** e **Ret_short_%** para cada operação:
  - **Ret_long_%**  = (Saída/Entrada − 1) × 100
  - **Ret_short_%** = (Entrada/Saída − 1) × 100
- **Ret_%_direcao** mantém a leitura direcional (ALTA = long, BAIXA = short). Se bater o alvo, vale **3,00%** por definição.

**Regras**
- Cruzamento **EMA9×EMA21** (alta/baixa).
- Sucesso se tocar **±3%** a favor da direção em **H = 5/10/15/20** dias úteis.
- Saída no **primeiro toque**; senão usa **D+H**.

**Saídas**
- Totais de **acertivas** (≥3%) e **não atingiram** 3%, mais **percentuais**.
- Tabela detalhada com **Ret_long_%**, **Ret_short_%** e **Ret_%_direcao**.
- Gráfico candlestick + EMAs com marcadores de sucesso/fracasso.