# EMA9/EMA21 — Regra de 3% em H dias (v11)

**Regra**
- Sinal: cruzamento **EMA9×EMA21** (alta/baixa).
- Sucesso: dentro de **H = 5/10/15/20** dias úteis após o sinal, o preço **atinge** **±3%** a favor da direção.
  - ALTA: usa **HIGH** para detectar toque em **+3%**.
  - BAIXA: usa **LOW** para detectar toque em **−3%**.
  - Saída no **primeiro toque**; se não tocar até D+H, marca como não atingiu (usa o fechamento de D+H para referência).
- **Sem** volume e **sem** EMA50.

**Saídas**
- **Nº de operações** acertivas (≥3%) e **não atingiram 3%**.
- **Percentuais** de ≥3% e <3%.
- Tabela detalhada (data do sinal, direção, preço de entrada, H, atingiu?, data/ preço de saída, barras até o evento, retorno no evento).
- Gráfico: candlestick (diário, 12 meses) + EMA9/EMA21, marcadores de sucesso (★) e falha (×).

## Rodar
```bash
pip install -r requirements.txt
streamlit run app.py
```