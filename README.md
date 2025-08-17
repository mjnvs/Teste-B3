# EMA9/EMA21 — Semana 1% (v14)

**Regras**
- **ALTA**: sucesso se subir **≥ 1%** em **até 5 pregões** após o cruzamento (usa HIGH).
- **BAIXA**: sucesso se cair **≥ 1%** em **até 5 pregões** (usa LOW).
- Sem volume/EMA50. Base diária, **últimos 12 meses**.
- Resultados **semanais** por semana do sinal.

**Saídas**
- Totais de acertos/erros (geral e por direção).
- **Taxa de sucesso semanal (%)** e tabela por semana.
- **Retorno do ativo em 4 semanas** (~20 pregões).
- Gráfico: Candlestick + EMA9/EMA21 com marcadores ★ (atingiu) e × (não).

## Rodar
```bash
pip install -r requirements.txt
streamlit run app.py
```