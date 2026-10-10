## Resultados RAGAS

### Por pergunta

| ID | Categoria | Pergunta | Faith. 512 | Faith. 1024 | Relev. 512 | Relev. 1024 |
| --- | --- | --- | --- | --- | --- | --- |
| P1 | Orçamento | O que é um orçamento pessoal e para que ele serve? | 0.83 | 1.00 | 0.74 | 0.87 |
| P2 | Receitas e despesas | Qual a diferença entre despesas fixas e variáveis? | 1.00 | 1.00 | 0.95 | 0.89 |
| P3 | Educação financeira | Por que é importante ter educação financeira? | 1.00 | 1.00 | 0.99 | 0.93 |
| P4 | Planejamento financeiro | Quais são as etapas para fazer um planejamento financeiro pessoal? | 1.00 | 1.00 | 0.88 | 0.80 |
| P5 | Reserva financeira | Quanto dinheiro devo ter separado para imprevistos? | 1.00 | 1.00 | 0.78 | 0.58 |
| P6 | Receitas e despesas | O que fazer quando as despesas são maiores que a receita? | 1.00 | 1.00 | 0.65 | 0.00 |

### Médias por configuração

| Configuração | Faithfulness (média) | Answer relevancy (média) | Fonte esperada recuperada | Tamanho médio do contexto (chars) | Faithfulness ≥ 0,7? |
| --- | --- | --- | --- | --- | --- |
| chunk 512 | 0.97 | 0.83 | 83% | 399 | SIM |
| chunk 1024 | 1.00 | 0.68 | 100% | 900 | SIM |

### Interpretação

- **Faithfulness:** chunks de 512 obtiveram média 0.97 e chunks de 1024 média 1.00.
- **Answer relevancy:** chunks de 512 obtiveram média 0.83 e chunks de 1024 média 0.68.
- **Alvo de faithfulness ≥ 0.7:** chunk 512 ATINGIU o alvo (≥ 0.7) e está na zona ideal (≥ 0.9); chunk 1024 ATINGIU o alvo (≥ 0.7) e está na zona ideal (≥ 0.9).
- **Melhor configuração: chunks de 1024** (maior faithfulness média).
- **Por quê (hipótese a confirmar nas respostas):** chunks maiores entregam mais contexto contínuo ao LLM, o que tende a reduzir afirmações sem suporte; em contrapartida, trazem mais texto irrelevante junto. Contexto médio recuperado: 399 chars (512) x 900 chars (1024).
- Pior faithfulness no chunk 512: P1 (Orçamento) = 0.83. Vale ler essa resposta e os trechos recuperados.
- Pior faithfulness no chunk 1024: P1 (Orçamento) = 1.00. Vale ler essa resposta e os trechos recuperados.
- Limitações: 6 perguntas, k=3, e o juiz do RAGAS é o mesmo modelo que gera as respostas (gemma4:cloud), o que pode favorecer as notas.
