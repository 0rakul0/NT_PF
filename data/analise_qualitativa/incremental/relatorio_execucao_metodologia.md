# Execucao da metodologia incremental

## Fundacao

- Base: 8600
- Amostra inicial: 860 (10%)
- Reserva incremental: 7740
- Clusters gerados: 20
- Clusters de ruido: 0
- Temas canonicos aceitos: 17
- Discriminadores WNN: 1048

## Lotes

- Iteracoes documentadas: 16
- Noticias nos lotes: 7740
- Capturadas por WNN: 4139
- Residuais apos WNN: 3601
- Processadas pela LLM residual: 3601
- Marcadores aprendidos para WNN: 923
- Candidatos compostos WNN: 2835
- Novos temas candidatos: 40
- Noticias raras promovidas a candidato: 40
- Noticias raras identificadas pelo Agente 3: 125
- Quarentenas tecnicas do Agente 3: 0
- Erros do Agente 3: 0
- Taxa WNN acumulada: 53.48%

## Interacoes

- lote_0001: docs=500, residual=284, wnn=216, llm=284, aprendizados=115, candidatos_compostos=192, taxa_wnn=43.20%
- lote_0002: docs=500, residual=276, wnn=224, llm=276, aprendizados=78, candidatos_compostos=202, taxa_wnn=44.80%
- lote_0003: docs=500, residual=296, wnn=204, llm=296, aprendizados=134, candidatos_compostos=182, taxa_wnn=40.80%
- lote_0004: docs=500, residual=297, wnn=203, llm=297, aprendizados=121, candidatos_compostos=198, taxa_wnn=40.60%
- lote_0005: docs=500, residual=244, wnn=256, llm=244, aprendizados=52, candidatos_compostos=190, taxa_wnn=51.20%
- lote_0006: docs=500, residual=276, wnn=224, llm=276, aprendizados=83, candidatos_compostos=196, taxa_wnn=44.80%
- lote_0007: docs=500, residual=288, wnn=212, llm=288, aprendizados=85, candidatos_compostos=147, taxa_wnn=42.40%
- lote_0008: docs=500, residual=174, wnn=326, llm=174, aprendizados=41, candidatos_compostos=140, taxa_wnn=65.20%
- lote_0009: docs=500, residual=180, wnn=320, llm=180, aprendizados=54, candidatos_compostos=197, taxa_wnn=64.00%
- lote_0010: docs=500, residual=174, wnn=326, llm=174, aprendizados=15, candidatos_compostos=175, taxa_wnn=65.20%
- lote_0011: docs=500, residual=189, wnn=311, llm=189, aprendizados=13, candidatos_compostos=176, taxa_wnn=62.20%
- lote_0012: docs=500, residual=189, wnn=311, llm=189, aprendizados=10, candidatos_compostos=178, taxa_wnn=62.20%
- lote_0013: docs=500, residual=211, wnn=289, llm=211, aprendizados=16, candidatos_compostos=179, taxa_wnn=57.80%
- lote_0014: docs=500, residual=205, wnn=295, llm=205, aprendizados=18, candidatos_compostos=178, taxa_wnn=59.00%
- lote_0015: docs=500, residual=223, wnn=277, llm=223, aprendizados=54, candidatos_compostos=203, taxa_wnn=55.40%
- lote_0016: docs=240, residual=95, wnn=145, llm=95, aprendizados=34, candidatos_compostos=102, taxa_wnn=60.42%

## Avaliacao WNN por tags da PF

- Referencia: tags criminais mapeadas de `x2`; elas nao entram na classificacao.
- Noticias com tag criminal mapeada: 2312
- Classificadas pela WNN: 1353
- Predicoes corretas: 1154
- Cobertura: 58.52%
- Precisao: 85.29%
- Recall: 49.91%
- F1: 62.97%
- Matriz de confusao: `D:\github\NT_PF\data\analise_qualitativa\incremental\avaliacao_crime_tags\matriz_confusao_crime_por_tags.csv`

## Graficos

- ![](data/analise_qualitativa/incremental/figures/wnn_residual_candidatos_por_iteracao.png)
- ![](data/analise_qualitativa/incremental/figures/taxas_wnn_llm_candidatos_por_iteracao.png)
- ![](data/analise_qualitativa/incremental/figures/linha_tempo_tipos_crime.png)

## Agente Organizador da Arvore

- Candidatos avaliados: 197
- Absorvidos por temas existentes: 191
- Promovidos a novos temas canonicos: 3
- Mantidos como folhas: 2
- Arvore refinada: `data/analise_qualitativa/incremental/arvore_temas_agent1_refinada.json`
