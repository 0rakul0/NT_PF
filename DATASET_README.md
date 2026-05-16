# Dataset README

## Escopo

Este documento descreve os dados usados e gerados pela metodologia incremental
do projeto `NT PF`.

## Entradas

- `data/pf_operacoes_index.csv`: indice estruturado da listagem publica da PF.
- `data/pf_operacoes_conteudos.csv`: manifesto de extracao das noticias.
- `data/noticias_markdown/*.md`: textos extraidos das noticias.

## Saidas Atuais

A execucao oficial grava os artefatos em:

- `data/analise_qualitativa/incremental/`
- `data/analise_qualitativa/lotes/`
- `data/analise_qualitativa/regex_classifier_rules.json`

Os principais artefatos esperados sao:

- `amostra_inicial.csv`
- `reserva_incremental.csv`
- `documentos_base.jsonl`
- `cluster_assignments_amostra.csv`
- `resumo_clusters_amostra.csv`
- `temas_canonicos_agent1.json`
- `regex_iniciais_agent2.json`
- `regex_banco_agent2.json`
- `metrics_batches.csv`
- `relatorio_execucao_metodologia.md`
- `run_manifest.json`
- `run_result.json`
- `events.jsonl`

## Metodo

A metodologia usa scripts encadeados. `amostragem.py` gera `documentos_base.jsonl`,
`amostra_inicial.csv` e `reserva_incremental.csv` com uma amostra de 3%,
estratificada por tempo para capturar fragmentos de diferentes momentos da base;
`clusterizacao_inicial.py` gera os clusters da amostra; `agente1_temas.py` gera
os nomes canonicos; `agente2_regex_inicial.py` gera as regex iniciais;
`processar_lotes.py` usa os 97% restantes em lotes. Cada lote passa primeiro
pelo banco ativo de regex. Esse banco comeca como a lista aprovada pelo Agente 2
em `regex_banco_agent2.json`; os residuos sao enviados para a LLM local e o
Agente 3 decide automaticamente quais aprendizados entram no arquivo ativo
`regex_classifier_rules.json`.

## Licenciamento

O portal de origem `gov.br/pf` possui licenciamento proprio para o conteudo
publicado. Antes de redistribuir textos integrais ou datasets derivados, revise
a licenca da fonte e mantenha atribuicao explicita.
