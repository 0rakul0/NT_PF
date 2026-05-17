# NT PF

Atlas analitico sobre noticias de operacoes da Policia Federal brasileira.

## Objetivo

O projeto implementa uma metodologia incremental autonomoma para organizar noticias publicas de operacoes da Policia Federal em temas canonicos, regras regex auditaveis e ciclos de aprendizado com LLM residual.

## Metodologia Atual

A execucao oficial usa:

1. Sincronizacao/geracao da base local de noticias.
2. Amostra inicial de 15% da base para fundacao, estratificada temporalmente.
3. Clusterizacao semantica da amostra.
4. Agente 1 como bifurcador de temas canonicos.
5. Agente 2 para gerar regex iniciais por tema.
6. Processamento dos 85% restantes em lotes incrementais.
7. Regex como primeira camada de classificacao.
8. LLM residual via OpenAI quando `PF_LLM_PROVIDER=openai`; fallback local com `llama3.2`.
9. Agente 3 para classificar residuais e registrar candidatos.
10. Agente Aprendiz de Regex para incorporar aprendizados automaticamente.
11. Graficos, metricas e README automatico por execucao.

A metodologia detalhada esta em [docs/arquitetura_treinamento_incremental.md](docs/arquitetura_treinamento_incremental.md).

O checkpoint da rodada final 15%/85% com OpenAI esta em [docs/resultado_rodada_final_15_openai.md](docs/resultado_rodada_final_15_openai.md).

## Como Rodar

Use apenas o arquivo:

```bat
rodar_sistema.bat
```

Ele nao exige argumentos. Basta clicar duas vezes ou executar no terminal.

O fluxo padrao:

- sincroniza a base;
- limpa artefatos anteriores;
- usa 15% para fundacao com fragmentos de diferentes momentos da base;
- usa 85% para execucao incremental;
- processa lotes de 10 noticias;
- gera clusters da amostra;
- registra metricas por lote;
- gera graficos e relatorios automaticos.

## Resultado da Rodada Final

Fechamento registrado em 2026-05-16:

- base total: 8106 noticias;
- fundacao: 1216 noticias, 15% da base;
- reserva incremental: 6890 noticias, 85% da base;
- clusters exploratorios: 24;
- temas canonicos aceitos pelo Agente 1: 17;
- regex iniciais aceitas pelo Agente 2: 322;
- lotes concluidos: 689;
- documentos processados na reserva: 6890;
- classificados por regex: 5661;
- enviados ao Agente 3/LLM residual: 1229;
- cobertura acumulada por regex: 82,1626%;
- regex incrementais incorporadas: 209;
- novos temas candidatos: 85;
- quarentenas do Agente 3: 64;
- erros de classificacao do Agente 3: 0.
- Agente Organizador da Arvore: 78 candidatos unicos avaliados, 13 absorvidos, 34 mantidos como folhas e 4 novos temas canonicos propostos.

Durante essa rodada foi incorporado um criterio metodologico novo: regex incrementais so entram no banco ativo quando possuem ancora de crime ou modus operandi da label canonica. Regex baseadas em nome de operacao, localidade, orgao ou termo operacional generico ficam em quarentena automatica.

Tambem foi corrigido o fluxo taxonomico: um agente separado, o Agente Organizador da Arvore, roda apos os lotes para olhar a arvore completa e decidir onde cada candidato se encaixa.

## Saidas

Os resultados ficam em:

```text
data/analise_qualitativa/incremental/
```

Principais arquivos:

- `README_METRICAS.md`: resumo automatico das metricas.
- `relatorio_execucao_metodologia.md`: relatorio da execucao.
- `metrics_batches.csv`: metricas por iteracao.
- `resumo_clusters_amostra.csv`: clusters gerados na amostra inicial.
- `temas_canonicos_agent1.json`: saida do Agente 1.
- `insumo_agente_organizador_arvore.json`: pacote completo visto pelo Agente Organizador da Arvore, com temas atuais, candidatos, contagens, evidencias, regex aprendidas e cosseno.
- `arvore_temas_agent1_refinada.json`: reorganizacao global dos temas candidatos pelo Agente Organizador da Arvore.
- `regex_iniciais_agent2.json`: saida auditavel do Agente 2.
- `regex_banco_agent2.json`: lista consumivel de regex iniciais aprovadas antes dos lotes.
- `data/analise_qualitativa/regex_classifier_rules.json`: banco ativo usado pelo classificador regex; comeca como copia do Agente 2 e recebe incorporacoes do Agente Aprendiz de Regex.
- `events.jsonl`: eventos completos da execucao.
- `figures/`: graficos gerados.

Os lotes ficam em:

```text
data/analise_qualitativa/lotes/
```

## Estrutura Principal

```text
NT_PF/
|-- rodar_sistema.bat
|-- rodar_sistema.py
|-- data/
|   |-- noticias_markdown/
|   |-- reference/
|   `-- analise_qualitativa/
|-- docs/
|   `-- arquitetura_treinamento_incremental.md
|-- scripts/
|   |-- incremental/
|   |   |-- run_all_incremental.py
|   |   |-- run_all_incremente.py
|   |   |-- amostragem.py
|   |   |-- clusterizacao_inicial.py
|   |   |-- agente1_temas.py
|   |   |-- agente2_regex_inicial.py
|   |   |-- processar_lotes.py
|   |   `-- relatorios.py
|   |-- agentes/
|   |-- schemas/
|   |-- tools/
|   |-- pf_incremental_methodology_run.py
|   |-- pf_operacoes_pipeline.py
|   |-- pf_llm_metadata.py
|   |-- pf_regex_classifier.py
|   `-- project_config.py
|-- pyproject.toml
`-- uv.lock
```

## Dependencias

O ambiente usa `uv` e Python `>=3.12,<3.14`.

As dependencias dos agentes LangChain ficam no grupo opcional `agents`, ja declarado no `pyproject.toml`.

## Configuracao LLM

Para usar OpenAI, crie/preencha o `.env`:

```text
PF_LLM_PROVIDER=openai
PF_OPENAI_API_KEY=sua_chave_openai_aqui
PF_OPENAI_MODEL=gpt-4.1-mini
```

Fallback local compativel com os estudos LangChain:

```text
PF_LLM_PROVIDER=ollama
PF_OLLAMA_MODEL=llama3.2
PF_OLLAMA_BASE_URL=http://localhost:11434
```
