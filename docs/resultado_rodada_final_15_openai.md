# Resultado da rodada final 15%/85% com OpenAI

Fechamento: 2026-05-16.

Este documento registra a rodada final da metodologia incremental com amostra de fundacao de 15% e reserva incremental de 85%. A execucao usa OpenAI como provedor principal e `llama3.2` como fallback local, mantendo o desenho de baixo custo: regex primeiro, LLM apenas no residual e incorporacao automatica de aprendizado quando houver regra auditavel.

## Configuracao

- Base total: 8106 noticias.
- Amostra inicial: 1216 noticias.
- Reserva incremental: 6890 noticias.
- Criterio da amostra: estratificacao temporal por ano.
- Lote incremental: 10 noticias.
- Provedor principal: OpenAI.
- Modelo principal: `gpt-4.1-mini`.
- Fallback local: `llama3.2`.
- Regex iniciais por tema: sem limite artificial baixo; alvo operacional de 30 por tema quando houver evidencia suficiente.
- Interferencia humana no ciclo: zero.

## Fundacao

- Clusterizacao da amostra: 24 clusters exploratorios.
- Algoritmo usado nesta execucao: fallback `minibatch_kmeans_fallback`.
- Agente 1: 17 temas canonicos aceitos.
- Agente 2: 322 regex iniciais aceitas.

Arquivos principais:

- `data/analise_qualitativa/incremental/amostragem_result.json`
- `data/analise_qualitativa/incremental/clusterizacao_result.json`
- `data/analise_qualitativa/incremental/temas_canonicos_agent1.json`
- `data/analise_qualitativa/incremental/regex_iniciais_agent2.json`
- `data/analise_qualitativa/incremental/regex_banco_agent2.json`
- `data/analise_qualitativa/regex_classifier_rules.json`

## Resultado final dos lotes

A execucao processou toda a reserva incremental: `689` lotes e `6890` noticias.

- Documentos processados: 6890 de 6890.
- Classificados por regex: 5661.
- Enviados ao Agente 3/LLM residual: 1229.
- Cobertura acumulada por regex: 82,1626%.
- Regex incrementais incorporadas: 209.
- Novos temas candidatos: 85.
- Quarentenas do Agente 3: 64.
- Erros de classificacao do Agente 3: 0.

## Evolucao por blocos de 50 lotes

| Ate o lote | Docs no bloco | Regex | LLM residual | Taxa regex | Regex aprendidas |
|---:|---:|---:|---:|---:|---:|
| 50 | 500 | 379 | 121 | 75,80% | 21 |
| 100 | 500 | 389 | 111 | 77,80% | 22 |
| 150 | 500 | 430 | 70 | 86,00% | 16 |
| 200 | 500 | 428 | 72 | 85,60% | 21 |
| 250 | 500 | 411 | 89 | 82,20% | 19 |
| 300 | 500 | 418 | 82 | 83,60% | 7 |
| 350 | 500 | 434 | 66 | 86,80% | 14 |
| 400 | 500 | 404 | 96 | 80,80% | 13 |
| 450 | 500 | 418 | 82 | 83,60% | 11 |
| 500 | 500 | 386 | 114 | 77,20% | 23 |
| 550 | 500 | 400 | 100 | 80,00% | 14 |
| 600 | 500 | 419 | 81 | 83,80% | 8 |
| 650 | 500 | 421 | 79 | 84,20% | 11 |
| 689 | 390 | 324 | 66 | 83,08% | 9 |

## Achado metodologico: qualidade do regex incremental

Durante a rodada, foi observado que algumas regex sugeridas apos revisao residual estavam usando pistas operacionais em vez de crime ou modus operandi. Exemplos de pistas inadequadas:

- nome da operacao;
- localidade;
- orgao publico;
- unidade administrativa;
- termos genericos de acao policial.

Esse comportamento poderia aumentar a cobertura aparente, mas com risco de falso positivo e perda de explicabilidade.

Decisao metodologica adotada:

- o Agente 3 classifica o residual;
- o Agente 2 so incorpora regex se houver ancora substantiva de crime ou modus operandi da label canonica;
- regras sem essa ancora entram em quarentena automatica;
- a cobertura passa a ser menos inflada e mais auditavel.

Exemplos de ancoras aceitas:

- `arma` + `fogo` + `ilegal` para `armas_municoes`;
- `radio` + `clandestina` para `radiodifusao_clandestina`;
- `vantagem` + `indevida` para `corrupcao_desvio_recursos_publicos`;
- `extracao` + `madeira` para `crimes_ambientais`;
- `quadrilha` + `roubo` para `crime_organizado`, quando o contexto sustenta organizacao criminosa.

## Achado operacional: retomada segura

A execucao foi interrompida uma vez, no lote 372, por um erro de I/O no Windows durante a escrita de `data/analise_qualitativa/regex_classifier_rules.json`. Apos a correcao, a rodada foi retomada e concluiu os 689 lotes.

Correcao aplicada:

- a compactacao do banco de regex deixou de escrever diretamente no arquivo final;
- agora escreve em arquivo temporario;
- em seguida substitui atomicamente o arquivo final.

Script de retomada criado e usado:

- `scripts/resume_final_15_llm.py`

Esse script retoma apenas `processar_lotes`, usa `resume_batches=True` e preserva os lotes ja registrados em `metrics_batches.csv`.

## Relatorios e graficos gerados

- `data/analise_qualitativa/incremental/README_METRICAS.md`
- `data/analise_qualitativa/incremental/relatorio_execucao_metodologia.md`
- `data/analise_qualitativa/incremental/metrics_batches.csv`
- `data/analise_qualitativa/incremental/insumo_agente_organizador_arvore.json`
- `data/analise_qualitativa/incremental/arvore_temas_agent1_refinada.json`
- `data/analise_qualitativa/incremental/figures/regex_vs_residual_por_iteracao.png`
- `data/analise_qualitativa/incremental/figures/taxa_regex_por_iteracao.png`
- `data/analise_qualitativa/incremental/figures/clusters_3d_amostra.html`

## Organizacao global dos temas candidatos

Apos os lotes, o Agente Organizador da Arvore recebeu os 17 temas canonicos ativos e os 78 temas candidatos unicos gerados pelo Agente 3. Essa etapa e obrigatoria para evitar duplicidade taxonomica sem alterar a funcao original do Agente 1 de fundacao.

O insumo da reorganizacao inclui os temas canonicos atuais, os candidatos do Agente 3, contagem por candidato, evidencias, regex aprendidas e sugestoes por similaridade do cosseno.

Resultado da reorganizacao:

- Candidatos avaliados: 78.
- Candidatos absorvidos por temas existentes: 13.
- Candidatos mantidos como folhas: 34.
- Candidatos promovidos: 31 ocorrencias.
- Novos temas canonicos unicos propostos: 4.

Novos temas canonicos propostos:

- `crimes_contra_saude_publica`
- `crimes_de_odio_internet`
- `falsificacao_documental`
- `seguranca_privada_clandestina`

## Leitura dos resultados

O resultado final sustenta a hipotese principal da metodologia:

- a fundacao semantica em 15% da base gerou taxonomia e regex suficientes para classificar a maior parte dos lotes seguintes;
- a LLM foi acionada apenas em residuos;
- os residuos classificados geraram aprendizado incremental;
- a taxa por regex fechou em 82,1626% no acumulado final;
- novos temas candidatos apareceram sem precisar de intervencao humana.

O ponto mais importante nao e apenas a cobertura. O principal ganho metodologico e separar:

- descoberta semantica inicial;
- regras deterministicas auditaveis;
- classificacao residual;
- aprendizado incremental;
- quarentena automatica;
- maturacao futura de novas folhas em temas canonicos.
