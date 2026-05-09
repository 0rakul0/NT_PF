# Artigo

Esta pasta contem um esqueleto LaTeX para transformar a metodologia do projeto em artigo.

## Arquivos

- `main.tex`: texto principal do artigo, com a pipeline completa.
- `references.bib`: bibliografia inicial sobre clusterizacao semantica, consenso de clusters, embeddings e uso de LLMs para codificacao/classificacao textual.
- `figures/hdbscan_ruido.png`: figura de diagnostico do HDBSCAN, gerada a partir de `figures/hdbscan_ruido_por_representacao.csv`.

## Como compilar

Com uma distribuicao LaTeX instalada:

```powershell
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Se as citacoes aparecerem como `?`, isso quase sempre significa que o BibTeX ainda nao foi executado ou que o PDF foi recompilado apenas uma vez depois dele. A sequencia acima precisa ser rodada dentro da pasta `artigo/`, porque o arquivo principal espera encontrar `references.bib` no mesmo diretorio.

No Overleaf, deixe o compilador em `pdfLaTeX` e clique em `Recompile from scratch` se as referencias ficarem presas em cache.

## Relacao com o pipeline

O artigo descreve a arquitetura atual do projeto em treze etapas: coleta, extracao, limpeza, metadados, classificacao regex, classificacao LLM, resumo controlado, aprendizado continuo de regras, clusterizacao em multiplas representacoes, consenso com K-means, hierarquico e HDBSCAN, clusters canonicos, series semanticas e producao dos artefatos analiticos.

O texto tambem inclui o algoritmo de reducao de custo: cada chamada LLM bem-sucedida pode gerar regras candidatas; regras validas entram na proxima execucao; com isso, parte dos casos antes caros passa a ser resolvida localmente por regex.

O resumo controlado entra como uma camada experimental complementar: ele nao substitui o texto integral, mas permite comparar se uma representacao mais padronizada melhora a estabilidade e a interpretabilidade dos agrupamentos.

## Exemplo de classificacao

O artigo e o TD incluem dois exemplos reais do corpus. Um caso de crimes ambientais e resolvido diretamente por regex, com confianca 0,95 e sem chamada a LLM. Outro caso, sobre comercio ilegal de medicamentos, fica abaixo do limiar regex e e encaminhado a LLM com schema fechado, retornando categoria canonica, modus operandi, resumo curto e evidencia textual. A comparacao mostra a contribuicao central do metodo: a LLM entra como camada complementar e geradora de aprendizado, nao como classificador unico de toda a base.

## Figuras

A figura do HDBSCAN pode ser regenerada com:

```powershell
python ..\scripts\pf_generate_artigo_figures.py
```

Ela usa os resultados consolidados em `figures/hdbscan_ruido_por_representacao.csv` e escreve `figures/hdbscan_ruido.png`.
