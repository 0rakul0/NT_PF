# Dataset README

## Escopo

Este documento descreve o snapshot de dados gerado pelo projeto `NT PF` e serve
como base para um eventual deposito separado no Zenodo dedicado aos artefatos
derivados do pipeline.

## Estrategia recomendada

Nao misture o deposito de software com o deposito de dados.

- Deposito 1: software do projeto, com codigo-fonte, `LICENSE`,
  `CITATION.cff`, `.zenodo.json` e documentacao.
- Deposito 2: dataset derivado, com CSVs, relatorio narrativo e, se for o caso,
  subconjuntos publicaveis dos textos extraidos.

Essa separacao evita misturar licencas diferentes no mesmo registro e facilita
citacao, manutencao e versionamento.

## Snapshot local observado em 2026-04-10

- `data/pf_operacoes_index.csv`: 7918 registros de listagem.
- `data/pf_operacoes_conteudos.csv`: 7918 registros no manifesto.
- `data/noticias_markdown/`: 7916 arquivos markdown extraidos com sucesso.
- `data/analise_qualitativa/`: tabelas derivadas e relatorio analitico.
- Relatorio analitico atual: 7916 noticias analisadas, periodo de 2011-11-29 a
  2026-04-10, 32 clusters semanticos e 205 series recorrentes.

Os dois registros restantes no manifesto correspondem a falhas de extracao
pontuais. O pipeline analitico foi ajustado para consumir apenas registros com
`status=ok` e arquivo markdown existente.

## Arquivos principais

- `data/pf_operacoes_index.csv`: indice bruto estruturado a partir da listagem
  paginada da PF.
- `data/pf_operacoes_conteudos.csv`: manifesto de extracao com status, caminho
  do markdown e metadados extraidos da pagina individual.
- `data/noticias_markdown/*.md`: corpo principal extraido de cada noticia.
- `data/analise_qualitativa/corpus_enriquecido.csv`: tabela mestra derivada com
  data normalizada, cluster, serie semantica, crimes, modus e geografia.
- `data/analise_qualitativa/vizinhos_semelhantes.csv`: pares de noticias com
  proximidade semantica por similaridade do cosseno.
- `data/analise_qualitativa/pares_recorrentes.csv`: pares com alta similaridade
  e informacao temporal para leitura de recorrencia.
- `data/analise_qualitativa/resumo_clusters.csv`: resumo por cluster com termos,
  crimes e estados mais frequentes.
- `data/analise_qualitativa/series_semanticas.csv`: grupos de noticias
  recorrentes ao longo do tempo.
- `data/analise_qualitativa/analise_qualitativa.md`: relatorio narrativo do
  snapshot analisado.

## Dicionario resumido dos campos

### `pf_operacoes_index.csv`

- `offset`: deslocamento da pagina de origem na listagem.
- `page_number`: numero da pagina derivado do deslocamento.
- `item_index`: posicao da noticia dentro da pagina coletada.
- `categoria`: rotulo visual da noticia na listagem.
- `titulo`: titulo exibido na listagem.
- `subtitulo`: resumo curto exibido na listagem.
- `data_publicacao`, `hora_publicacao`, `publicado_em`: campos temporais
  capturados na listagem.
- `tipo_conteudo`: tipo textual exibido na listagem, como `Noticia`.
- `tags`: tags da listagem concatenadas com separador ` | `.
- `total_tags`: numero de tags identificadas.
- `link`: URL canonica da noticia.

### `pf_operacoes_conteudos.csv`

- `link`: URL canonica usada como chave de juncao.
- `markdown_path`: caminho local do markdown extraido.
- `status`: `ok` quando a extracao foi concluida e `error` quando houve falha.
- `titulo_extraido`, `subtitulo_extraido`: metadados lidos da pagina da noticia.
- `publicado_em_extraido`, `atualizado_em_extraido`: timestamps da pagina.
- `tags_extraidas`: tags da pagina individual.
- `erro`: mensagem de erro capturada na extracao, quando existir.

### `corpus_enriquecido.csv`

- `data_publicacao_dt`, `ano`, `mes`, `ano_mes`: normalizacao temporal.
- `nome_operacao`: nome inferido da operacao quando presente no titulo.
- `cluster_id`, `cluster_label`: agrupamento semantico da noticia.
- `semantic_series_id`: identificador da serie recorrente.
- `crime_labels`, `modus_labels`: rotulos heuristicas extraidos do texto.
- `ufs_mencionadas`, `estados_mencionados`: geografia detectada no conteudo.
- `texto_busca_normalizado`: texto consolidado e normalizado para busca.
- `markdown_path`: referencia ao texto bruto extraido.

## Proveniencia e metodo

- Fonte primaria: portal publico de noticias de operacoes da Policia Federal.
- Coleta: raspagem paginada da listagem de noticias e visita individual de cada
  link para extracao do corpo principal.
- Conversao textual: HTML principal convertido para markdown.
- Analise: vetorizacao TF-IDF, reducao de dimensionalidade, clusterizacao,
  vizinhanca semantica e heuristicas de rotulagem de crimes e modus operandi.

## Licenciamento e atribuicao

O portal de origem `gov.br/pf` informa licenciamento proprio para o conteudo
publicado no site. Antes de depositar os textos completos extraidos ou um
dataset que os redistribua, revise a licenca indicada pela fonte e verifique se
o tipo de reutilizacao pretendido esta coberto.

Como regra pratica:

- o software do projeto pode ser publicado separadamente com licenca MIT;
- os dados derivados devem citar explicitamente a fonte publica;
- a redistribuicao dos textos integrais extraidos merece revisao juridica ou,
  no minimo, uma decisao editorial consciente antes do deposito.

## Checklist para o deposito de dados

- Incluir este documento no registro do Zenodo.
- Informar a data do snapshot e a data da coleta.
- Informar a versao do software que gerou os artefatos.
- Descrever limites metodologicos e falhas conhecidas de extracao.
- Declarar claramente a licenca do dataset ou as restricoes aplicaveis.
