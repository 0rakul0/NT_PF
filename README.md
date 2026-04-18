# NT PF

Atlas analitico sobre noticias de operacoes da Policia Federal brasileira. O projeto coleta a base publica de noticias, extrai o conteudo principal de cada pagina, organiza os dados em artefatos tabulares e abre um painel narrativo em Streamlit para explorar crimes, padroes temporais, clusters semanticos e clusters canonicos guiados por identidade.

## Fonte

Base publica consultada:

- [Noticias de operacoes da PF](https://www.gov.br/pf/pt-br/assuntos/noticias/noticias-operacoes?b_start:int=0)

## Objetivo

O foco do projeto e identificar quais tipos de crimes aparecem com maior recorrencia ao longo do tempo e observar correlacoes entre temas, contextos operacionais e distribuicao territorial. Um exemplo de pergunta que o trabalho tenta responder e como lavagem de dinheiro se relaciona com outros crimes em anos e contextos diferentes.

## O que o projeto faz

1. Coleta a listagem paginada de operacoes publicadas no portal da PF.
2. Estrutura os metadados basicos em CSV.
3. Abre cada noticia individualmente e extrai o conteudo principal em markdown.
4. Gera artefatos analiticos com classificacao, recorrencia temporal por cluster canonico, pares semelhantes e distribuicoes por ano.
5. Publica um painel em Streamlit para leitura exploratoria e narrativa dos resultados.

## Stack

- Python
- Pandas
- Streamlit
- Plotly
- scikit-learn
- BeautifulSoup
- requests
- docling

## Estrutura do repositorio

```text
NT_PF/
|-- .zenodo.json
|-- CITATION.cff
|-- DATASET_README.md
|-- LICENSE
|-- data/
|   |-- analise_qualitativa/   # saidas geradas pelo pipeline
|   |-- noticias_markdown/     # markdown de cada noticia extraida
|   `-- reference/
|       `-- brazil_states.geojson
|-- scripts/
|   |-- pf_operacoes_pipeline.py
|   `-- pf_analise_qualitativa.py
|-- streamlit_app.py
|-- requirements.txt
`-- .gitignore
```

## Reproducibilidade

- Ambiente validado com Python 3.13.12.
- As dependencias do projeto estao fixadas em [requirements.txt](requirements.txt) para facilitar a recriacao do ambiente.
- O manifesto `data/pf_operacoes_conteudos.csv` pode registrar extracoes com falha; a etapa de analise consome apenas registros com `status=ok` e markdown realmente disponivel.

## Como executar

Crie o ambiente virtual e instale as dependencias:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Se voce for levar o projeto para outra maquina, prefira usar o bootstrap local e o lockfile do ambiente validado, porque isso reduz bastante conflito de bibliotecas:

```powershell
.\setup_local.ps1
```

Esse script:

- procura um Python 3.13
- cria `.venv`
- atualiza `pip`, `setuptools` e `wheel`
- instala `requirements-lock.txt` quando ele existir
- roda `pip check` no final

Se der conflito em outra maquina, o primeiro ponto para conferir e a versao do Python. O ambiente deste projeto foi validado com `Python 3.13.12`, mas as dependencias principais usadas para analise local tambem aceitam `Python 3.12`.

### Modo leve para outra maquina

Se a outra maquina ja vai receber os arquivos locais prontos em `data/noticias_markdown`, `data/pf_operacoes_index.csv` e `data/pf_operacoes_conteudos.csv`, voce nao precisa instalar a pilha completa de coleta com `docling`.

Nesse caso, use o setup leve:

```powershell
.\setup_runtime.ps1
```

Ele instala apenas o necessario para:

- rodar `run_local.py`
- gerar os artefatos analiticos
- abrir `streamlit_app.py`
- usar a LLM local com `ollama`

Esse e o caminho mais indicado quando houver conflito de bibliotecas em outra maquina.

### Modo com extracao completa

Se a outra maquina tambem vai coletar e extrair noticias do portal, use o setup de extracao completa:

```powershell
.\setup_extraction.ps1
```

Esse fluxo instala em etapas:

- base de analise e dashboard
- cliente local do `ollama`
- pilha de extracao com `beautifulsoup4`, `requests`, `lxml` e `docling`
- par compativel `pydantic==2.12.5` + `pydantic_core==2.41.5` antes do restante

Essa instalacao em etapas costuma ser mais estavel em Python 3.12 do que tentar resolver tudo de uma vez com um unico `pip install -r requirements.txt`.

### Modo local sem argumentos

Se os arquivos locais ja estiverem presentes em `data/noticias_markdown`, `data/pf_operacoes_index.csv` e `data/pf_operacoes_conteudos.csv`, voce pode rodar o pipeline inteiro sem passar argumentos:

```powershell
.\.venv\Scripts\python.exe .\run_local.py
```

Ou, no PowerShell do Windows:

```powershell
.\run_local.ps1
```

Esse fluxo usa os caminhos padrao do repositorio, processa os arquivos locais com a LLM, gera os artefatos analiticos e deixa o painel pronto para abrir.

Se quiser testar sem processar toda a base, voce pode limitar a etapa da LLM sem usar argumentos de linha de comando:

```powershell
$env:PF_LLM_LIMIT="20"
.\.venv\Scripts\python.exe .\run_local.py
```

Se quiser pular a etapa da LLM e apenas regenerar os artefatos analiticos com o JSONL ja existente:

```powershell
$env:PF_SKIP_LLM="1"
.\.venv\Scripts\python.exe .\run_local.py
```

### Etapa 1: sincronizar a base automaticamente

```powershell
.\.venv\Scripts\python.exe .\scripts\pf_operacoes_pipeline.py sync --index-csv .\data\pf_operacoes_index.csv --content-csv .\data\pf_operacoes_conteudos.csv --markdown-dir .\data\noticias_markdown
```

Esse comando atualiza o indice e baixa apenas as noticias que ainda nao existem na base local.

Se quiser rodar as etapas separadamente, os comandos antigos `collect` e `extract` continuam disponiveis.

### Etapa 2: gerar a analise qualitativa

```powershell
.\.venv\Scripts\python.exe .\scripts\pf_analise_qualitativa.py --output-dir .\data\analise_qualitativa
```

### Etapa 2b: gerar metadados estruturados com LLM local

O projeto tambem pode ler cada noticia em markdown e separar duas camadas: `metadata_extraido`, lido diretamente do arquivo, e `inferencia_llm`, usada apenas para interpretar o corpo da noticia. O script `scripts/pf_llm_metadata.py` agora usa o cliente oficial `ollama`, com saida estruturada guiada por schema Pydantic.

```powershell
.\.venv\Scripts\python.exe .\scripts\pf_llm_metadata.py
```

Por padrao, ele usa estas constantes internas:

- `data/noticias_markdown` como entrada
- `data/analise_qualitativa/metadados_llm_noticias.jsonl` como JSONL
- `data/analise_qualitativa/metadados_llm_noticias.csv` como CSV tabular
- `gemma4:e4b` como modelo
- `http://localhost:11434` como host do Ollama

Se quiser limitar o processamento sem voltar ao uso de argumentos, basta definir a variavel de ambiente antes da execucao:

```powershell
$env:PF_LLM_LIMIT="5"
.\.venv\Scripts\python.exe .\scripts\pf_llm_metadata.py
```

Os contratos oficiais dessa saida ficam nas classes `NoticiaMetadataExtraido`, `NoticiaLLMInference` e `NoticiaEnriquecida`, definidas em `scripts/pf_llm_models.py`. A ideia e usar a LLM apenas para inferir identidade canonica, crimes mais presentes e modus operandi, deixando titulo, datas, tags e demais metadados estruturais como leitura direta do markdown.

Cada registro retorna uma estrutura como:

```text
Titulo: PF combate disseminacao de pornografia infantil pela internet
Data: 18/05/2019
Dateline: Manaus/AM.
Tags diretas: [Operacao PF, Destaque]
Operacao direta: Sem nome de operacao explicito
Identidade canonica: crime_abuso_sexual_infantil
Classificacao: Por crime
Crimes mais presentes: abuso_sexual_infantil
Modus operandi: atuacao_online, busca_apreensao
```

### Etapa 3: abrir o painel

```powershell
.\.venv\Scripts\python.exe -m streamlit run .\streamlit_app.py
```

## Artefatos gerados

- `data/pf_operacoes_index.csv`: indice estruturado com titulo, subtitulo, data, tags e link.
- `data/pf_operacoes_conteudos.csv`: manifesto com o status da extracao de cada noticia.
- `data/noticias_markdown/*.md`: texto principal de cada noticia convertido para markdown.
- `data/analise_qualitativa/`: tabelas analiticas e relatorio narrativo.
- `streamlit_app.py`: painel para leitura exploratoria dos resultados.

## Citacao e licenca

- O codigo deste projeto esta licenciado sob MIT. Veja [LICENSE](LICENSE).
- A citacao de software recomendada esta em [CITATION.cff](CITATION.cff).
- Os metadados iniciais para integracao com Zenodo estao em [.zenodo.json](.zenodo.json).

## Partes do painel Streamlit

O `streamlit_app.py` organiza a leitura em uma navegacao lateral com seis partes principais. Cada uma responde a um tipo diferente de pergunta sobre o acervo.

### Panorama

E a porta de entrada do painel. Resume o corpus com metricas gerais, introduz a historia analitica do projeto e mostra uma visao ampla do conjunto de noticias antes do mergulho por tema. Serve para responder perguntas como volume total, distribuicao geral e dimensao do acervo analisado.

### Crimes e Modus

Explora os crimes rotulados e os modos de operacao identificados no corpus. Esta parte mostra recorrencia por ano, comparacoes de sinais ao longo do tempo e leituras territoriais por estado. E a secao para entender quais crimes aparecem mais, como eles evoluem e onde ganham maior intensidade.

### Clusters

Agrupa noticias semanticamente parecidas. Aqui o painel mostra o tamanho de cada cluster, termos dominantes, crimes mais frequentes, linha do tempo, mapa por estados citados e uma rede 3D de proximidade entre clusters. Nessa rede, a similaridade entre um cluster e outro e calculada a partir do corpus textual agregado de cada cluster, ou seja, pela uniao dos textos das noticias que pertencem a ele. Quando um cluster aparece solto, isso nao significa erro automaticamente: indica apenas que, no limiar atual da rede, o corpus agregado dele nao encontrou conexoes fortes o suficiente com os demais clusters. Esta secao ajuda a enxergar blocos tematicos do acervo, em vez de olhar noticia por noticia.

### Tempo por Clusters Canonicos

Mostra identidades canonicas estaveis ao longo do tempo, como `crime_abuso_sexual_infantil` ou `crime_contrabando_descaminho`. O painel separa uma visao executiva e uma exploracao detalhada, com ranking de clusters canonicos, filtros por tipo, intensidade e periodo. E a parte usada para identificar repeticao, persistencia e condensacao tematica sem depender apenas de clustering nao supervisionado.

### Vizinhanca Semantica

Traz a leitura de caso. A partir de uma noticia-fonte, o painel recupera os vizinhos mais proximos por similaridade do cosseno, exibe o markdown extraido e mostra as noticias relacionadas. Serve para sair do agregado e voltar ao detalhe, inspecionando exemplos concretos de proximidade semantica.

### Artefatos

Funciona como inventario final do pipeline. Lista os principais arquivos produzidos pela analise e exibe o relatorio narrativo consolidado. Esta secao ajuda a conectar o painel visual com os artefatos tabulares e textuais gerados durante o processamento.

### Navegacao lateral

A barra lateral organiza o percurso sugerido de leitura do painel nesta ordem:

1. Panorama
2. Crimes e Modus
3. Clusters
4. Tempo por Clusters Canonicos
5. Vizinhanca Semantica
6. Artefatos

### Estado inicial sem dados

Quando os arquivos gerados pelo pipeline ainda nao existem, o app nao falha. Em vez disso, ele mostra uma tela de orientacao com os comandos necessarios para reconstruir os dados localmente antes de abrir o painel completo.

## Publicacao no Zenodo

Para uma publicacao mais limpa e reutilizavel, a recomendacao e separar dois registros:

1. `Software`: codigo-fonte, README, licenca MIT, citacao e metadados do Zenodo.
2. `Dataset`: CSVs e artefatos analiticos derivados, acompanhados de documentacao metodologica e dicionario dos arquivos.

O arquivo [DATASET_README.md](DATASET_README.md) descreve os artefatos e a estrategia recomendada para esse segundo deposito.

## Versao enxuta para GitHub e para o deposito de software

Este repositorio foi preparado para subir ao GitHub sem levar o volume inteiro de artefatos gerados localmente. Por isso, os seguintes caminhos ficam fora do versionamento:

- `data/pf_operacoes_index.csv`
- `data/pf_operacoes_conteudos.csv`
- `data/noticias_markdown/`
- `data/analise_qualitativa/*.csv`
- `data/analise_qualitativa/*.md`

O arquivo `data/reference/brazil_states.geojson` continua versionado porque e uma referencia estatica usada pelo mapa do painel.

Se voce clonar o projeto e abrir o app sem gerar os dados antes, o `streamlit_app.py` mostra uma tela de orientacao com os comandos necessarios para reconstruir os artefatos localmente.

## Observacoes

- O pipeline depende de acesso a rede para consultar o portal publico da PF.
- A primeira execucao pode demorar porque envolve raspagem, extracao textual e geracao de artefatos analiticos.
- O repositorio foi mantido enxuto de proposito para facilitar publicacao, clonagem e manutencao no GitHub.
- O portal de origem da PF informa licenciamento proprio para o conteudo publicado em `gov.br`; por isso, o deposito de software e o deposito de dados derivados devem ser tratados separadamente e com atribuicao explicita a fonte publica.
