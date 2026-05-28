# Experimento WNN com discriminadores regex

Este diretorio contem um experimento independente de classificacao textual usando uma **rede neural sem peso** (WNN, *weightless neural network*) como memoria associativa.

A proposta e demonstrar, em pequena escala, como uma WNN pode classificar documentos a partir de discriminadores regex externos:

```text
documento
v
discriminadores regex
v
features binarias
v
memoria associativa WNN
v
decisao com votos, confianca e margem
v
se houver baixa confianca: nao classificado
```

Importante: **este experimento nao usa LLM residual**.

A WNN tem apenas dois comportamentos:

```text
1. classificar, quando a pontuacao for suficiente;
2. abster-se, quando a pontuacao for fraca ou ambigua.
```

O foco e **precisao, qualidade e auditabilidade**, nao cobertura maxima. Um documento sem sinal suficiente deve permanecer como `nao_classificado`, em vez de ser forcado para uma classe.

Para o artigo, a contribuicao central pode ser resumida assim:

> A WNN nao substitui a regex; ela usa discriminadores regex como sinais binarios e aprende combinacoes recorrentes, aceitando classificacao apenas quando ha confianca e margem suficientes.

## Arquivos

```text
wnn_experimento/
|-- README.md
|-- wnn_lake_regex_demo_pf.py
|-- wnn_unsupervised_signature_demo.py
|-- wnn_regex_bank_pf.json
|-- data_exemplo_noticias/
    |-- manifest.csv
    |-- noticias_markdown/
```

`wnn_lake_regex_demo_pf.py` e o motor supervisionado por pseudo-labels regex. Ele classifica em labels quando a decisao passa nos limiares.

`wnn_unsupervised_signature_demo.py` e o motor nao supervisionado por assinaturas. Ele nao usa labels no treino; memoriza assinaturas das 210 noticias de treino e busca vizinhos parecidos nas 90 noticias de teste.

`wnn_regex_bank_pf.json` contem os discriminadores regex por tema. Esse arquivo pode ser trocado para outro dominio.

`data_exemplo_noticias/manifest.csv` lista 300 noticias de exemplo e aponta para os arquivos Markdown copiados em `data_exemplo_noticias/noticias_markdown/`. Essa pasta fica dentro de `wnn_experimento` para facilitar publicar o experimento como um repositorio separado.

## Como rodar

Dentro da pasta do experimento:

```powershell
python wnn_lake_regex_demo_pf.py
```

Sem argumentos, o script ja usa:

- `data_exemplo_noticias/manifest.csv`
- `data_exemplo_noticias/noticias_markdown/`
- `wnn_regex_bank_pf.json`
- divisao 70% treino e 30% teste
- limiares conservadores para rotulagem e decisao

Para rodar a versao nao supervisionada:

```powershell
python wnn_unsupervised_signature_demo.py
```

Ela tambem nao recebe argumentos.

## Como funciona

### 1. Entrada de dados

O motor le um CSV de catalogo. No exemplo, o arquivo e:

```text
data_exemplo_noticias/manifest.csv
```

As colunas principais sao:

```text
status
markdown_path
titulo_extraido
subtitulo_extraido
tags_extraidas
link
```

A coluna `markdown_path` aponta para o texto completo da noticia.

### 2. Discriminadores regex

Os discriminadores ficam em JSON separado:

```json
{
  "trafico_drogas": [
    "trafic\\w+\\s+de\\s+drogasv",
    "\\bcocaina\\b|\\bmaconha\\b"
  ],
  "moeda_falsa": [
    "moeda\\s+falsa",
    "cedulasv\\s+falsasv"
  ]
}
```

Cada regex vira uma feature binaria auditavel. Se a regex bate no documento, a feature e ativada.

Exemplo:

```text
trafico_drogas::d01 = ativo
trafico_drogas::d02 = ativo
crime_organizado::d01 = inativo
```

### 3. Rotulagem fraca

Para fins de demonstracao, o proprio banco de regex cria rotulos de treino quando ha evidencia minima.

O script so aceita um rotulo fraco quando:

- o tema tem um numero minimo de hits;
- ha margem minima sobre o segundo tema;
- o documento nao esta excessivamente ambiguo.

Isso nao substitui validacao humana. E uma forma barata de construir uma memoria inicial.

### 4. Treino WNN

A WNN memoriza:

- features isoladas;
- pares de features;
- trios de features.

Pares e trios recebem mais peso na votacao porque representam combinacoes de evidencias, nao apenas uma palavra isolada.

### 5. Divisao treino/teste

Por padrao, as 300 noticias do manifest sao divididas antes da rotulagem fraca:

```text
210 noticias para treino
90 noticias para teste
```

Depois dessa divisao, a rotulagem fraca por regex e aplicada dentro de cada
grupo. A WNN so consegue treinar com noticias do conjunto de treino que recebem
um rotulo de referencia. Noticias sem sinal suficiente continuam no conjunto,
mas nao entram na memoria supervisionada.

Isso separa duas contagens:

```text
noticias no treino/teste: 210 / 90
noticias rotuladas para treinar/avaliar: depende dos discriminadores
```

### 6. Decisao por qualidade

Na inferencia, cada tema recebe votos. A decisao so e aceita se passar por limiares de:

- confianca;
- margem sobre o segundo tema;
- votos minimos.

Se a decisao nao passar nesses criterios, o documento e tratado como:

```text
nao_classificado
```

Isso e intencional. Para um sistema orientado a qualidade, e melhor recusar uma classificacao do que forcar uma resposta ruim.

### 7. Limiar usado

O script imprime os limiares usados na decisao:

```text
Limiar usado: {'confidence_min': 0.45, 'margin_min': 2, 'top_votes_min': 1}
```

Esses tres valores controlam quando a WNN aceita uma classificacao.

`confidence_min` e a proporcao minima dos votos que a classe vencedora precisa concentrar.

Exemplo:

```text
trafico_drogas: 12 votos
crime_organizado: 3 votos
lavagem_dinheiro: 1 voto
total: 16 votos

confianca = 12 / 16 = 0.75
```

Como `0.75 >= 0.45`, esse criterio passa.

`margin_min` e a diferenca minima entre a primeira e a segunda classe.

Exemplo:

```text
trafico_drogas: 12
crime_organizado: 3

margem = 12 - 3 = 9
```

Como `9 >= 2`, esse criterio passa.

Se fosse:

```text
trafico_drogas: 4
crime_organizado: 3

margem = 4 - 3 = 1
```

A decisao nao seria aceita, porque `1 < 2`.

`top_votes_min` e o minimo absoluto de votos que a classe vencedora precisa ter. Com `top_votes_min = 1`, basta o tema vencedor ter pelo menos um voto. Esse criterio evita aceitar uma decisao sem qualquer evidencia.

A WNN so classifica quando os tres criterios passam ao mesmo tempo:

```text
confianca >= confidence_min
margem >= margin_min
votos_do_vencedor >= top_votes_min
```

Caso contrario, o status e `nao_classificado`.

## Saida visual

O script imprime uma tabela com os principais campos para inspecao:

```text
# | titulo | referencia | decisao_wnn | top_score | conf | margem | status
```

Exemplo conceitual:

```text
1 | PF combate trafico... | trafico_drogas | trafico_drogas | trafico_drogas:12 | 1.000 | 12 | classificado
2 | PF apura caso ambiguo | crimes_ambientais | nao_classificado | crimes_ambientais:4 | 0.500 | 1 | abstencao
```

Campos:

- `referencia`: rotulo produzido pelas regex de referencia para avaliacao.
- `decisao_wnn`: decisao aceita pela WNN, ou `nao_classificado`.
- `top_score`: tema mais votado e sua pontuacao.
- `conf`: proporcao dos votos do tema vencedor sobre o total de votos.
- `margem`: diferenca entre o primeiro e o segundo tema.
- `status`: `classificado` ou `abstencao`.

## O que acontece com documentos sem discriminadores suficientes

Documentos sem sinal suficiente nao sao classificados.

Eles podem cair em tres situacoes:

```text
sem_sinal
sinal_fraco
ambiguo
```

Na versao atual, todos aparecem operacionalmente como `nao_classificado` quando a WNN nao atinge os limiares. Isso preserva a precisao das decisoes aceitas.

## Versao nao supervisionada por assinaturas

O arquivo `wnn_unsupervised_signature_demo.py` testa outra abordagem: usar a WNN como memoria de assinaturas, sem labels no treino.

Fluxo:

```text
300 noticias
v
split 70/30
v
210 noticias de treino entram na memoria
v
cada noticia vira uma assinatura de features regex
v
90 noticias de teste procuram vizinhos parecidos
v
saida: conhecido, novo ou sem_sinal
```

Essa abordagem usa todas as noticias do treino, mesmo quando nao ha classe de referencia. Ela nao aprende `trafico_drogas`, `crimes_ambientais` ou outra label supervisionada. Em vez disso, ela responde se a assinatura do documento de teste parece conhecida pela memoria.

As saidas sao:

```text
conhecido
novo
sem_sinal
```

`conhecido` significa que a noticia de teste encontrou pelo menos um vizinho de treino com similaridade suficiente.

`novo` significa que ha algum sinal, mas nenhum vizinho parecido o bastante.

`sem_sinal` significa que nenhum discriminador regex foi ativado.

A tabela visual dessa versao mostra:

```text
# | titulo_teste | status | tema_sugerido | features | sim_top | features_comuns | vizinho_mais_proximo
```

`tema_sugerido` nao e uma classe aprendida. E apenas o tema mais frequente entre os discriminadores ativados no proprio documento.

Exemplo de resultado da rodada atual:

```text
Noticias lidas: 300
Noticias no treino: 210
Noticias no teste: 90
Assinaturas memorizadas: 201
Noticias de teste com algum sinal: 86
Status no teste: {'conhecido': 85, 'sem_sinal': 4, 'novo': 1}
```

## Como adaptar para outro repositorio

O script nao recebe argumentos de linha de comando. Para usar outro conjunto de
dados ou outro banco de regex, edite a secao `Configuracao fixa do experimento`
no topo de `wnn_lake_regex_demo_pf.py`.

Os principais campos sao:

```python
CSV_PATH = ROOT / "data_exemplo_noticias" / "manifest.csv"
REGEX_BANK_PATH = SCRIPT_DIR / "wnn_regex_bank_pf.json"
PATH_COLUMN = "markdown_path"
TITLE_COLUMN = "titulo_extraido"
TRAIN_RATIO = 0.70
CONFIDENCE_MIN = 0.45
MARGIN_MIN = 2
TOP_VOTES_MIN = 1
EXAMPLES_TO_PRINT = 10
```

O banco de regex precisa manter o formato:

```json
{
  "label_canonica": [
    "regex_1",
    "regex_2"
  ]
}
```

## Como interpretar os resultados

O script imprime:

- quantidade de documentos lidos;
- quantidade de temas no banco regex;
- quantidade de discriminadores;
- quantidade de documentos rotulados por regex fraca;
- exemplos de treino e teste;
- exemplos rotulados usados no treino WNN;
- exemplos rotulados usados na avaliacao;
- proporcao treino/teste;
- cobertura da WNN;
- precisao entre decisoes aceitas;
- tabela visual com titulo, classificacao e pontuacao;
- exemplos nao classificados por baixa confianca.

As metricas sao calculadas contra o rotulo regex de referencia. Para medir qualidade final, recomenda-se validar uma amostra das decisoes aceitas por revisao humana.

## Papel metodologico

Este experimento nao substitui regex por WNN.

Ele demonstra uma camada associativa:

```text
discriminadores regex
v
features binarias
v
WNN associativa
v
classificado ou nao_classificado
```

A vantagem da WNN e memorizar combinacoes recorrentes de sinais com baixo custo e boa rastreabilidade. A limitacao e que ela nao entende semanticamente o texto; ela reconhece padroes ja observados.

Por isso, seu uso mais adequado e conservador: aceitar apenas decisoes de alta margem e deixar o restante como `nao_classificado`.

