# Proposta de Artigo

## Titulo provisório

**Reducao progressiva de custo em classificacao textual incremental por meio de memoria WNN auditavel e revisao residual por LLM**

Alternativas:

1. **Classificacao incremental de noticias com memoria WNN e revisao residual por LLM**
2. **Uma arquitetura hibrida WNN+LLM para reduzir custo de classificacao textual ao longo do tempo**
3. **Memoria WNN auditavel para absorcao progressiva de classificacoes residuais por LLM**

## Problema de pesquisa

Modelos de linguagem de grande porte sao capazes de classificar textos complexos com boa flexibilidade semantica, mas seu uso continuo em grandes bases documentais impõe custo computacional e financeiro elevado. Em contextos institucionais com fluxo incremental de noticias, muitos casos sao recorrentes e poderiam ser resolvidos por uma camada deterministica mais barata, desde que esta conseguisse absorver gradualmente os padroes aprendidos nas revisoes anteriores.

## Pergunta de pesquisa

**E possivel reduzir progressivamente o custo de classificacao textual por LLM em um fluxo incremental de noticias, transferindo padroes recorrentes para uma memoria WNN auditavel sem perder utilidade operacional?**

## Hipotese

**A medida que exemplos revisados por LLM alimentam discriminadores reutilizaveis na memoria WNN, a cobertura da classificacao deterministica cresce, a taxa residual diminui e o custo medio por documento tende a cair ao longo das iteracoes.**

## Objetivo geral

Propor e avaliar uma arquitetura hibrida de classificacao textual incremental na qual uma memoria WNN absorve progressivamente padroes recorrentes inicialmente resolvidos por LLM, de modo a reduzir o custo marginal de classificacao ao longo do tempo.

## Objetivos especificos

1. Projetar uma camada WNN capaz de representar discriminadores binarios reutilizaveis para classificacao textual.
2. Separar a decisao principal de `crime` do eixo complementar de `modus_operandi`, preservando maior estabilidade no nucleo autonomo do sistema.
3. Encaminhar apenas casos ambiguos, novos ou insuficientemente cobertos para revisao residual por LLM.
4. Converter evidencias residuais em novos discriminadores capazes de ampliar a memoria WNN.
5. Medir a evolucao da cobertura da WNN, da taxa residual e do custo por lote ao longo da execucao incremental.

## Tese central do artigo

A LLM deve ser tratada como mecanismo inicial de descoberta e supervisao semantica, enquanto a WNN deve funcionar como memoria operacional de baixo custo para padroes recorrentes. Nesse arranjo, o custo mais alto concentra-se nas iteracoes iniciais, sendo amortizado progressivamente a medida que a memoria aprende e passa a resolver autonomamente uma parcela crescente dos documentos.

## Contribuicoes esperadas

1. Uma arquitetura hibrida WNN+LLM para classificacao textual incremental com foco em reducao progressiva de custo.
2. Um mecanismo de aprendizado residual em que decisoes da LLM sao convertidas em discriminadores reutilizaveis na memoria WNN.
3. Uma separacao operacional entre o eixo principal de `crime` e o eixo complementar de `modus_operandi`, aumentando a autonomia da camada deterministica.
4. Um modelo auditavel de memoria binaria versionada, no qual os padroes aprendidos permanecem rastreaveis ao longo das iteracoes.

## Resumo preliminar

Este trabalho propoe uma arquitetura hibrida para classificacao textual incremental de noticias institucionais, combinando uma memoria WNN auditavel com revisao residual por modelos de linguagem de grande porte. A proposta parte do pressuposto de que, em fluxos documentais extensos, a LLM e eficaz para resolver casos novos, ambiguos ou semanticamente ricos, mas seu uso continuo em toda a base torna o processo oneroso. Para enfrentar esse problema, o sistema aplica primeiro uma camada deterministica baseada em discriminadores binarios versionados, responsavel por atribuir autonomamente o crime canonico principal sempre que houver evidencia suficiente. Casos nao resolvidos pela WNN sao encaminhados a uma LLM, cuja decisao residual nao encerra apenas a classificacao do documento, mas tambem alimenta a expansao da memoria com novos discriminadores reutilizaveis. O eixo de `modus_operandi` e tratado como camada complementar, enriquecendo a saida final sem comprometer a estabilidade do nucleo de classificacao por crime. A hipotese central e que, com o crescimento incremental da memoria, a cobertura da WNN aumenta, a taxa residual diminui e o custo medio por documento cai ao longo das rodadas. O artigo descreve a arquitetura, o fluxo de aprendizado residual, os mecanismos de auditoria e um desenho experimental orientado por metricas de cobertura, residual, tokens consumidos e custo operacional.

## Estrutura sugerida do artigo

### 1. Introducao

Apresentar o problema de custo em classificacao textual com LLM, o contexto incremental de noticias, a necessidade de auditabilidade e a ideia de amortizacao progressiva do custo por meio de memoria WNN.

Fechar a introducao com:

1. problema
2. objetivo
3. hipotese
4. contribuicoes
5. organizacao do artigo

### 2. Fundamentacao teorica e trabalhos relacionados

Dividir em quatro blocos:

1. classificacao textual incremental
2. modelos de linguagem para revisao semantica
3. redes neurais sem peso e WiSARD/WNN
4. arquiteturas hibridas com camada deterministica e camada residual

Aqui voce pode usar os dois artigos-base para sustentar:

1. representacao binaria de entrada
2. discriminadores por categoria
3. aprendizado incremental
4. adequacao da WiSARD/WNN a tarefas linguisticas

### 3. Arquitetura proposta

Explicar o pipeline:

1. entrada documental
2. preprocessamento linguistico
3. projecao em memoria WNN
4. decisao autonoma de `crime`
5. extracao complementar de `modus_operandi`
6. revisao residual por LLM
7. aprendizado de novos discriminadores
8. atualizacao versionada da memoria

Pontos que devem aparecer claramente:

1. a WNN decide primeiro o `crime`
2. o `modus_operandi` qualifica a saida, mas nao deve derrubar sozinho um crime bem sustentado
3. a LLM atua apenas sobre residuos, ambiguidades ou novidades
4. cada revisao pode ampliar a memoria e reduzir custo futuro

### 4. Memoria WNN e representacao binaria

Descrever:

1. vocabulario discriminativo versionado
2. posicoes estaveis na memoria
3. crescimento incremental por anexacao a direita
4. comparabilidade entre vetores antigos e novos
5. grade visual como recurso de auditoria

Tambem explicar os dois eixos:

1. `canonical_label` como eixo principal e mais estavel
2. `modus_operandi` como eixo complementar e mais dinamico

### 5. Aprendizado residual por LLM

Descrever o pacote residual enviado a LLM:

1. texto semantico
2. saida parcial da WNN
3. discriminadores acionados
4. candidatos por similaridade
5. contexto de abstencao

Depois explicar como a saida residual e convertida em aprendizado:

1. classificacao em tema existente
2. reforco de discriminadores ja existentes
3. inclusao de novo discriminador
4. possivel criacao de tema candidato

### 6. Desenho experimental

Definir:

1. base documental
2. criterio de particionamento em lotes
3. configuracao inicial da memoria
4. condicoes de aceite da WNN
5. papel da LLM nos residuos

Métricas principais:

1. cobertura da WNN por lote
2. taxa residual por lote
3. quantidade de decisoes LLM por lote
4. tokens consumidos por lote
5. custo medio por documento
6. novos discriminadores aceitos por lote
7. proporcao de saidas com `modus_operandi`

### 7. Resultados e discussao

A discussao deve mostrar:

1. custo alto no inicio
2. crescimento da memoria ao longo das rodadas
3. aumento da cobertura WNN
4. queda do residual
5. reducao do custo medio por documento
6. limites da autonomia em casos raros ou ambiguos

Graficos recomendados:

1. WNN vs residual por lote
2. cobertura acumulada da WNN
3. tokens ou custo por lote
4. crescimento do numero de discriminadores
5. distribuicao final de crimes e `modus_operandi`

### 8. Limitacoes

Assumir explicitamente:

1. dependencia inicial de LLM
2. risco de erro em aprendizagem residual mal generalizada
3. necessidade de governanca sobre expansao da memoria
4. sensibilidade a mudanca de dominio
5. variabilidade maior do eixo de `modus_operandi`

### 9. Conclusao

Retomar a pergunta central e responder em termos operacionais:

1. a WNN absorve padroes recorrentes
2. a LLM fica reservada a casos mais caros e raros
3. o custo tende a cair com o amadurecimento da memoria
4. a arquitetura se mantem auditavel e incremental

## Frases fortes para usar no artigo

### Problema

O uso continuo de LLM em classificacao documental de larga escala oferece flexibilidade semantica, mas impõe custo operacional elevado quando aplicado indistintamente a casos recorrentes e a casos novos.

### Proposta

A arquitetura proposta desloca a LLM para o papel de supervisora residual e utiliza a WNN como memoria operacional de baixo custo para padroes recorrentes aprendidos ao longo do tempo.

### Hipotese economica

O custo elevado concentra-se nas iteracoes iniciais e tende a ser amortizado progressivamente a medida que a memoria WNN amplia sua cobertura sobre os casos recorrentes.

### Nucleo metodologico

O eixo de crime canonico sustenta a decisao autonoma principal, enquanto o eixo de modus operandi atua como complemento sem comprometer a estabilidade do nucleo deterministico.

## Possivel pergunta da banca ou de revisores

**Se a LLM e mais flexivel, por que nao usá-la em tudo?**

Resposta curta:

Porque a proposta nao busca substituir flexibilidade por rigidez, mas deslocar custo alto para onde ele realmente agrega valor: novidade, ambiguidade e excecao. Casos recorrentes devem ser absorvidos por uma memoria deterministica mais barata, auditavel e reutilizavel.

## Possivel fechamento de introducao

Este artigo defende que sistemas de classificacao textual incremental nao precisam escolher entre interpretabilidade e flexibilidade semantica. Ao combinar uma memoria WNN auditavel com revisao residual por LLM, torna-se possivel concentrar o custo alto no momento de descoberta, transferir conhecimento para uma camada deterministica reutilizavel e reduzir progressivamente o custo marginal de classificacao ao longo do tempo.
