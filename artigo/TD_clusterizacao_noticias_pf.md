# Metodologia autonoma de classificacao incremental de noticias da Policia Federal

## Resumo

Este documento formaliza uma metodologia autonoma, incremental e auditavel para classificar noticias publicas da Policia Federal por temas criminais e modus operandi. A proposta combina amostragem temporal estratificada, clusterizacao exploratoria, consolidacao semantica por similaridade do cosseno, agentes de linguagem com respostas estruturadas, geracao e validacao de expressoes regulares, classificacao residual por LLM e reorganizacao periodica da arvore tematica.

O objetivo metodologico e reduzir o custo de inferencia ao longo do tempo sem abrir mao de rastreabilidade. A LLM nao atua como classificador principal permanente; ela e acionada apenas quando o banco de regex nao encontra evidencia suficiente. Quando a LLM classifica um residual, esse caso pode produzir aprendizado reutilizavel para os lotes seguintes. Quando o caso ainda nao tem recorrencia, ele e registrado como `noticias_raras`, preservando memoria e evitando a criacao prematura de regex ruidoso.

Na execucao documentada, a base possuia 8.106 noticias. A metodologia utilizou 15% da base como amostra inicial estratificada no tempo, totalizando 1.216 noticias. Os 85% restantes, com 6.890 noticias, foram processados em 14 lotes de 500 noticias, com ultimo lote de 390. A etapa inicial gerou 24 clusters consolidados, 17 temas canonicos iniciais e 5.739 regex iniciais aceitas. Durante a execucao incremental, 6.527 noticias foram classificadas diretamente por regex, 363 seguiram para revisao residual por LLM, 51 regras foram aprendidas e 48 casos foram posteriormente reavaliados a partir da arvore refinada. Ao final, 41 desses casos foram absorvidos por macrotemas e 7 permaneceram como `noticias_raras`.

## 1. Problema

Bases textuais institucionais crescem continuamente. Em contextos como noticias da Policia Federal, os textos apresentam grande diversidade de temas, variacao lexical, repeticao de formatos, mudancas temporais de enfoque e presenca de elementos acidentais, como localidades, nomes de operacoes e entidades. Classificar manualmente esse volume e caro, lento e pouco escalavel. Classificar tudo com LLM tambem e custoso, alem de dificultar reproducibilidade quando nao ha uma camada deterministica de verificacao.

A metodologia proposta parte de quatro premissas:

1. A maior parte da base tende a ser composta por temas recorrentes.
2. Temas recorrentes podem ser capturados por regras regex bem ancoradas em crime, conduta ou modus operandi.
3. A LLM deve ser usada prioritariamente nos residuos, isto e, nos casos que escapam das regras.
4. Cada residuo deve deixar uma trilha auditavel e, quando possivel, transformar-se em aprendizado para reduzir chamadas futuras de LLM.

O resultado esperado nao e apenas uma classificacao pontual da base, mas um procedimento transferivel de treinamento incremental autonomo, sem intervencao humana no ciclo operacional.

## 2. Objetivo

O objetivo geral e construir um sistema de classificacao incremental, autonomo e transparente que:

- descubra uma fundacao tematica minima a partir de uma amostra temporal da base;
- gere temas canonicos por crimes e modus operandi, evitando localidades e entidades como categorias finais;
- crie regex iniciais para temas recorrentes;
- processe a massa restante em lotes;
- acione LLM apenas para residuos;
- aprenda novas regex a partir de casos residuais classificados;
- reorganize periodicamente a arvore de temas;
- trate casos sem recorrencia como `noticias_raras`, sem descartar sua memoria;
- documente metricas, evidencias, decisoes e graficos de cada rodada.

## 3. Trabalhos relacionados

A metodologia proposta dialoga com cinco linhas de pesquisa: supervisao fraca, modelagem de topicos, uso de LLMs como fontes de rotulagem, construcao automatica de taxonomias e bootstrapping de regras. Essas linhas mostram que o problema de reduzir custo de rotulagem e inferencia e recorrente, mas tambem evidenciam diferencas importantes em relacao ao desenho adotado neste trabalho.

Snorkel e a literatura de *data programming* tratam a rotulagem como um problema de supervisao fraca. Em vez de depender de uma base manualmente rotulada, especialistas escrevem funcoes de rotulagem que expressam heuristicas, regras, dicionarios ou sinais externos; depois, um modelo de labels combina essas fontes ruidosas. A semelhanca com este trabalho esta no uso de regras programaticas como mecanismo escalavel de classificacao. A diferenca principal e que aqui as regras nao sao apenas entradas para treinar um classificador posterior: elas permanecem como classificador deterministico operacional, versionado, auditavel e atualizado incrementalmente. Alem disso, as regras sao geradas e refinadas por agentes, sem intervencao humana no ciclo operacional.

Trabalhos de supervisao fraca orientada por ontologias tambem sao proximos, especialmente quando usam expressoes regulares ou regras como fontes de labels. Eles mostram que ontologias e bases de conhecimento podem produzir sinais fracos uteis em dominios especializados. A diferenca deste trabalho e que a taxonomia nao e assumida como completamente pronta: ela nasce de clusters, e depois e reorganizada por agentes quando residuos e candidatos novos aparecem.

A modelagem de topicos, especialmente abordagens como BERTopic, combina embeddings, clusterizacao e representacao de topicos por termos caracteristicos. Essa linha e semelhante a etapa de fundacao tematica, pois usa agrupamentos textuais para descobrir temas latentes. A diferenca e que este trabalho nao para na descoberta de topicos. Os clusters sao tratados como folhas exploratorias, nao como classes finais; depois sao interpretados por agentes, convertidos em temas canonicos criminais e finalmente operacionalizados por regex.

Pesquisas recentes usam LLMs como funcoes de rotulagem dentro de frameworks de supervisao fraca. Esse desenho se aproxima da decisao de acionar LLM apenas nos casos residuais. A diferenca e que aqui a LLM nao e usada para rotular toda a base nem para produzir apenas labels fracos destinados a treinar outro modelo. Ela atua como revisora de excecoes, gera evidencias estruturadas e pode produzir aprendizado convertido em regex para reduzir chamadas futuras.

Tambem ha trabalhos sobre LLMs gerando funcoes de rotulagem automaticamente. A semelhanca esta na tentativa de automatizar a criacao de regras. A diferenca e que este trabalho impoe uma cadeia de controle de dominio: o Agente 3 primeiro classifica ou cria candidato; o Agente Aprendiz gera regex; a regra precisa estar ancorada em crime ou modus operandi; e o Agente Organizador revisa a arvore global para evitar microtemas ou regras contaminadas por localidade e entidade.

Outra linha relacionada e a construcao automatica de taxonomias com LLMs, embeddings, palavras-chave e clusterizacao. Essa literatura se aproxima do Agente 1 e do Agente Organizador da Arvore. O diferencial deste trabalho e integrar taxonomia e classificacao incremental em um mesmo ciclo fechado: a arvore nao e apenas um produto final de organizacao conceitual, mas uma estrutura operacional que afeta regex, revisao residual, noticias raras e metricas de custo.

Por fim, tecnicas classicas de bootstrapping de padroes em extracao de informacao, como DIPRE e Snowball, partem de exemplos ou sementes para extrair padroes e encontrar novas instancias. A semelhanca esta no principio de transformar evidencias observadas em regras reutilizaveis. A diferenca e que este trabalho aplica esse principio a classificacao tematica incremental, com agentes especializados, memoria de noticias raras, validacao por regex e relatorios de cobertura por lote.

Assim, a contribuicao deste trabalho nao esta em propor isoladamente clusterizacao, regex, LLM ou supervisao fraca. O destaque esta na engenharia do ciclo completo: uma metodologia autonoma, sem intervencao humana operacional, que descobre temas em amostra temporal, gera temas canonicos, cria regex iniciais, classifica a massa em lotes, usa LLM apenas nos residuos, aprende novas regras, reorganiza a arvore e documenta custo, cobertura, evidencias e excecoes.

## 4. Metodologia incremental proposta

A metodologia proposta organiza a classificacao de noticias em um ciclo incremental, autonomo e auditavel. A ideia central e separar o trabalho caro, que exige interpretacao por modelo de linguagem, do trabalho recorrente, que pode ser resolvido por regex validadas. Assim, a LLM nao e usada como classificador permanente da base inteira; ela atua nos casos residuais, produz evidencias e alimenta regras reutilizaveis para as proximas rodadas.

O processo parte de uma base historica de noticias e a divide em duas massas. A primeira massa forma a fundacao tematica: uma amostra temporal estratificada usada para descobrir temas, folhas e padroes iniciais. A segunda massa forma a reserva incremental: o conjunto restante, processado em lotes, que testa a capacidade do banco de regex de classificar a base com baixo custo de inferencia.

![Conceito circular da metodologia incremental autonoma](media/figura-1-conceito-circular-metodologia.png)

A figura apresenta o conceito geral da metodologia. O ciclo comeca pela descoberta dos temas recorrentes, passa pela formalizacao desses temas em regex, processa novos lotes, envia apenas os residuos para revisao por LLM, aprende novas regras e reorganiza periodicamente a arvore tematica. O ponto essencial e que cada excecao classificada deve deixar uma trilha auditavel e, quando possivel, reduzir a chance de uma chamada futura de LLM para casos semelhantes.

### 4.1 Objetivo da classificacao e pontos de atencao

A classificacao tenta identificar o dominio criminal ou o modus operandi principal de cada noticia. O foco nao e classificar localidades, unidades da federacao, nomes de operacao, orgaos parceiros ou entidades ocasionais. Esses elementos podem ajudar na auditoria, mas nao devem virar tema canonico.

O classificador busca categorias como `trafico_drogas`, `crimes_contra_criancas`, `crime_organizado`, `corrupcao_desvio_recursos_publicos`, `crimes_ambientais`, `armas_municoes`, `falsificacao_documental`, `moeda_falsa` e outros temas criminais observados na base.

Os pontos frageis da metodologia exigem controle explicito:

- clusters podem agrupar noticias por lugar, instituicao ou forma textual, e nao por crime;
- regex muito especificas podem capturar apenas uma noticia e nao generalizar;
- regex muito amplas podem contaminar temas diferentes;
- nomes de operacao e localidades nao devem ser usados como ancora principal;
- casos raros nao devem virar tema definitivo antes de demonstrar recorrencia;
- candidatos novos precisam ser comparados com todos os temas existentes antes de serem promovidos.

### 4.2 Ingestao e divisao da base

A ingestao separa a base em duas partes. A primeira e uma amostra inicial estratificada no tempo, usada para montar a fundacao tematica. A segunda e a reserva incremental, usada para executar o ciclo de classificacao, aprendizado e medicao.

![Fundacao tematica a partir da amostra inicial](media/figura-2-fundacao-tematica.png)

A figura mostra a primeira parte da metodologia. A amostra temporal entra na etapa de texto de dominio criminal, depois passa por clusterizacao exploratoria e consolidacao por similaridade do cosseno. Em seguida, o Agente 1 ajusta os temas canonicos a partir dos clusters e o Agente 2 transforma esses temas em regex iniciais. O resultado dessa etapa e o primeiro banco de regex ativo.

Na execucao documentada, a base tinha 8.106 noticias. A amostra inicial correspondeu a 15% da base, com 1.216 noticias, e a reserva incremental ficou com 6.890 noticias. A estratificacao por ano evita que a fundacao seja dominada por um periodo especifico e aumenta a chance de capturar temas recorrentes em diferentes momentos da serie historica.

| Item | Valor |
|---|---:|
| Base total | 8.106 noticias |
| Amostra inicial | 1.216 noticias |
| Fracao da amostra | 15% |
| Reserva incremental | 6.890 noticias |
| Estratificacao | Ano |

A tabela apresenta a divisao operacional da base. Essa separacao sustenta a proposta de baixo custo: a amostra inicial e suficiente para criar a fundacao tematica, enquanto a maior parte da base e reservada para medir se o regex passa a dominar a classificacao.

### 4.3 Texto de dominio criminal, clusterizacao e similaridade

Antes da clusterizacao, cada noticia passa por uma etapa de normalizacao e reducao para texto de dominio criminal. Essa representacao prioriza titulo, subtitulo, tags, condutas, crimes, objetos ilicitos, modus operandi e trechos relevantes do corpo. Ao mesmo tempo, reduz o peso de localidades, nomes proprios, nomes de operacao, orgaos parceiros e termos administrativos genericos.

A clusterizacao tem papel exploratorio. Ela nao define a classe final da noticia. Sua funcao e organizar a amostra inicial em folhas semanticamente proximas para que o Agente 1 consiga enxergar a diversidade tematica. A metodologia e compativel com HDBSCAN; na execucao documentada, quando a densidade ficou instavel, foi acionado `minibatch_kmeans_fallback` como alternativa operacional de baixo custo. Em ambos os casos, a clusterizacao e apenas uma etapa de apoio.

Depois da clusterizacao bruta, a similaridade do cosseno consolida clusters semanticamente proximos. Essa etapa corrige fragmentacoes naturais: dois clusters diferentes podem representar folhas de um mesmo tema, como abuso sexual infantil, pornografia infantojuvenil e compartilhamento de material.

| Item | Valor |
|---|---:|
| Clusters brutos | 34 |
| Clusters consolidados | 24 |
| Grupos fundidos por cosseno | 5 |
| Clusters de ruido | 0 |

A tabela resume o efeito da consolidacao. A reducao de 34 clusters brutos para 24 consolidados indica que parte da separacao inicial era granular demais para virar tema final. A similaridade do cosseno atua como apoio analitico, nao como decisor unico.

![Principais grupos consolidados da amostra inicial](media/figura-2-clusters-fundacao.png)

A figura mostra os principais grupos consolidados da fundacao tematica. Ela deve ser lida como fotografia da amostra inicial: aponta onde havia massa tematica suficiente para orientar o Agente 1, mas nao substitui a decisao canonica dos agentes.

### 4.4 Agente 1: temas canonicos e arvore operacional

O Agente 1 recebe os clusters consolidados e decide como eles devem ser nomeados e agregados. Ele atua como bifurcador de temas: junta folhas que pertencem ao mesmo dominio criminal, separa subtemas quando existe identidade criminal distinta e impede que localidades, entidades ou nomes de operacao virem classes finais.

Esse agente precisa olhar o conjunto inteiro de temas disponiveis antes de decidir se uma folha e novo tema ou se deve ser absorvida por tema existente. Essa regra e importante porque a base e incremental: sem uma visao global, o sistema tenderia a acumular microtemas redundantes.

![Arvore operacional de temas canonicos e folhas de clusters](media/figura-6-arvore-operacional-temas-folhas.png)

A figura apresenta a arvore operacional produzida a partir dos clusters consolidados. A coluna da esquerda mostra temas canonicos; a coluna intermediaria mostra as folhas de cluster que alimentam cada tema; a coluna da direita resume termos dominantes usados como evidencia. Essa visualizacao deixa claro que o tema final nao e o cluster isolado, mas a agregacao analitica de folhas por familia criminal dominante.

### 4.5 Agente 2: geracao de regex iniciais

O Agente 2 recebe os temas canonicos do Agente 1 e as evidencias associadas a cada folha. Sua funcao e gerar regex iniciais suficientes para cobrir a diversidade observada dentro de cada tema. Nao ha limite artificial de regex por tema: a quantidade depende da variedade de condutas, objetos, termos e modos de operacao encontrados nas folhas.

As regex devem ser ancoradas em crime, conduta ou modus operandi. Elas nao devem depender de localidade, nome de operacao, orgao parceiro ou entidade acidental. Cada regra precisa guardar label, fonte, exemplos positivos, exemplos negativos quando disponiveis e justificativa minima de incorporacao.

| Item | Valor |
|---|---:|
| Regex iniciais aceitas | 5.739 |
| Padroes iniciais ativos apos consolidacao | 5.146 |

A tabela mostra a escala da cobertura inicial criada pelo Agente 2. O numero alto de regex e esperado porque cada tema pode ter varias folhas internas. O objetivo nao e criar uma regra generica demais, mas um banco deterministico amplo o bastante para capturar recorrencias sem chamar LLM.

### 4.6 Caminho da noticia na classificacao incremental

Depois da fundacao tematica, a reserva incremental e processada em lotes. Cada noticia passa primeiro pelo parser, que estrutura titulo, subtitulo, tags, data e corpo. Em seguida, o classificador regex tenta atribuir uma label. Quando a regex classifica acima do limiar, a decisao e registrada diretamente. Quando falha, a noticia segue para o Agente 3.

![Execucao incremental em lotes com classificacao regex-first](media/figura-3-execucao-incremental-lotes.png)

A figura apresenta o caminho operacional de cada noticia. O banco de regex aparece antes da LLM porque a proposta e `regex-first`: resolver deterministicamente o que ja foi aprendido e reservar inferencia para os residuos. Esse desenho permite medir, lote a lote, quanto da base foi absorvido por regra e quanto ainda depende de interpretacao por modelo.

Na execucao documentada, os lotes tinham 500 noticias, exceto o ultimo. A reserva incremental de 6.890 noticias foi processada em 14 lotes. No acumulado, 6.527 noticias foram classificadas por regex e 363 seguiram para LLM residual.

| Indicador | Valor |
|---|---:|
| Noticias na reserva incremental | 6.890 |
| Lotes processados | 14 |
| Tamanho medio dos lotes | 492,14 |
| Capturadas por regex | 6.527 |
| Residuais enviados a LLM | 363 |
| Taxa regex acumulada | 94,73% |
| Taxa residual LLM | 5,27% |
| Aprendizados por lote, em media | 3,64 |

A tabela resume os resultados operacionais sem detalhar cada lote individual. A taxa de 94,73% mostra que o classificador deterministico dominou a execucao; os 5,27% residuais concentraram o custo de LLM nos casos que realmente exigiam revisao.

### 4.7 Agente 3: revisao residual

O Agente 3 atua apenas quando a regex nao classifica uma noticia. Ele recebe o texto estruturado, as labels canonicas existentes, sugestoes por similaridade do cosseno, tags, titulo e evidencias textuais do corpo. Sua decisao segue tres caminhos:

1. classificar em tema canonico existente, quando houver encaixe defensavel;
2. criar `novo_tema_candidato`, quando houver subtema substantivo ainda nao coberto;
3. marcar como `noticias_raras`, quando nao houver encaixe nem recorrencia suficiente.

O Agente 3 nao deve criar microtema para toda excecao. Tambem nao deve forcar classificacao quando a noticia nao cabe nos temas disponiveis. Sua saida e estruturada por schema e precisa conter decisao, label, confianca, evidencia textual, justificativa e resumo curto.

### 4.8 Agente Aprendiz de Regex e noticias raras

O Agente Aprendiz de Regex recebe uma decisao residual e tenta transformar a evidencia em regra reutilizavel. Ele nao reclassifica a noticia; sua funcao e gerar regex candidata e validar se ela tem ancora em crime ou modus operandi. Uma regra so entra no banco ativo se capturar o caso positivo, evitar exemplos negativos e nao depender apenas de localidade, entidade ou nome de operacao.

![Aprendizado residual e reorganizacao da arvore tematica](media/figura-4-aprendizado-reorganizacao.png)

A figura mostra a parte adaptativa da metodologia. Quando o Agente 3 encontra tema canonico, o caso pode alimentar o Agente Aprendiz de Regex. Quando identifica subtema novo, gera candidato para a arvore. Quando nao ha encaixe, o caso entra como `noticias_raras` e recebe uma assinatura. Se a assinatura rara reaparece, ela deixa de ser apenas excecao e volta ao ciclo como candidato.

Na execucao documentada, 51 regras foram aprendidas no ciclo residual e permaneceram ativas no banco final.

| Item | Valor |
|---|---:|
| Regras aprendidas no ciclo residual | 51 |
| Padroes aprendidos ativos | 51 |
| Quarentenas reavaliadas | 48 |
| Reclassificadas para macrotemas | 41 |
| Mantidas como `noticias_raras` | 7 |
| `noticias_raras` no banco de regex | Nao |

A tabela mostra que o ciclo residual produziu aprendizado sem transformar `noticias_raras` em categoria criminal comum. Casos raros permanecem auditaveis, mas so geram regex quando apresentam recorrencia ou quando sao absorvidos por tema defensavel.

### 4.9 Agente Organizador da Arvore

O Agente Organizador da Arvore executa uma revisao global depois do processamento incremental. Ele recebe os temas canonicos atuais, candidatos criados pelo Agente 3, contagens, evidencias, regex aprendidas, sugestoes por similaridade do cosseno e o banco de regex ativo.

Sua responsabilidade e impedir que a taxonomia cresca por acumulacao desordenada. Ele decide se um candidato deve ser absorvido por tema existente, consolidado em macrotema, promovido a novo tema canonico, mantido como raro ou descartado como ruido.

Na execucao refinada, o organizador avaliou 55 candidatos, absorveu 19 em temas existentes e promoveu 6 macrotemas: `ameacas_e_terrorismo`, `crimes_contra_saude_publica`, `crimes_de_odio_e_extremismo`, `crimes_patrimoniais`, `falsificacao_documental` e `seguranca_privada_clandestina`.

### 4.10 Banco de regex, metricas e auditabilidade

O banco de regex e o classificador deterministico principal. Ele e versionado em JSON e registra label, fonte, exemplos, usos e padroes. Ao final da execucao documentada, o banco continha 23 classificadores ativos e 5.197 padroes regex ativos.

| Item | Valor |
|---|---:|
| Classificadores ativos | 23 |
| Padroes regex ativos | 5.197 |
| Padroes vindos do Agente 2 | 5.146 |
| Padroes aprendidos pelo residual | 51 |
| Labels ativas finais no banco | 23 |

A tabela descreve a composicao do banco deterministico. A maior parte dos padroes veio da fundacao tematica, enquanto o ciclo residual adicionou regras novas para reduzir chamadas futuras de LLM.

![Regex versus residual por iteracao](media/figura-3-regex-vs-residual.png)

A figura compara, por iteracao, quantas noticias foram resolvidas por regex e quantas precisaram de LLM residual. O contraste evidencia que o regex domina o fluxo operacional.

![Taxa regex por iteracao](media/figura-4-taxa-regex.png)

A figura mostra a estabilidade da taxa de classificacao por regex ao longo dos lotes. Ela permite acompanhar se o sistema perde cobertura em algum ponto ou se o aprendizado incremental mantem a classificacao deterministica em patamar elevado.

![Noticias por tema apos classificacao das noticias raras](media/figura-5-temas-finais.png)

A figura apresenta a distribuicao final das noticias por tema. Os maiores volumes ficaram em `trafico_drogas`, `crimes_contra_criancas`, `crime_organizado` e `corrupcao_desvio_recursos_publicos`, enquanto `noticias_raras` permaneceu residual, com 7 casos finais.

Cada etapa produz artefatos persistentes. Isso permite reconstruir a origem da amostra, os clusters, as decisoes dos agentes, as regras incorporadas, as metricas por lote e os casos raros.

| Artefato | Funcao |
|---|---|
| `documentos_base.jsonl` | Base estruturada usada na execucao |
| `amostra_inicial.csv` | Amostra temporal da fundacao |
| `reserva_incremental.csv` | Massa processada em lotes |
| `resumo_clusters_amostra.csv` | Resumo dos clusters da amostra |
| `temas_canonicos_agent1.json` | Temas iniciais do Agente 1 |
| `regex_iniciais_agent2.json` | Regex iniciais propostas |
| `regex_classifier_rules.json` | Banco ativo de regex |
| `metrics_batches.csv` | Metricas por lote |
| `events.jsonl` | Trilha completa de eventos |
| `temas_candidatos_agent3.jsonl` | Candidatos criados no residual |
| `arvore_temas_agent1_refinada.json` | Arvore refinada |
| `noticias_raras_observacoes.jsonl` | Memoria incremental de noticias raras |
| `classificacoes_incrementais_pos_quarentena.csv` | Saida final consolidada |

A tabela lista os artefatos de auditoria. Eles tornam o ciclo transparente: cada classificacao pode ser rastreada ate a regra, o agente, o lote ou a decisao residual que a produziu.

## 5. Criterios de qualidade

A metodologia avalia qualidade por criterios operacionais e epistemicos.

### 5.1 Custo

O custo e medido pela proporcao de noticias classificadas por regex antes de acionar LLM. Na execucao documentada, 94,73% da reserva incremental foi resolvida por regex.

### 5.2 Cobertura

Cobertura e a capacidade do banco de regex capturar temas recorrentes. A cobertura aumenta quando regex aprendidas entram no banco e passam a classificar casos futuros.

### 5.3 Precisao operacional

Precisao operacional e protegida por validadores que rejeitam regex ancoradas apenas em localidade, entidade, nome de operacao ou termo generico. O foco deve ser crime, conduta ou modus operandi.

### 5.4 Estabilidade taxonomica

A arvore de temas nao deve crescer por acumulacao desordenada de microtemas. O Agente Organizador da Arvore consolida candidatos e evita que cada excecao vire uma categoria.

### 5.5 Transparencia

Toda decisao relevante deve deixar evidencia textual, justificativa, label, fonte e arquivo de origem.

## 6. Limitacoes

A metodologia ainda possui limitacoes importantes.

Primeiro, a qualidade da fundacao depende da amostra inicial. Uma amostra pequena pode deixar de observar temas raros ou emergentes. A estratificacao temporal reduz esse risco, mas nao o elimina.

Segundo, clusterizacao nao equivale a categoria criminal. Clusters podem refletir formato textual, localidade ou termos institucionais. Por isso, a classificacao final depende dos agentes e dos validadores de dominio.

Terceiro, regex sao interpretaveis, mas podem gerar falso positivo se forem amplas demais. A validacao por exemplos negativos e a exigencia de ancora criminal mitigam esse risco.

Quarto, noticias raras exigem memoria incremental. Se forem ignoradas, o sistema perde aprendizado; se forem promovidas cedo demais, o banco fica ruidoso. Por isso, a assinatura rara recorrente e uma etapa central.

Quinto, os resultados dependem do modelo LLM disponivel e da qualidade dos schemas estruturados. O uso de fallback local ou remoto deve ser registrado em eventos.

## 7. Conclusao

A metodologia implementa um ciclo fechado de classificacao incremental autonomo. Ela usa uma amostra temporal para descobrir a fundacao tematica, clusterizacao e similaridade para organizar a diversidade inicial, agentes especializados para nomear temas e gerar regex, regex para classificar a maior parte da massa, LLM apenas para residuos e um mecanismo de aprendizado que converte excecoes recorrentes em regras reutilizaveis.

O resultado principal e a demonstracao de uma arquitetura de baixo custo e alta rastreabilidade: 94,73% da reserva incremental foi classificada por regex, enquanto 5,27% exigiu LLM residual. As noticias que nao se encaixaram imediatamente nao foram descartadas; foram convertidas em `noticias_raras`, com assinaturas auditaveis capazes de alimentar futuros candidatos quando houver recorrencia.

Essa abordagem e transferivel para outros dominios textuais em que haja grande volume, crescimento continuo, baixa disponibilidade de rotulagem humana, necessidade de transparencia e pressao por reducao de custo de inferencia.

## 8. Apendice: resultados por lote

| Lote | Noticias | Regex | Residual/LLM | Aprendizados | Taxa regex |
|---|---:|---:|---:|---:|---:|
| lote_0001 | 500 | 487 | 13 | 1 | 97,40% |
| lote_0002 | 500 | 469 | 31 | 4 | 93,80% |
| lote_0003 | 500 | 479 | 21 | 4 | 95,80% |
| lote_0004 | 500 | 486 | 14 | 3 | 97,20% |
| lote_0005 | 500 | 466 | 34 | 7 | 93,20% |
| lote_0006 | 500 | 474 | 26 | 3 | 94,80% |
| lote_0007 | 500 | 481 | 19 | 1 | 96,20% |
| lote_0008 | 500 | 470 | 30 | 4 | 94,00% |
| lote_0009 | 500 | 477 | 23 | 2 | 95,40% |
| lote_0010 | 500 | 462 | 38 | 4 | 92,40% |
| lote_0011 | 500 | 477 | 23 | 3 | 95,40% |
| lote_0012 | 500 | 464 | 36 | 2 | 92,80% |
| lote_0013 | 500 | 472 | 28 | 6 | 94,40% |
| lote_0014 | 390 | 363 | 27 | 7 | 93,08% |

A tabela apresenta o detalhamento por lote que foi resumido no corpo da metodologia. Ela fica no apendice para preservar a rastreabilidade dos resultados sem interromper a narrativa principal.

## 9. Referencias conceituais

- McInnes, L.; Healy, J.; Astels, S. HDBSCAN: Hierarchical density based clustering. Journal of Open Source Software, 2017.
- Reimers, N.; Gurevych, I. Sentence-BERT: Sentence embeddings using Siamese BERT-networks. EMNLP-IJCNLP, 2019.
- Ratner, A. et al. Snorkel: Rapid Training Data Creation with Weak Supervision. arXiv:1711.10160, 2017. https://arxiv.org/abs/1711.10160
- Ratner, A. et al. Snorkel DryBell: A Case Study in Deploying Weak Supervision at Industrial Scale. arXiv:1812.00417, 2018. https://arxiv.org/abs/1812.00417
- Fries, J. A. et al. Ontology-driven weak supervision for clinical entity classification in electronic health records. Nature Communications, 2021. https://www.nature.com/articles/s41467-021-22328-4
- Blei, D. M.; Ng, A. Y.; Jordan, M. I. Latent Dirichlet Allocation. Journal of Machine Learning Research, 2003.
- Grootendorst, M. BERTopic: Neural topic modeling with a class-based TF-IDF procedure. arXiv:2203.05794, 2022. https://arxiv.org/abs/2203.05794
- Smith, R. et al. Language Models in the Loop: Incorporating Prompting into Weak Supervision. arXiv:2205.02318, 2022. https://arxiv.org/abs/2205.02318
- Guan, N.; Chen, K.; Koudas, N. Can Large Language Models Design Accurate Label Functions? arXiv:2311.00739, 2023. https://arxiv.org/abs/2311.00739
- Huang, C.; He, G. Text Clustering as Classification with LLMs. arXiv:2410.00927, 2024. https://huggingface.co/papers/2410.00927
- Balakrishnan, A. Automated Taxonomy Construction Using Large Language Models: A Comparative Study of Fine-Tuning and Prompt Engineering. Information, 2025. https://www.mdpi.com/2673-4117/6/11/283
- Agichtein, E.; Gravano, L. Snowball: Extracting Relations from Large Plain-Text Collections. ACM Digital Library, 2000.
