# Metodologia incremental com WNN e LLM para organização temática de textos institucionais

## Introdução

Bases textuais institucionais crescem continuamente, combinam vocabulário especializado, temas recorrentes e fenômenos emergentes. Em notícias da Polícia Federal, um mesmo documento pode mencionar crimes, modos de atuação, localidades, nomes de operações e órgãos envolvidos. Transformar esse fluxo em categorias temáticas estáveis é difícil devido à variação lexical, à ausência de rótulos completos e ao custo da revisão especializada.

A literatura demonstra o potencial de classificadores incrementais baseados em redes neurais sem pesos para tarefas textuais, especialmente pela rapidez de treinamento e pela capacidade de incorporar novos exemplos. Contudo, ainda há uma lacuna entre esses classificadores e as necessidades de organização de bases institucionais em evolução: não basta atribuir documentos a um conjunto fixo de rótulos. É necessário descobrir temas iniciais, controlar a criação de novas categorias, distinguir atributos substantivos de elementos acidentais — como localidades e nomes de operações — e tratar casos raros ou ambíguos sem desestabilizar a taxonomia.

Além disso, modelos de linguagem de grande porte (LLMs) ampliam a capacidade de interpretar exceções, mas sua aplicação indiscriminada a toda a base aumenta custo e reduz previsibilidade operacional. Assim, este trabalho busca responder à seguinte questão de pesquisa:

> Como organizar e classificar incrementalmente uma base textual institucional em crescimento, reduzindo o custo de inferência por LLM e preservando auditabilidade, adaptação a novos temas e estabilidade da taxonomia?

Para respondê-la, propõe-se uma metodologia híbrida e auditável que combina descoberta inicial de temas, memória WNN baseada em discriminadores, processamento incremental em lotes, revisão residual por LLM e reorganização controlada da árvore temática.

## Estratégia de evidências

A classificação concentra-se no `crime` canônico principal da notícia. O corpo do texto é a fonte prioritária, pois descreve o fato jurídico e suas evidências; o título é preservado como evidência secundária quando o corpo não ativa discriminadores suficientes. As tags institucionais da fonte são usadas como pistas auxiliares e só reforçam um crime já ativado por evidência textual, nunca criando uma classificação isoladamente. A origem de cada evidência é registrada, preservando a auditabilidade do processo. A identificação de *modus operandi* fica fora do escopo desta versão e pode ser tratada como trabalho futuro.

## Metodologia

A metodologia parte da base completa de notícias e aplica uma divisão temporalmente estratificada. Trinta por cento dos documentos formam a fundação temática: essa parcela é pré-processada, agrupada com HDBSCAN e consolidada por similaridade do cosseno. Os setenta por cento restantes constituem a reserva incremental, processada em lotes para avaliar a cobertura da memória WNN e a necessidade de revisão residual.

O fluxo é organizado em quatro funções numeradas, representadas no [diagrama metodológico no FigJam](https://www.figma.com/board/NEFlYExBMSeOXFmj0I6cWl):

1. **Consolidar temas canônicos.** A primeira função interpreta os clusters exploratórios, agrupa folhas semanticamente equivalentes e impede que localidades, nomes de operações ou órgãos sejam promovidos a categorias de crime.
2. **Gerar discriminadores de crime.** A segunda função transforma cada tema canônico em conjuntos pequenos de marcadores lexicais substantivos. Esses marcadores formam a memória WNN e permitem rastrear a evidência usada em cada decisão.
3. **Classificar incrementalmente.** Para cada notícia da reserva, o corpo pré-processado é projetado na memória binária. A WNN aceita um crime somente quando há evidência, confiança e margem suficientes; caso contrário, a notícia permanece residual.
4. **Revisar e aprender nos resíduos.** A quarta função analisa apenas os casos residuais, ambíguos ou semanticamente inéditos. A revisão pode confirmar um crime canônico, registrar uma notícia rara ou propor um novo tema. Marcadores aprovados retornam à memória, enquanto candidatos a tema seguem para governança da taxonomia.

Esse desenho limita o uso da LLM aos documentos para os quais a memória associativa não apresenta segurança suficiente. Assim, preserva-se uma camada determinística, auditável e de baixo custo para os crimes recorrentes, sem impedir a adaptação a novos padrões textuais.

### Política de crescimento da memória

A memória usa um vocabulário versionado de posições fixas `a₁, a₂, ..., aₙ`. Cada posição representa uma palavra discriminativa normalizada, e cada crime é definido por uma máscara vetorial contínua que aponta para as posições que o fundamentam. A imagem binária é apenas a visualização desse vetor como uma grade ou cartão perfurado: as quebras de linha não representam novas dimensões. Assim, o marcador composto `cartão perfurado` reutiliza as posições já existentes para `cartão` e `perfurado`, sem criar uma posição específica para a expressão inteira.

As posições são imutáveis durante a execução: uma palavra já mapeada mantém seu índice, e uma palavra aprovada posteriormente é acrescentada apenas ao final do vetor. Novas variantes oriundas da revisão residual permanecem registradas para auditoria, mas só recebem uma posição inédita após recorrência mínima de duas confirmações. Essa política evita que sinônimos ocasionais, erros de geração ou variações lexicais isoladas provoquem crescimento desordenado da memória, preservando a comparabilidade dos vetores anteriores por preenchimento de zeros à direita.
