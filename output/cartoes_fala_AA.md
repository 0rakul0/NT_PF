# Cartões de fala - AA

Use um cartão por vez. Não precisa ler palavra por palavra: eles funcionam como apoio para manter a lógica da apresentação.

## Artigos apresentados

1. **Semi-Supervised Classification of Social Textual Data Using WiSARD** - Fabio Rangel, Fabricio Firmino, Priscila Machado Vieira Lima e Jonice Oliveira (2016).
2. **Theoretical Results on a Weightless Neural Classifier and Application to Computational Linguistics** - Hugo Cesar de Castro Carneiro (2017).

---

## Cartão 1 - Abertura e conexão

**Ideia principal:** os dois textos estudam redes neurais sem pesos aplicadas a texto, mas por ângulos diferentes.

**Fala sugerida:**

> Hoje eu vou apresentar dois trabalhos ligados à WiSARD, uma rede neural sem pesos baseada em memória. O primeiro propõe usar a WiSARD de forma semissupervisionada para classificar textos sociais. O segundo aprofunda a base teórica da arquitetura e mostra uma aplicação em linguística computacional. Juntos, eles ajudam a entender tanto a eficiência prática quanto a robustez do modelo.

**Transição:** “Vou começar pelo problema tratado no primeiro artigo.”

---

## Cartão 2 - AA1: problema

**Artigo:** *Semi-Supervised Classification of Social Textual Data Using WiSARD* (Rangel et al., 2016).

**Ideia principal:** texto de rede social é difícil de rotular e muda ao longo do tempo.

**Fala sugerida:**

> O primeiro artigo parte de um problema muito comum em redes sociais: há muito texto disponível, mas rotular manualmente cada postagem é caro. Além disso, os textos são curtos, têm gírias, hashtags, erros de escrita e vocabulário que muda rapidamente. Portanto, um modelo que dependa apenas de exemplos rotulados pode ficar desatualizado ou exigir muito trabalho humano.

**Ponto para lembrar:** dado não rotulado é abundante; rótulo é caro.

**Transição:** “Antes de treinar o classificador, os textos passam por uma etapa de preparação.”

---

## Cartão 3 - AA1: pré-processamento e limpeza

**Artigo:** *Semi-Supervised Classification of Social Textual Data Using WiSARD* (Rangel et al., 2016).

**Ideia principal:** a limpeza reduz ruído e transforma texto social em atributos mais úteis para a classificação.

**Fala sugerida:**

> O artigo não envia o texto bruto diretamente para a WiSARD. Primeiro, ele aplica uma etapa de pré-processamento. Os documentos são convertidos para letras minúsculas; a pontuação, os links e as menções de usuários do Twitter são removidos. Depois, é aplicado o stemming de Porter, que reduz palavras relacionadas a uma mesma raiz. Por exemplo, palavras com variações morfológicas passam a ter uma representação mais próxima. Isso ajuda a diminuir a variação superficial do vocabulário e a tornar a comparação entre textos mais consistente.

**Por que essa etapa importa:**

> Em redes sociais, links, menções e variações de escrita podem aparecer muitas vezes sem informar a classe do texto. A limpeza tenta fazer o classificador prestar mais atenção nos termos semanticamente relevantes.

**Cuidado ao falar:** o artigo descreve remoção de pontuação, links e menções; não afirma que hashtags foram removidas.

**Transição:** “Com o texto preparado, o artigo propõe aproveitar exemplos não rotulados de forma controlada.”

---

## Cartão 4 - AA1: o que é a SSW

**Artigo:** *Semi-Supervised Classification of Social Textual Data Using WiSARD* (Rangel et al., 2016).

**Ideia principal:** a Semi-Supervised WiSARD usa confiança para decidir quando aprender com um exemplo não rotulado.

**Fala sugerida:**

> A solução proposta é a Semi-Supervised WiSARD, ou SSW. Primeiro, a WiSARD aprende com os exemplos que já têm rótulo. Depois, ela prevê a classe dos textos não rotulados. Mas esses textos só voltam para o treinamento se a previsão tiver confiança suficiente. Assim, o modelo tenta ampliar seu conhecimento sem propagar indiscriminadamente os próprios erros.

**Definição simples:** WiSARD usa memórias RAM para reconhecer padrões, em vez de pesos ajustados por retropropagação.

**Transição:** “Essa confiança é o mecanismo que torna o autoaprendizado mais controlado.”

---

## Cartão 5 - AA1: experimento e resultado

**Artigo:** *Semi-Supervised Classification of Social Textual Data Using WiSARD* (Rangel et al., 2016).

**Ideia principal:** SSW não foi a mais acurada, mas treinou muito mais rápido.

**Fala sugerida:**

> Os autores comparam a SSW com S3VM e EM-NB em três bases de sentimento: duas do Twitter e uma do IMDB. A S3VM teve a melhor acurácia nos três conjuntos, então não seria correto dizer que a SSW venceu em qualidade preditiva. Mas a SSW teve o menor tempo de ajuste em todos os testes. No IMDB, por exemplo, ela treinou em aproximadamente 0,21 segundo, enquanto a S3VM levou cerca de 15 segundos e a EM-NB, mais de um minuto.

**Frase-chave:** “O ganho do artigo está no equilíbrio entre acurácia competitiva e velocidade de treinamento.”

**Transição:** “Isso é especialmente interessante quando os dados chegam continuamente, como em um fluxo de textos.”

---

## Cartão 6 - AA1: leitura crítica

**Artigo:** *Semi-Supervised Classification of Social Textual Data Using WiSARD* (Rangel et al., 2016).

**Ideia principal:** o artigo tem valor para fluxo de dados, mas reconhece limitações.

**Fala sugerida:**

> A principal contribuição do primeiro trabalho é mostrar que uma rede sem pesos pode ser uma alternativa eficiente quando o tempo de treinamento importa. Como limitação, a melhor acurácia ainda ficou com a S3VM. Os autores também apontam como trabalho futuro testar bases maiores de fluxo de dados e criar um mecanismo de esquecimento, porque os temas e o vocabulário podem mudar ao longo do tempo.

**Transição:** “O segundo trabalho ajuda justamente a entender melhor a robustez dessa família de classificadores.”

---

## Cartão 7 - AA2: escopo da tese

**Artigo:** *Theoretical Results on a Weightless Neural Classifier and Application to Computational Linguistics* (Carneiro, 2017).

**Ideia principal:** a tese tem uma frente prática em NLP e outra teórica.

**Fala sugerida:**

> O segundo texto é uma tese de doutorado e tem um escopo mais amplo. Ela segue duas frentes. A primeira aplica uma rede neural sem pesos à etiquetagem gramatical de palavras, chamada POS tagging. A segunda analisa teoricamente a WiSARD e a técnica de bleaching, buscando entender sua capacidade de generalização.

**Definição simples:** POS tagging é atribuir uma classe gramatical, como substantivo ou verbo, a cada palavra de uma frase.

**Transição:** “Primeiro, vou explicar a aplicação linguística.”

---

## Cartão 8 - AA2: mWANN-Tagger

**Artigo:** *Theoretical Results on a Weightless Neural Classifier and Application to Computational Linguistics* (Carneiro, 2017).

**Ideia principal:** uma boa representação textual melhora a classificação.

**Fala sugerida:**

> Na aplicação prática, o mWANN-Tagger classifica a função gramatical das palavras em vários idiomas. Para isso, ele usa a própria palavra, o contexto das palavras vizinhas e os sufixos. Os sufixos são importantes principalmente para palavras desconhecidas no treinamento. A tese mostra que uma configuração universal de parâmetros pode ter desempenho próximo dos melhores ajustes específicos por idioma, e que enriquecer a representação com prefixos e a etiqueta anterior melhora os resultados.

**Ponto para lembrar:** “A arquitetura importa, mas a forma de representar a entrada importa muito.”

**Transição:** “Além da aplicação, a tese também investiga por que o bleaching é importante.”

---

## Cartão 9 - AA2: bleaching

**Artigo:** *Theoretical Results on a Weightless Neural Classifier and Application to Computational Linguistics* (Carneiro, 2017).

**Ideia principal:** bleaching evita saturação e preserva a generalização teórica estudada.

**Fala sugerida:**

> Em uma WiSARD, muita informação acumulada pode saturar a memória e dificultar a distinção entre padrões. O bleaching é uma técnica que eleva o limiar de ativação para filtrar respostas fracas e resolver empates. A tese calcula a dimensão VC das variantes estudadas e conclui que o bleaching não reduz a capacidade teórica de generalização. Em termos simples: ele traz mais robustez operacional sem demonstrar uma perda teórica nessa capacidade.

**Evite dizer:** “bleaching aumenta a acurácia em qualquer situação”. O resultado teórico é sobre robustez e generalização, não uma garantia universal de acurácia.

**Transição:** “Com isso, os dois textos se conectam: um destaca eficiência em classificação textual; o outro explica aplicações e bases teóricas para tornar a arquitetura mais robusta.”

---

## Cartão 10 - Fechamento do AA

**Ideia principal:** WiSARD/WNN é interessante quando eficiência, aprendizagem incremental e explicabilidade importam.

**Fala sugerida:**

> Para concluir, o primeiro artigo mostra que a WiSARD semissupervisionada pode aprender rapidamente com poucos rótulos e muitos textos não rotulados. A tese mostra que redes sem pesos também podem ser aplicadas a tarefas linguísticas e estudadas formalmente, especialmente com a técnica de bleaching. A mensagem principal é que essas redes são uma alternativa relevante quando precisamos de aprendizado incremental, baixo custo de ajuste e decisões baseadas em padrões de memória mais explícitos.

## Perguntas que podem aparecer

**“A SSW foi melhor que a S3VM?”**  
Em acurácia, não. A S3VM foi melhor nos três conjuntos; a SSW se destacou no tempo de treinamento.

**“Bleaching é um tipo de treinamento?”**  
É uma técnica de decisão que ajusta o limiar das ativações, ajudando a filtrar respostas fracas e a lidar com saturação ou empates.

**“Qual é a relação entre os textos?”**  
O primeiro explora a WiSARD em classificação semissupervisionada de textos; o segundo amplia a base aplicada e teórica da família de classificadores sem pesos.
