# Registro de alterações — WNN incremental

## Finalidade

Este documento registra as alterações introduzidas no ciclo de classificação incremental de notícias da Polícia Federal. Ele separa a entrada da retina, a referência de avaliação e as etapas que podem alterar a memória WNN.

## Contrato dos dados

| Campo | Papel no ciclo | Uso permitido |
|---|---|---|
| x1 = titulo | Auditoria e identificação | Não entra na retina, na WNN ou no prompt da LLM. |
| x2 = tags | Referência pós-decisão | Mapeia a taxonomia, calcula métricas e supervisiona o aprendizado; não entra na retina nem no prompt. |
| x3 = texto_noticia | Corpo documental | Única entrada textual da retina e do Agente 3. |
| x4 = data_noticia | Ordem temporal | Separa os 10% iniciais da fundação e os 90% da reserva incremental. |

## Retina e memória WNN

O texto x3 passa por normalização linguística, extração de tokens e expressões de domínio e projeção em vocabulário versionado. A retina produz um vetor binário: uma posição vale 1 quando o termo correspondente aparece no corpo da notícia e 0 caso contrário. As máscaras dos discriminadores operam sobre esse vetor e registram pontuação, cobertura, confiança, margem, posições ativadas e versão da memória.

O vocabulário é expansível somente por acréscimo. Uma nova palavra permanece pendente até obter duas confirmações recorrentes; posições já existentes não são reordenadas.

## Ciclo de decisão e aprendizado

1. A WNN classifica autonomamente quando há evidência, confiança e margem suficientes.
2. Casos residuais, ambíguos ou semanticamente inéditos seguem ao Agente 3.
3. O Agente 3 usa somente x3, propõe crime canônico, marcadores secundários, evidência e confiança.
4. O Agente 4 consulta x2 apenas depois da decisão:
   - confirma a aprendizagem se o rótulo coincidir com uma referência;
   - corrige o rótulo de aprendizagem quando existe uma única referência;
   - em caso multirrótulo, aplica desambiguação por pontuação WNN e margem;
   - bloqueia a atualização quando a evidência não permite separação segura.
5. Somente uma decisão autorizada pelo Agente 4 gera novos discriminadores e atualiza a memória reversa.

## Regras de classe e cobertura

- crime_organizado é classe pai estrutural. Quando x3 estabelece vínculo organizacional explícito, como facção, associação criminosa, coordenação, liderança, divisão de funções ou cadeia operacional, o delito concreto continua como crime principal e crime_organizado é registrado como marcador pai secundário. A métrica por classe considera esse marcador secundário para contabilizar a relação pai--filha; a métrica global de decisão principal permanece separada.
- crimes_contra_criancas recebeu padrões adicionais para divulgação de pornografia, imagens de abuso e violência sexual infantil.
- crimes_previdenciarios recebeu padrões específicos para fraude no INSS, benefício indevido, aposentadoria e pensão irregulares.
- corrupcao_desvio_recursos_publicos recebeu padrões para superfaturamento, contrato público, emendas e recursos federais desviados.
- contrabando_descaminho recebeu padrões para fraude aduaneira, cigarros de origem estrangeira, eletrônicos importados e mercadorias sem documentação.
- armas_municoes possui guarda contra menção incidental: em presença de máscara completa de crime contra crianças, ambiental ou trabalho escravo, a arma não assume o crime principal sem âncora de porte, posse, comércio, tráfico, fornecimento, armamento ou munição.

O refinamento curado é aditivo: preserva discriminadores existentes e cria um snapshot antes de alterar o banco.

Os limiares de confiança começam calibrados por classe: 50% para crimes contra crianças, contrabando/descaminho e corrupção/desvio de recursos públicos; 45% para crimes previdenciários e tráfico de drogas; e 65% para crime organizado. Após cada lote, a calibração dinâmica consulta $x_2$ somente pós-decisão e aplica, no máximo, dois pontos percentuais no limiar da classe para o lote seguinte. Com ao menos 10 referências, precisão de pelo menos 80% e revocação inferior a 30% reduzem o limiar; precisão abaixo de 80% o eleva. As faixas são 35--70% para classes usuais e 60--80% para crime organizado. As margens mínimas específicas são preservadas e cada mudança fica em `limiares_dinamicos_wnn.json`.

Quando a similaridade do cosseno aponta o mesmo tema líder da WNN e a notícia já possui evidência lexical suficiente, a decisão é aceita mesmo se confiança ou margem não atingirem o limiar. A decisão é registrada como `accepted_cosine_supported`; candidatos com divergência relevante entre WNN e similaridade permanecem residuais.

## Avaliação e painel

As métricas comparam decisões já produzidas com x2 mapeado. O painel apresenta:

- precisão, revocação, F1 e cobertura da WNN;
- qualidade da saída bruta do Agente 3;
- resultado final após validação do Agente 4;
- desempenho por máscara de crime, com referências x2, previsões WNN, acertos, precisão, revocação e F1;
- taxa de aceitação WNN, taxa residual/LLM, acurácia final, precisão final e diversidade por lote.

A métrica global da WNN mede a decisão principal autônoma. A métrica por classe considera o conjunto de rótulos emitidos, incluindo o marcador pai crime_organizado quando houver relação estrutural explícita; assim, ela mede corretamente a relação pai--filha sem confundi-la com a decisão principal.

## Sincronização da base

A sincronização consulta as páginas recentes da fonte e compara título normalizado, data e URL com a base local. Notícias já existentes são puladas; títulos ausentes da base são baixados e materializados antes da nova execução. A contagem exibida pelo sítio não é usada como critério exclusivo de parada.

## Organizador da árvore e realimentação métrica

Ao fim do rodar_sistema, o organizador:

1. consolida candidatos de temas e remapeia a taxonomia quando necessário;
2. compacta o banco de discriminadores;
3. lê os lotes já concluídos e aplica feedback pós-decisão ao banco.

No feedback métrico, uma máscara completa que contribuiu para uma decisão WNN correta recebe confirmação adicional. Um discriminador aprendido ou generalizado só é marcado como quarantined se acumular duas ou mais ativações incorretas e nenhum acerto. Regras curadas não são desativadas automaticamente. A WNN ignora itens em quarentena.

Cada aplicação cria um snapshot em data/analise_qualitativa/run_snapshots/ e grava o relatório data/analise_qualitativa/incremental/feedback_metricas_discriminadores.json. A realimentação afeta apenas execuções futuras, jamais a decisão histórica usada para as métricas.

## Operação

Padrões generalizados compostos apenas por vocabulário jurídico amplo, como associação criminosa, fraude, organização e documento falso, são rebaixados a contexto. Em uma competição não resolvida entre as duas classes líderes, a similaridade do cosseno pode escolher uma delas; essa decisão é registrada como `accepted_cosine_tiebreak`.

Um processo Python já iniciado mantém em memória o prompt, as regras e o código carregados no início. Para aplicar alterações de código, encerre o processo após terminar o lote atual e reinicie-o; com resume_batches=true, os lotes concluídos são pulados e o próximo lote pendente é retomado.

O rodar_sistema executa o organizador ao final da rodada. A execução intermediária por lote é opcional e depende de PF_THEME_TREE_REVIEW_INTERVAL_BATCHES.

## Verificação

Os testes cobrem: separação temporal, pré-processamento, retina binária, proibição de uso de tags/título como entrada, Agente 4, regras estruturais de crime organizado, supressão de armas incidentais e realimentação métrica do organizador.
