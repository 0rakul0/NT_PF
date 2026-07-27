# Comparações entre rodadas

Antes de uma nova execução completa, arquive aqui os artefatos que servirão como linha de base. A execução por `rodar_sistema.py` sincroniza a base de notícias por padrão e, depois, reinicia o diretório incremental para produzir uma nova rodada limpa.

Configuração-padrão atual:

- amostra temporalmente estratificada de 30% (`sample_fraction=0.30`);
- lotes de 100 documentos (`batch_size=100`);
- limiar de confiança de 0,70 para aceitar uma classificação WNN;
- corpo textual como evidência primária para crime e *modus operandi*;
- título como evidência secundária e tags como pistas auxiliares auditáveis.

Para executar uma rodada completa com sincronização da base, use `python -B rodar_sistema.py` a partir da raiz do projeto. Não defina `PF_SKIP_SYNC=true`.
