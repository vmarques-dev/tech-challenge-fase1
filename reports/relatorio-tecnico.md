# Relatório Técnico - Tech Challenge Fase 1

**Projeto:** Sistema de apoio ao diagnóstico de câncer de mama com Machine Learning
**Dataset:** Wisconsin Diagnostic Breast Cancer (WDBC)

---

## 1. Contexto e objetivo

Um hospital universitário busca um sistema inteligente de suporte ao diagnóstico capaz de acelerar a triagem de exames e apoiar as decisões clínicas. Nesta primeira fase, o foco é a **classificação de exames com dados estruturados**: a partir de medidas extraídas de imagens digitalizadas de massas mamárias, prever se um tumor é **maligno** ou **benigno**.

O objetivo não é substituir o(a) médico(a), mas oferecer uma **ferramenta de apoio** que destaque automaticamente os casos de maior risco, priorizando a triagem. A decisão final permanece sempre com o profissional de saúde.

## 2. Os dados

- **Fonte:** [Breast Cancer Wisconsin (Diagnostic)](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data).
- **Dimensões:** 569 amostras × 33 colunas originais.
- **Alvo:** `diagnosis` — `M` (maligno) ou `B` (benigno).
- **Features:** 30 medidas numéricas dos núcleos celulares (raio, textura, perímetro, área, suavidade, compacidade, concavidade, pontos côncavos, simetria e dimensão fractal), cada uma em três variações: **média** (`_mean`), **erro padrão** (`_se`) e **pior valor** (`_worst`).

### Distribuição das classes

As classes estão **moderadamente desbalanceadas**: ~63% benignas e ~37% malignas. Esse desbalanceamento é tratado com *split* estratificado e influencia diretamente a escolha das métricas de avaliação (ver Seção 5).

## 3. Estratégias de pré-processamento

Todo o pré-processamento está em `notebooks/01-eda.ipynb` e foi pensado para produzir uma base limpa, sem *data leakage* e adequada à modelagem.

1. **Remoção de colunas irrelevantes.**
   - `id`: identificador único do paciente, sem valor preditivo.
   - `Unnamed: 32`: coluna-artefato totalmente vazia (`NaN`), gerada na exportação do CSV.

2. **Verificação de valores ausentes e inconsistências.** Após a remoção dessas colunas, não restaram valores ausentes - as 30 features numéricas estão completas, dispensando imputação.

3. **Codificação da variável-alvo.** O diagnóstico foi mapeado para valores numéricos definindo **maligno como classe positiva (1)** e benigno como (0). Essa escolha não é arbitrária: a classe positiva é a de maior interesse clínico, e métricas como *recall* e *precision* são reportadas em relação a ela.
   - `M` (maligno) → `1`
   - `B` (benigno) → `0`

4. **Análise de correlação.** A matriz de correlação revelou **forte multicolinearidade** entre as features de tamanho (`radius`, `perimeter`, `area`), que descrevem essencialmente o mesmo conceito físico - o tamanho do núcleo. As features de "pior valor" (`_worst`) apresentaram a maior correlação marginal com o diagnóstico.

5. **Padronização (StandardScaler).** Aplicada na etapa de modelagem, com um cuidado essencial: o scaler é ajustado **apenas no conjunto de treino** (`fit_transform`) e somente aplicado ao teste (`transform`). Isso evita *data leakage* - o conjunto de teste não influencia os parâmetros (média e desvio) da transformação. A padronização é necessária para a Regressão Logística (sensível à escala) e dispensável para a Árvore de Decisão (invariante à escala).

A base limpa e codificada é exportada para `data/processed/wdbc_processed.csv`, consumida pelo notebook de modelagem.

## 4. Modelagem

Todo o pipeline de modelagem está em `notebooks/02-modelagem.ipynb`.

### Separação treino / teste

Divisão **80% treino / 20% teste** (`train_test_split`), com:
- `stratify=y` - preserva a proporção de classes (~63/37) em ambos os conjuntos, crítico em dados desbalanceados;
- `random_state=42` - garante reprodutibilidade.

Resultado: 455 amostras de treino e 114 de teste, com a proporção de classes mantida em ambos.

> **Observação sobre validação.** Para este escopo, adotamos um *split* treino/teste com modelos de configuração padrão, sem busca de hiperparâmetros. Como não houve ajuste iterativo baseado em um conjunto de validação, o conjunto de teste permanece um estimador honesto da generalização. Uma evolução natural (Seção 7) é introduzir validação cruzada para ajuste de hiperparâmetros.

### Modelos escolhidos e justificativa

Foram treinados **dois modelos** de naturezas diferentes, conforme exigido (duas ou mais técnicas):

| Modelo | Por que foi escolhido |
|---|---|
| **Regressão Logística** | Modelo linear, robusto e amplamente usado como referência em classificação binária médica. Principal vantagem: **alta interpretabilidade** - os coeficientes indicam a direção e a magnitude do efeito de cada feature, algo valioso num contexto clínico que exige justificativas. |
| **Árvore de Decisão** | Serve de **baseline comparativo** de natureza não linear. Captura interações entre features e é invariante à escala. Útil para verificar se um modelo mais flexível supera o linear - e, neste caso, para evidenciar o risco de *overfitting* de uma árvore sem poda. |

## 5. Avaliação e escolha da métrica

### Por que recall é a métrica prioritária

Num sistema de **triagem clínica**, nem todos os erros têm o mesmo custo:

- **Falso negativo** (prever benigno quando é maligno): um câncer deixa de ser sinalizado - o erro **mais grave**, pois pode atrasar o tratamento.
- **Falso positivo** (prever maligno quando é benigno): gera exames adicionais e ansiedade, mas é corrigido na investigação subsequente.

Como o falso negativo é muito mais custoso, a métrica prioritária é o **recall da classe maligna** (proporção de tumores malignos efetivamente detectados). Além disso, como as classes são desbalanceadas, a *accuracy* isolada é enganosa - por isso reportamos também **precision** e **F1-score**.

### Resultados

**Regressão Logística**

| Classe | Precision | Recall | F1-score |
|---|---|---|---|
| Benigno (0) | 0,96 | 0,99 | 0,97 |
| **Maligno (1)** | **0,97** | **0,93** | **0,95** |
| Accuracy | | | **0,96** |

Matriz de confusão (linha = real, coluna = previsto):

```
              prev. benigno   prev. maligno
real benigno       71              1          (1 falso positivo)
real maligno        3             39          (3 falsos negativos)
```

**Árvore de Decisão**

| Classe | Precision | Recall | F1-score |
|---|---|---|---|
| Benigno (0) | 0,94 | 0,94 | 0,94 |
| **Maligno (1)** | **0,90** | **0,90** | **0,90** |
| Accuracy | | | **0,93** |

Matriz de confusão:

```
              prev. benigno   prev. maligno
real benigno       68              4          (4 falsos positivos)
real maligno        4             38          (4 falsos negativos)
```

### Comparação

A **Regressão Logística supera a Árvore de Decisão em todas as métricas** e, crucialmente, comete **menos falsos negativos** (3 vs. 4) - exatamente o erro que mais queremos evitar. Some-se a isso a maior interpretabilidade, e a Regressão Logística é o **modelo escolhido**.

## 6. Interpretação dos resultados

A interpretabilidade foi abordada por dois métodos complementares.

### Coeficientes da Regressão Logística

Como as features foram padronizadas, os coeficientes são diretamente comparáveis. As features de maior peso (`texture_worst`, `radius_se`, `symmetry_worst`, `concave points_mean`, `concavity_worst`) apontam que **textura, concavidade e pontos côncavos** dos núcleos estão entre os sinais mais relevantes de malignidade. Vale lembrar que o coeficiente reflete a **relevância condicional** da feature no modelo multivariado, que pode diferir da correlação marginal isolada vista na EDA (efeito da multicolinearidade).

### SHAP (SHapley Additive exPlanations)

O SHAP confirma, por um método independente, a importância das features e - diferentemente dos coeficientes globais - explica a contribuição de cada feature para **previsões individuais**. Isso é especialmente valioso num contexto clínico: permite mostrar ao(à) médico(a) *por que* o modelo classificou um exame específico como suspeito, tornando a ferramenta auditável em vez de uma "caixa-preta".

## 7. Discussão crítica: o modelo pode ser usado na prática?

**Sim, mas exclusivamente como ferramenta de apoio à decisão - nunca como substituto do diagnóstico médico.**

**Como poderia ser usado.** O modelo se encaixa como uma camada de **triagem e priorização**: ao processar os exames, sinaliza os casos com alta probabilidade de malignidade para que sejam revisados primeiro, ajudando a otimizar o tempo da equipe clínica. O SHAP fornece, para cada caso, a justificativa da recomendação, mantendo o processo transparente.

**Limitações que impedem o uso autônomo.**

- **Falsos negativos ainda existem.** O modelo deixou passar 3 tumores malignos no conjunto de teste. Num uso autônomo isso seria inaceitável; como apoio, o(a) médico(a) revisa todos os casos e funciona como rede de segurança.
- **Dados de uma única fonte.** O WDBC vem de uma instituição e de um período específicos. O desempenho pode cair frente a equipamentos, populações e protocolos diferentes (*distribution shift*), exigindo validação externa antes de qualquer implantação.
- **Features dependem de medições prévias.** O modelo consome medidas já extraídas das imagens; ele não processa a imagem bruta. A qualidade dessa extração é um pré-requisito fora do escopo deste modelo.
- **Validação ainda preliminar.** Os resultados vêm de um único *split* treino/teste. Validação cruzada e ajuste de limiar (*threshold*) - possivelmente reduzindo o limiar para priorizar ainda mais o recall - são passos necessários antes do uso real.

**Conclusão.** O modelo é promissor e tem qualidade técnica para um piloto de **apoio à triagem**, sob supervisão médica e com monitoramento contínuo. **A palavra final no diagnóstico é, e deve permanecer, do(a) médico(a).**

---

*Os resultados, gráficos e o código completo que sustentam este relatório estão nos notebooks `notebooks/01-eda.ipynb` e `notebooks/02-modelagem.ipynb`.*
