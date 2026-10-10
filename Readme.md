# CKP02 — FinControl Assistant

**Prompt Engineering and Artificial Intelligence**

O **FinControl Assistant** é um assistente financeiro conversacional que evoluiu, no CKP02, para uma arquitetura baseada em **Retrieval-Augmented Generation (RAG)**. O sistema utiliza documentos reais do domínio financeiro como base de conhecimento, permitindo recuperar informações relevantes e gerar respostas fundamentadas nas fontes recuperadas.

---

## Integrantes

- **Caio Henique Ferraz da Silva** — RM 568992
- **Enzo Caruso Peter** — RM 570908
- **Leonardo Robert Maulicino** — RM 570329
- **Lucas Ramos de Sousa** — RM 573901
- **Leonardo Figueredo dos Santos** — RM 573653
- **Pablo Renato dos Santos** — RM 569894
- **Matheus Pimenta Martini** — RM 569400

---

## Modelos usados

- gemma4:cloud (chatbot);
- nomic-embed-text (modelo de embedding - roda no Ollama local).

---

## Domínio

O **FinControl Assistant** é um assistente e organizador financeiro desenvolvido para auxiliar o usuário no controle e na organização de suas informações financeiras.

O sistema trabalha com informações como:

- Receitas;
- Despesas;
- Categorias;
- Transações;
- Saldo;
- Histórico financeiro.

A proposta é oferecer uma interação conversacional por meio de um chatbot, permitindo que o usuário consulte e organize suas informações financeiras de maneira prática.

No CKP02, o projeto incorpora uma base de conhecimento documental e um pipeline RAG, ampliando a capacidade de recuperar informações relevantes de documentos e gerar respostas fundamentadas nas fontes disponíveis.

---

## Por que escolhemos esse domínio?

Escolhemos o domínio de **Assistente / Organizador Financeiro** porque a organização das finanças é uma atividade importante no cotidiano.

A proposta é utilizar Inteligência Artificial para facilitar o registro, a consulta e a organização de informações financeiras, permitindo que o usuário interaja com o sistema de forma conversacional.

Além disso, o domínio permite trabalhar com dados estruturados, histórico de transações, validação de informações com Pydantic e recuperação de informações a partir de documentos reais por meio de uma arquitetura RAG.

---

## Usuários-alvo

O FinControl Assistant é destinado a pessoas que desejam:

- Organizar suas receitas e despesas;
- Registrar e consultar transações financeiras;
- Acompanhar seu saldo;
- Consultar seu histórico financeiro;
- Utilizar um assistente conversacional para auxiliar na organização das finanças;
- Consultar informações presentes na base documental do sistema.

---

## Objetivo do projeto

### CKP01 — Chatbot financeiro

O objetivo do CKP01 foi desenvolver um chatbot utilizando técnicas e ferramentas de engenharia de prompts e integração com modelos de linguagem:

- Prompt Engineering;
- LangChain;
- LangChain Expression Language (LCEL);
- Memória gerenciada;
- Pydantic;
- Context Engineering;
- Integração com modelos de linguagem por meio do Ollama Cloud.

### CKP02 — Pipeline RAG

No CKP02, o projeto evolui com a implementação de uma arquitetura **Retrieval-Augmented Generation (RAG)**, composta por recuperação de informações em uma base documental e geração de respostas fundamentadas no contexto recuperado.

As principais tecnologias e componentes são:

- **LangChain e LCEL:** composição do fluxo de recuperação, construção do contexto, prompt e geração;
- **`nomic-embed-text` via Ollama:** geração de embeddings para consultas e documentos;
- **ChromaDB local:** armazenamento vetorial e recuperação de trechos relevantes;
- **`gemma4:cloud` via Ollama Cloud:** modelo utilizado na geração das respostas;
- **RAGAS:** avaliação da qualidade das respostas geradas pelo pipeline RAG.

O pipeline foi organizado de forma modular para permitir sua reutilização em etapas posteriores do projeto, incluindo a integração como ferramenta (`@tool`) no CKP03.

---

## Tabela de Requisitos (Rubrica FIAP)

A tabela abaixo relaciona os requisitos técnicos do CKP02 aos componentes conhecidos do projeto.

| Requisito | Arquivo ou componente | Descrição |
| :--- | :--- | :--- |
| **Base de conhecimento real** | Base documental (PDFs) | Utilização de pelo menos cinco documentos reais do domínio, em PDF, TXT ou Markdown, com fontes citadas. |
| **Chunking** | Pipeline de ingestão | Utilização de `RecursiveCharacterTextSplitter` com `separators=["\n\n", "\n", ". ", " ", ""]`, `chunk_overlap` entre 10% e 15% do `chunk_size` e comparação de pelo menos duas configurações. |
| **Embeddings** | `app/vector_store.py` | Utilização de `nomic-embed-text` via Ollama para gerar embeddings de documentos e consultas. |
| **Modelo de geração** | `app/rag.py` | Utilização de `gemma4:cloud` via Ollama Cloud, com `temperature=0`, para gerar respostas fundamentadas no contexto recuperado. |
| **Banco vetorial** | `app/vector_store.py` | Armazenamento dos embeddings no ChromaDB local. |
| **Pipeline end-to-end** | `app/rag.py` | Fluxo de recuperação, construção de contexto, geração de resposta e apresentação das fontes utilizadas. |
| **Avaliação com RAGAS** | Integração de avaliação e notebook | Comparação de pelo menos duas configurações de chunking utilizando `faithfulness` e `answer_relevancy`, com pelo menos cinco perguntas de teste. |

---

## Justificativa da Memória e Experimento de Context Rot (CKP01)

Esta seção documenta o experimento de memória desenvolvido no CKP01, preservado como parte do histórico técnico do projeto.

Para investigar o equilíbrio entre retenção de contexto e gerenciamento do histórico conversacional, foi adotado o `ConversationTokenBufferMemory`, com avaliação de diferentes limites de tokens.

O experimento de degradação de contexto (*Context Rot*) consistiu em inserir uma informação relevante no primeiro turno — meta de economia de R$ 850,00 — e acrescentar turnos de preenchimento à conversa. Foram consideradas janelas de 800, 1000, 1200 e 1500 tokens, com validações a cada cinco turnos.

### Resultados observados (`resultados_context_rot.csv`)

| Janela de tokens | Turno 5 | Turno 10 | Turnos 15–35 | Comportamento observado |
| :---: | :---: | :---: | :---: | :--- |
| **800** | Lembra | Esquece | Esquece | O limite curto causou expurgo prematuro do histórico. A meta do turno 1 foi eliminada rapidamente para alocar as novas perguntas. |
| **1000** | Lembra | Esquece | Esquece | **(NOSSA ESCOLHA)**: Oferece contexto suficiente para manter o fluxo natural do bate-papo sem acumular lixo informacional antigo. A informação inicial degrada de forma controlada após 10 turnos |
| **1500** | Lembra | Esquece | Esquece | Foi observado estouro do limite no turno 10, atribuído ao tamanho do prompt de sistema e ao histórico de transações. |

**Conclusão do experimento:** no contexto testado, a configuração de 1000 tokens foi escolhida para gerenciar o histórico conversacional. Os resultados indicaram, entretanto, que o aumento da janela não garantiu a retenção da informação inicial.

Esse experimento pertence à etapa de memória conversacional do CKP01. Ele não deve ser confundido com a avaliação de recuperação e geração do pipeline RAG implementado no CKP02.

---

## Como executar o projeto

### 1. Preparar o ambiente

Certifique-se de ter o Python 3.10 ou superior instalado.

Renomeie o arquivo `.env.example` para `.env` e configure a variável de ambiente `OLLAMA_API_KEY` com sua chave de acesso ao Ollama Cloud.

Não compartilhe o arquivo `.env` nem exponha a chave em commits ou no README.

Crie e ative um ambiente virtual, se desejar:

```bash
python -m venv venv
```

Linux/macOS:

```bash
source venv/bin/activate
```

Windows:

```bash
venv\Scripts\activate
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

As dependências presentes no requirements.txt são: 

- langchain;
- langchain-ollama;
- langchain-classic;
- tiktoken;
- pydantic;
- python-dotenv;
- gradio;
- langchain-community;
- langchain-text-splitters;
- chromadb;
- langchain-chroma;
- pypdf;
- ollama;
- datasets;
- ragas;

### 2. Executar a interface do chatbot

Para iniciar a interface Gradio do assistente financeiro:

```bash
python -m app.main
```

### 3. Executar o experimento de Context Rot

Para reproduzir o experimento de gerenciamento de memória do CKP01:

```bash
python -m app.context_rot
```

### 4. Preparar a base vetorial

Para gerar o banco vetorial utilizado pelo pipeline RAG:

```bash
python -m app.vector_store
```

Execute essa etapa após configurar a base documental e o ambiente necessário para a geração de embeddings.

### 5. Testar o pipeline RAG

Para executar os testes de recuperação e geração:

```bash
python -m app.teste_rag
```

Para executar os testes com a configuração de chunk de 1024:

```bash
python -m app.teste_rag --chunk 1024
```

Também é possível realizar uma consulta diretamente pelo módulo RAG:

```bash
python -m app.rag "Quanto devo ter de reserva de emergência?"
```

Os comandos específicos do pipeline RAG pressupõem que a base vetorial e as variáveis de ambiente necessárias estejam configuradas.

---

## Tecnologias Utilizadas

- **Python:** linguagem de implementação;
- **LangChain:** composição dos componentes de processamento de linguagem natural;
- **LCEL:** composição das etapas de prompt, modelo e parser;
- **Pydantic v2:** validação e estruturação de dados;
- **Gradio:** interface gráfica do chatbot;
- **Ollama Cloud:** acesso ao modelo `gemma4:cloud`;
- **`nomic-embed-text`:** modelo de embeddings;
- **ChromaDB:** armazenamento e recuperação vetorial;
- **RAGAS:** avaliação do pipeline RAG;
- **Tiktoken:** contagem de tokens utilizada no experimento de Context Rot;
- **Python-dotenv:** carregamento de variáveis de ambiente a partir de arquivo `.env`.

---

## Estrutura do Projeto

```text
CKP02_FinControlAssistant_equipe01/
│
├── app/
│   ├── __init__.py
│   ├── chain.py                # Chains de conversação e análise Pydantic
│   ├── prompt.py               # Template de prompt do chatbot
│   ├── memory_manager.py       # Configuração da memória de tokens
│   ├── context_rot.py          # Experimento de degradação de memória
│   ├── main.py                 # Interface Gradio
│   ├── vector_store.py         # Armazenamento e recuperação vetorial
│   ├── rag.py                  # Pipeline de recuperação e geração
│   ├── teste_rag.py            # Testes do pipeline RAG
│   ├── teste_busca.py          # Testes da recuperação semântica de trechos no ChromaDB
│   ├── avaliacao_ragas.py       # Avaliação das respostas do RAG com métricas RAGAS e comparação de chunking
│   └── schemas/
│       ├── __init__.py
│       ├── Historico.py        # Schema de validação do histórico
│       └── Transacao.py        # Schema de validação de transações
│
├── data/                       # Base documental do projeto
├── resultados_ragas/           # Resultados do avaliacao_ragas.py
├── app/
│   └── CPK02_DocMind_RAG.ipynb
├── .env.example
├── requirements.txt
└── README.md
```

---

## Pipeline RAG (Fase 3 — Retriever, prompt e geração)

**Implementação:** `app/rag.py`  
**Testes:** `app/teste_rag.py`

O pipeline RAG recebe uma pergunta, recupera trechos relevantes da base vetorial, monta o contexto utilizado pelo modelo de linguagem e retorna uma resposta acompanhada das fontes.

### Fluxo do pipeline

```text
Pergunta do usuário
        |
        v
recuperar()
        |
        v
montar_contexto()
        |
        v
Prompt RAG
        |
        v
gemma4:cloud
        |
        v
StrOutputParser
        |
        v
Resposta + fontes
```

A recuperação utiliza a função `buscar()` do módulo `app/vector_store.py`, que consulta o ChromaDB com embeddings produzidos por `nomic-embed-text`.

### Componentes do pipeline

| Etapa | Função | Responsabilidade |
| :--- | :--- | :--- |
| **Retriever** | `recuperar(pergunta, tamanho_chunk, k)` | Utiliza `buscar()` para recuperar os `k` trechos mais próximos, incluindo informações de arquivo, página e similaridade. |
| **Construção do contexto** | `montar_contexto(trechos)` | Numera os trechos (`[1]`, `[2]` etc.) e inclui informações de arquivo e página para permitir a identificação das fontes. |
| **Prompt** | `prompt_rag` | Orienta o modelo a responder somente com base nos trechos fornecidos, citar os trechos utilizados e retornar uma frase predefinida quando a base não contiver a resposta. |
| **Geração** | `gerar_resposta()` | Utiliza LCEL para compor `prompt \| ChatOllama \| StrOutputParser`, com `gemma4:cloud` e `temperature=0`. Inclui tentativas de repetição para erros 503 e 429. |
| **Formatação das fontes** | `montar_fontes()` e `formatar_resposta_com_fontes()` | Organiza as fontes utilizadas, incluindo arquivo, página, similaridade e trecho recuperado. |
| **Pipeline completo** | `responder(pergunta, tamanho_chunk=512, k=3)` | Integra recuperação, construção de contexto, geração e formatação da resposta. Aceita os tamanhos de chunk 512 e 1024. |


### Como adicionar novos documentos à base de conhecimento

1. Coloque o novo documento PDF no diretório `data/`, localizado na raiz do projeto.
2. Abra `app/vector_store.py` e adicione o caminho do arquivo à lista `arquivos_pdf`, dentro da função `processar_e_armazenar_documentos()`. 
3. Verifique se o Ollama está disponível e se o modelo de embeddings está instalado:

   ```bash
   ollama pull nomic-embed-text
   ```

4. Na raiz do projeto, execute:

   ```bash
   python -m app.vector_store
   ```
   Esse comando carrega todos os PDFs listados em arquivos.pdf, divide os chunks, gera os embeddings e **RECRIA** as coleções com todos os            documentos listados.  

5. Aguarde a conclusão do carregamento, da divisão em chunks, da geração de embeddings com `nomic-embed-text` e da persistência no ChromaDB.
6. Teste a recuperação do conteúdo com:

   ```bash
   python -m app.rag "Sua pergunta sobre o novo documento"
   ```

**Observações:**
- O processo atual aceita arquivos PDF por meio do `PyPDFLoader`.
- Todos os documentos que devem permanecer na base precisam constar na lista `arquivos_pdf`.
- A execução recria as coleções `fin_control_512` e `fin_control_1024`, apagando as versões anteriores antes de armazenar os novos chunks.
- Para utilizar outro formato de arquivo, é necessário adaptar o código de carregamento.

### Integração com RAGAS

A função `gerar_amostra_ragas()` permite gerar amostras para a avaliação do pipeline.

Exemplo de utilização:

```python
from app.rag import gerar_amostra_ragas

PERGUNTAS = [
    # Inserir pelo menos cinco perguntas de teste.
]

amostras_512 = [
    gerar_amostra_ragas(pergunta, tamanho_chunk=512)
    for pergunta in PERGUNTAS
]

amostras_1024 = [
    gerar_amostra_ragas(pergunta, tamanho_chunk=1024)
    for pergunta in PERGUNTAS
]
```

Cada amostra contém os campos `user_input`, `response` e `retrieved_contexts`, podendo incluir `reference` quando uma resposta esperada é fornecida.

Para a avaliação de *faithfulness*, o campo `response` contém somente o texto gerado pelo modelo, sem a lista formatada de fontes acrescentada à resposta final. Isso permite avaliar a fundamentação textual sem incluir a apresentação das fontes como parte da resposta gerada.

## Avaliação do RAG e comparação de chunking

### Objetivo

Avaliar o impacto de diferentes configurações de chunking na recuperação de contexto e na geração de respostas do FinControl Assistant.

Foram comparadas duas configurações de `chunk_size`:

- **Configuração A:** `chunk_size=512`;
- **Configuração B:** `chunk_size=1024`.

As mesmas seis perguntas foram submetidas à aplicação em ambas as configurações, permitindo comparar as métricas obtidas sob um conjunto comum de perguntas.

### Metodologia

Para cada pergunta e configuração de chunking, foram registrados:

- Identificador e categoria da pergunta;
- Pergunta enviada ao modelo (`user_input`);
- Percentual de fontes esperadas recuperadas;
- Média de caracteres no contexto;
- Métricas RAGAS: `faithfulness` e `answer_relevancy`.

Os resultados foram organizados em duas planilhas: uma com os registros individuais de cada pergunta e outra com as médias das métricas para cada configuração.

### Resultados (por pergunta)

| ID | Categoria | Pergunta | Faith. 512 | Faith. 1024 | Relev. 512 | Relev. 1024 |
| --- | --- | --- | --- | --- | --- | --- |
| P1 | Orçamento | O que é um orçamento pessoal e para que ele serve? | 0.83 | 1.00 | 0.74 | 0.87 |
| P2 | Receitas e despesas | Qual a diferença entre despesas fixas e variáveis? | 1.00 | 1.00 | 0.95 | 0.89 |
| P3 | Educação financeira | Por que é importante ter educação financeira? | 1.00 | 1.00 | 0.99 | 0.93 |
| P4 | Planejamento financeiro | Quais são as etapas para fazer um planejamento financeiro pessoal? | 1.00 | 1.00 | 0.88 | 0.80 |
| P5 | Reserva financeira | Quanto dinheiro devo ter separado para imprevistos? | 1.00 | 1.00 | 0.78 | 0.58 |
| P6 | Receitas e despesas | O que fazer quando as despesas são maiores que a receita? | 1.00 | 1.00 | 0.65 | 0.00 |

### Médias por configuração

| Configuração | Faithfulness (média) | Answer relevancy (média) | Fonte esperada recuperada | Tamanho médio do contexto (chars) | Faithfulness ≥ 0,7? |
| --- | --- | --- | --- | --- | --- |
| chunk 512 | 0.97 | 0.83 | 83% | 399 | SIM |
| chunk 1024 | 1.00 | 0.68 | 100% | 900 | SIM |

### Configurações e contexto recuperado

| Configuração   | Tamanho do chunk |        Overlap | Chunks recuperados (`k`) |
| -------------- | ---------------: | -------------: | -----------------------: | 
| Configuração A |   512 caracteres |  51 caracteres |                        3 | 
| Configuração B |  1024 caracteres | 102 caracteres |                        3 | 

**Observação:** o overlap foi configurado em aproximadamente 10% do tamanho de cada chunk. A recuperação da fonte esperada indica se ao menos um dos documentos esperados para aquela pergunta apareceu entre os documentos recuperados; isso não substitui a avaliação de relevância semântica realizada pelo RAGAS.


## Análise crítica dos resultados e escolha da configuração

### Comparação das métricas

A avaliação foi realizada com seis perguntas, utilizando três chunks recuperados por consulta (`k=3`) e comparando duas configurações de chunking: 512 e 1024 caracteres.

Ambas as configurações superaram o requisito mínimo de faithfulness de 0,70 e atingiram a referência de 0,90. Entretanto, os resultados mostram diferenças entre as configurações: o chunk 1024 apresentou maior faithfulness média, enquanto o chunk 512 obteve maior answer relevancy média.

### Análise crítica

A configuração com chunks de 1024 apresentou faithfulness médio de 1,00 e recuperou a fonte esperada nas seis perguntas avaliadas. Apesar disso, sua answer relevancy média foi inferior à da configuração de 512 caracteres.

Um caso relevante ocorreu na pergunta P6: “O que fazer quando as despesas são maiores que a receita?”. Com chunk 1024, a resposta explicou o significado de um resultado deficitário, mas não apresentou ações específicas para reverter a situação. Os contextos recuperados nessa configuração também não continham as orientações específicas que estavam disponíveis na recuperação com chunk 512. A answer relevancy dessa resposta foi avaliada em 0,00.

Esse caso demonstra que uma pontuação elevada de faithfulness, isoladamente, não garante que a resposta atenda integralmente à pergunta. A fidelidade ao contexto e a relevância da resposta devem ser analisadas em conjunto.

A recuperação da fonte esperada também deve ser interpretada como um indicador auxiliar: não encontrar o documento previamente esperado não significa necessariamente que os trechos recuperados sejam irrelevantes ou incorretos.

### Escolha da configuração

Com base nos resultados, adotamos provisoriamente chunks de 512 caracteres como configuração padrão do projeto. Essa escolha considera o equilíbrio entre faithfulness elevada (0,9722) e maior answer relevancy média (0,8315).

A configuração de 1024 caracteres continua sendo uma alternativa válida, especialmente porque apresentou faithfulness de 1,00 e recuperou a fonte esperada em todas as perguntas avaliadas. No entanto, sua menor relevância média indica que fornecer mais contexto não garante, por si só, respostas mais adequadas à pergunta.

A escolha de 512 caracteres não representa uma superioridade em todas as métricas, mas uma decisão baseada no conjunto de resultados observados. Essa conclusão é provisória, pois a avaliação utilizou seis perguntas e o mesmo modelo, `gemma4:cloud`, tanto para gerar as respostas quanto para avaliar as métricas RAGAS.

### Limitações da avaliação

* Foram avaliadas seis perguntas, o que limita a generalização dos resultados para outras consultas.
* A recuperação utilizou `k=3`, isto é, até três trechos por pergunta.
* O mesmo modelo foi utilizado na geração das respostas e no julgamento RAGAS, o que pode influenciar as pontuações.
* A resposta P6 com chunk 1024 merece investigação adicional devido à baixa answer relevancy.

Portanto, os resultados fornecem evidências iniciais para a escolha da configuração, mas avaliações adicionais serão necessárias para confirmar seu desempenho em um conjunto maior e mais diversificado de perguntas.

### Testes do pipeline

Para executar os testes documentados:

```bash
# Gera o banco vetorial
python -m app.vector_store

# Executa cinco perguntas e uma pergunta fora do escopo,
# nas configurações de chunking de 512 e 1024
python -m app.teste_rag

# Executa os testes com chunk de 1024
python -m app.teste_rag --chunk 1024

# Executa uma consulta direta ao pipeline RAG
python -m app.rag "Quanto devo ter de reserva de emergência?"
```

A variável `OLLAMA_API_KEY` deve estar configurada no arquivo `.env` para execução local ou nos Secrets do Google Colab, quando executado nesse ambiente. NUNCA revele as variáveis do .env.

---