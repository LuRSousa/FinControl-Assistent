"""
M5 — Fase 3: RAG (retriever + prompt + gemma4:cloud + resposta fundamentada + fontes).

Este módulo NÃO recria embeddings nem ChromaDB: usa as funções do M4
(`buscar` e `nome_do_arquivo` de app/vector_store.py), que abrem as coleções
já salvas em ./chroma_db.

Fluxo:  pergunta -> recuperar() -> montar_contexto() -> prompt | llm | parser -> resposta + fontes

Funções principais:
    responder(pergunta, tamanho_chunk=512, k=3)  -> dict completo (usado pelo M6/RAGAS e pelo Gradio)
    gerar_amostra_ragas(pergunta, ...)           -> dict no formato esperado pelo RAGAS
    perguntar_a_base(consulta)                   -> str pronta para virar @tool no CKP03

Uso rápido (da raiz do projeto):
    from app.rag import responder
    r = responder("Quanto devo guardar por mês?", tamanho_chunk=1024)
    print(r["resposta"]); print(r["fontes"])
"""

import os
import time
from functools import lru_cache

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from app.vector_store import COLECOES, buscar, nome_do_arquivo

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

# Exigência do CKP02: exclusivamente gemma4:cloud via Ollama Cloud, temperature=0
MODELO_CHAT = "gemma4:cloud"
URL_OLLAMA_CLOUD = "https://ollama.com"
TEMPERATURA = 0

# Quantos chunks vão para o LLM e quantos caracteres do trecho aparecem na fonte
K_PADRAO = 3
TAMANHO_TRECHO_FONTE = 300

# Frase fixa quando a base não tem a resposta (evita alucinação e facilita testar)
RESPOSTA_SEM_CONTEXTO = "Não encontrei essa informação nos documentos da base."


# ---------------------------------------------------------------------------
# LLM (gemma4:cloud)
# ---------------------------------------------------------------------------

def obter_api_key():
    # No Colab a chave fica em Secrets; localmente vem do .env / variável de ambiente.
    chave = os.getenv("OLLAMA_API_KEY")
    if chave:
        return chave

    try:
        from dotenv import load_dotenv
        load_dotenv()
        chave = os.getenv("OLLAMA_API_KEY")
        if chave:
            return chave
    except ImportError:
        pass

    try:
        from google.colab import userdata  # só existe dentro do Colab
        return userdata.get("OLLAMA_API_KEY")
    except Exception:
        pass

    raise RuntimeError(
        "OLLAMA_API_KEY não encontrada. Defina no arquivo .env (local) "
        "ou em Secrets do Colab antes de gerar respostas."
    )


@lru_cache(maxsize=1)
def criar_llm():
    # Criado só na primeira chamada (e reaproveitado), para o import não falhar sem chave.
    return ChatOllama(
        model=MODELO_CHAT,
        base_url=URL_OLLAMA_CLOUD,
        client_kwargs={"headers": {"Authorization": f"Bearer {obter_api_key()}"}},
        temperature=TEMPERATURA,  # 0 = respostas fundamentadas e reproduzíveis
    )


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

PROMPT_SISTEMA = """Você é o FinControl Assistant, especialista em educação e planejamento financeiro pessoal.
Você responde perguntas usando EXCLUSIVAMENTE os trechos de documentos oficiais (BCB, CVM, CAIXA, SUSEP) fornecidos abaixo.

<regras>
- Use somente informações que estejam nos trechos. Não use conhecimento próprio nem invente números, percentuais ou prazos.
- Se os trechos não tiverem a resposta, responda exatamente: "{resposta_sem_contexto}"
- Se os trechos responderem só em parte, responda apenas a parte coberta e diga o que não foi encontrado.
- Cite a origem no próprio texto usando o número do trecho entre colchetes, por exemplo [1] ou [2][3], logo após cada afirmação.
- Não escreva uma lista de fontes no final; ela é adicionada automaticamente pelo sistema.
- Responda em português do Brasil, em linguagem simples, de forma direta (até 5 frases).
- Não recomende produtos financeiros específicos (ações, fundos, criptomoedas, bancos).
</regras>

<trechos>
{contexto}
</trechos>"""

prompt_rag = ChatPromptTemplate.from_messages([
    ("system", PROMPT_SISTEMA),
    ("human", "{pergunta}"),
])


# ---------------------------------------------------------------------------
# Retriever (usa buscar() do M4)
# ---------------------------------------------------------------------------

def validar_tamanho_chunk(tamanho_chunk):
    if tamanho_chunk not in COLECOES:
        raise ValueError(
            f"Tamanho de chunk inválido: {tamanho_chunk}. Use um destes: {sorted(COLECOES)}."
        )


def recuperar(pergunta, tamanho_chunk=512, k=K_PADRAO):
    """Busca os k chunks mais parecidos com a pergunta na coleção do tamanho escolhido.

    Devolve uma lista de dicionários (já na ordem de relevância), cada um com:
    indice, texto, arquivo, pagina, similaridade.
    """
    validar_tamanho_chunk(tamanho_chunk)

    trechos = []
    for posicao, (documento, similaridade) in enumerate(
        buscar(pergunta, tamanho_chunk=tamanho_chunk, k=k), start=1
    ):
        trechos.append({
            "indice": posicao,
            "texto": documento.page_content,
            "arquivo": nome_do_arquivo(documento),
            "pagina": documento.metadata.get("page", 0) + 1,  # PyPDFLoader conta a partir de 0
            "similaridade": round(float(similaridade), 4),
        })
    return trechos


def montar_contexto(trechos):
    # Cada trecho entra numerado, com arquivo e página, para o LLM poder citar [n].
    blocos = []
    for t in trechos:
        texto = " ".join(t["texto"].split())  # tira quebras de linha do PDF
        blocos.append(f"[{t['indice']}] (arquivo: {t['arquivo']}, página {t['pagina']})\n{texto}")
    return "\n\n".join(blocos)


# ---------------------------------------------------------------------------
# Geração
# ---------------------------------------------------------------------------

def gerar_resposta(pergunta, contexto, tentativas=3, espera=5):
    """prompt | llm | parser (LCEL). Tenta de novo em caso de 503 da Ollama Cloud."""
    cadeia = prompt_rag | criar_llm() | StrOutputParser()
    entrada = {
        "pergunta": pergunta,
        "contexto": contexto,
        "resposta_sem_contexto": RESPOSTA_SEM_CONTEXTO,
    }

    for tentativa in range(1, tentativas + 1):
        try:
            return cadeia.invoke(entrada).strip()
        except Exception as erro:
            indisponivel = getattr(erro, "status_code", None) in (429, 503)
            if indisponivel and tentativa < tentativas:
                print(f"Ollama Cloud indisponível ({erro.status_code}). Tentativa {tentativa}/{tentativas}, aguardando {espera}s...")
                time.sleep(espera)
            else:
                raise


def montar_fontes(trechos):
    # Fontes para exibir ao usuário: uma por chunk recuperado, na mesma numeração [n] da resposta.
    fontes = []
    for t in trechos:
        trecho = " ".join(t["texto"].split())
        if len(trecho) > TAMANHO_TRECHO_FONTE:
            trecho = trecho[:TAMANHO_TRECHO_FONTE] + "..."
        fontes.append({
            "indice": t["indice"],
            "arquivo": t["arquivo"],
            "pagina": t["pagina"],
            "similaridade": t["similaridade"],
            "trecho": trecho,
        })
    return fontes


def formatar_resposta_com_fontes(resultado):
    """Texto final para o usuário (terminal/Gradio): resposta + lista de fontes."""
    linhas = [resultado["resposta"], "", "Fontes:"]
    for f in resultado["fontes"]:
        linhas.append(
            f"[{f['indice']}] {f['arquivo']} — página {f['pagina']} (similaridade {f['similaridade']:.2f})"
        )
    return "\n".join(linhas)


def responder(pergunta, tamanho_chunk=512, k=K_PADRAO):
    """Pipeline completo: retrieve -> generate.

    tamanho_chunk: 512 ou 1024 (coleção do ChromaDB usada na busca).
    Retorna um dict com:
        pergunta, tamanho_chunk, k,
        resposta   -> texto gerado pelo LLM (com citações [n]); é o que o RAGAS avalia
        contextos  -> lista com o texto de cada chunk recuperado (retrieved_contexts do RAGAS)
        fontes     -> lista de {indice, arquivo, pagina, similaridade, trecho}
    """
    if not pergunta or not pergunta.strip():
        raise ValueError("A pergunta não pode estar vazia.")

    trechos = recuperar(pergunta, tamanho_chunk=tamanho_chunk, k=k)

    if trechos:
        resposta = gerar_resposta(pergunta, montar_contexto(trechos))
    else:
        resposta = RESPOSTA_SEM_CONTEXTO

    return {
        "pergunta": pergunta,
        "tamanho_chunk": tamanho_chunk,
        "k": k,
        "resposta": resposta,
        "contextos": [t["texto"] for t in trechos],
        "fontes": montar_fontes(trechos),
    }


# ---------------------------------------------------------------------------
# Integração com o M6 (RAGAS) e com o CKP03 (agente)
# ---------------------------------------------------------------------------

def gerar_amostra_ragas(pergunta, tamanho_chunk=512, k=K_PADRAO, resposta_esperada=None):
    """Mesmo pipeline, mas no formato de linha do RAGAS (user_input / response / retrieved_contexts).

    Exemplo para o M6:
        amostras = [gerar_amostra_ragas(p, tamanho_chunk=512) for p in PERGUNTAS]
        dataset = EvaluationDataset.from_list(amostras)
    """
    r = responder(pergunta, tamanho_chunk=tamanho_chunk, k=k)
    amostra = {
        "user_input": r["pergunta"],
        "response": r["resposta"],
        "retrieved_contexts": r["contextos"],
    }
    if resposta_esperada:
        amostra["reference"] = resposta_esperada
    return amostra


def perguntar_a_base(consulta, tamanho_chunk=512):
    """Resposta + fontes em texto único. No CKP03 basta decorar com @tool
    (a docstring abaixo vira a descrição da tool para o agente)."""
    resultado = responder(consulta, tamanho_chunk=tamanho_chunk)
    return formatar_resposta_com_fontes(resultado)


if __name__ == "__main__":
    import sys

    pergunta_cli = " ".join(sys.argv[1:]) or "Quanto do meu salário eu deveria guardar todo mês?"
    for tamanho in COLECOES:
        print(f"\n=== Chunks de {tamanho} ===")
        print(formatar_resposta_com_fontes(responder(pergunta_cli, tamanho_chunk=tamanho)))
