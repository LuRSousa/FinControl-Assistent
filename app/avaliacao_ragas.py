# M6 — Avaliação RAGAS: compara chunks de 512 x 1024 com faithfulness e answer_relevancy.
#
# Coloque este arquivo em app/ e rode da raiz do projeto:
#     pip install ragas pandas
#     python -m app.avaliacao_ragas --diagnostico   # descobre ONDE está o erro (rode este primeiro)
#     python -m app.avaliacao_ragas                 # gera respostas + avalia (512 e 1024)
#     python -m app.avaliacao_ragas --so-gerar      # só gera as respostas (não usa o ragas)
#     python -m app.avaliacao_ragas --reaproveitar  # pula a geração e reaproveita respostas_rag.json
#     python -m app.avaliacao_ragas --k 3           # k de chunks recuperados (padrão 3)
#
# Arquivos gerados (raiz do projeto). Os CSVs usam ";" e vírgula decimal, para abrir direto no Excel BR:
#     respostas_rag.json       -> respostas + contextos (cache; evita gerar tudo de novo se o RAGAS falhar)
#     resultados_ragas.csv     -> uma linha por (chunk, pergunta), com métricas, fontes e resposta
#     comparacao_ragas.csv     -> UMA linha por pergunta, 512 x 1024 lado a lado + linha MEDIA
#     resumo_ragas.csv         -> médias por configuração, alvo ≥ 0,7 e melhor configuração
#     resultados_ragas.md      -> tabelas + interpretação, pronto para colar no README/notebook

import argparse
import json
import math
import os
import traceback

import pandas as pd

from app.rag import MODELO_CHAT, URL_OLLAMA_CLOUD, obter_api_key, responder
from app.vector_store import COLECOES, MODELO_EMBEDDING

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

ALVO_FAITHFULNESS = 0.7        # aprovação (CKP02)
ZONA_IDEAL_FAITHFULNESS = 0.9  # zona ideal (escala da Aula 07)
LIMITE_ALUCINACAO = 0.5        # abaixo disso = alucinação significativa
MARGEM_EMPATE = 0.02           # diferença menor que isso é considerada empate

ARQUIVO_RESPOSTAS = "respostas_rag.json"
CSV_KW = {"sep": ";", "decimal": ",", "encoding": "utf-8-sig", "index": False}  # padrão Excel BR

METRICAS = ["faithfulness", "answer_relevancy"]

# Perguntas cobrindo partes diferentes da base (uma por categoria + 1 extra).
# "fonte_esperada": trechos do nome do PDF onde a resposta deveria estar.
PERGUNTAS = [
    {"id": "P1", "categoria": "Orçamento", "fonte_esperada": ["SUSEP", "CVM"],
     "pergunta": "O que é um orçamento pessoal e para que ele serve?"},
    {"id": "P2", "categoria": "Receitas e despesas", "fonte_esperada": ["CAIXA"],
     "pergunta": "Qual a diferença entre despesas fixas e variáveis?"},
    {"id": "P3", "categoria": "Educação financeira", "fonte_esperada": ["BCB"],
     "pergunta": "Por que é importante ter educação financeira?"},
    {"id": "P4", "categoria": "Planejamento financeiro", "fonte_esperada": ["CVM"],
     "pergunta": "Quais são as etapas para fazer um planejamento financeiro pessoal?"},
    {"id": "P5", "categoria": "Reserva financeira", "fonte_esperada": ["CVM", "BCB"],
     "pergunta": "Quanto dinheiro devo ter separado para imprevistos?"},
    {"id": "P6", "categoria": "Receitas e despesas", "fonte_esperada": ["CAIXA", "SUSEP", "CVM"],
     "pergunta": "O que fazer quando as despesas são maiores que a receita?"},
]


def eh_nan(valor):
    return valor is None or (isinstance(valor, float) and math.isnan(valor))


def fmt(valor):
    return "n/d" if eh_nan(valor) else f"{valor:.2f}"


# ---------------------------------------------------------------------------
# Fase 1 — geração das respostas do RAG (não depende do ragas)
# ---------------------------------------------------------------------------

def gerar_linhas(tamanho_chunk, k):
    """Roda o RAG (M5) para todas as perguntas com um tamanho de chunk."""
    linhas = []
    for q in PERGUNTAS:
        print(f"[chunk {tamanho_chunk}] {q['id']}: {q['pergunta']}")
        r = responder(q["pergunta"], tamanho_chunk=tamanho_chunk, k=k)

        arquivos = [f["arquivo"] for f in r["fontes"]]
        acerto_fonte = any(tag in arq for arq in arquivos for tag in q["fonte_esperada"])
        tamanhos = [len(c) for c in r["contextos"]]

        linhas.append({
            "chunk": tamanho_chunk,
            "id": q["id"],
            "categoria": q["categoria"],
            "user_input": r["pergunta"],
            "response": r["resposta"],
            "retrieved_contexts": r["contextos"],
            "fontes": "; ".join(f"{f['arquivo']} p.{f['pagina']}" for f in r["fontes"]),
            "acerto_fonte": acerto_fonte,
            "contexto_medio_chars": sum(tamanhos) / len(tamanhos) if tamanhos else 0,
        })
    return linhas


def gerar_todas(k):
    todas = []
    for tamanho in COLECOES:
        todas.extend(gerar_linhas(tamanho, k))
        # salva a cada configuração: se algo quebrar depois, a geração não se perde
        with open(ARQUIVO_RESPOSTAS, "w", encoding="utf-8") as arquivo:
            json.dump(todas, arquivo, ensure_ascii=False, indent=2)
    print(f"Respostas salvas em {ARQUIVO_RESPOSTAS}")
    return todas


# ---------------------------------------------------------------------------
# Fase 2 — RAGAS (cada métrica roda isolada: se uma falhar, a outra continua)
# ---------------------------------------------------------------------------

def preparar_asyncio():
    # Em notebook/Colab o loop já está rodando; o ragas precisa do nest_asyncio.
    try:
        import asyncio
        asyncio.get_running_loop()
    except RuntimeError:
        return  # script normal: nada a fazer
    try:
        import nest_asyncio
        nest_asyncio.apply()
    except ImportError:
        print("AVISO: rodando dentro de notebook; instale com: pip install nest_asyncio")


def criar_juiz():
    """LLM juiz (gemma4:cloud, temperature=0) e embeddings (nomic-embed-text) para o RAGAS."""
    from langchain_ollama import ChatOllama, OllamaEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    llm = ChatOllama(
        model=MODELO_CHAT,
        base_url=URL_OLLAMA_CLOUD,
        client_kwargs={"headers": {"Authorization": f"Bearer {obter_api_key()}"}},
        temperature=0,
    )
    embeddings = OllamaEmbeddings(model=MODELO_EMBEDDING)  # answer_relevancy precisa de embeddings
    return LangchainLLMWrapper(llm), LangchainEmbeddingsWrapper(embeddings)


def criar_metricas():
    # Compatível com versões diferentes do ragas (classes x instâncias prontas)
    try:
        from ragas.metrics import AnswerRelevancy, Faithfulness
        return {"faithfulness": Faithfulness(), "answer_relevancy": AnswerRelevancy()}
    except ImportError:
        from ragas.metrics import answer_relevancy, faithfulness
        return {"faithfulness": faithfulness, "answer_relevancy": answer_relevancy}


def avaliar_linhas(linhas, llm_juiz, embeddings_juiz, metricas):
    """Calcula as métricas do RAGAS para as linhas de UM tamanho de chunk."""
    from ragas import EvaluationDataset, evaluate
    from ragas.run_config import RunConfig

    dataset = EvaluationDataset.from_list([
        {"user_input": l["user_input"], "response": l["response"],
         "retrieved_contexts": l["retrieved_contexts"]}
        for l in linhas
    ])
    # poucos workers + retries: a Ollama Cloud responde 429/503 sob carga
    config = RunConfig(timeout=240, max_retries=6, max_wait=30, max_workers=2)

    for nome, metrica in metricas.items():
        print(f"  RAGAS: {nome} (chunk {linhas[0]['chunk']})...")
        try:
            resultado = evaluate(
                dataset=dataset, metrics=[metrica], llm=llm_juiz, embeddings=embeddings_juiz,
                run_config=config, raise_exceptions=False,
            )
            df = resultado.to_pandas()
            coluna = nome if nome in df.columns else next(
                c for c in df.columns if c not in ("user_input", "response", "retrieved_contexts", "reference"))
            notas = list(df[coluna])
        except Exception as erro:
            print(f"  !! {nome} falhou: {type(erro).__name__}: {erro}")
            print("     (detalhes abaixo; mande esse trecho para ajustarmos)")
            traceback.print_exc(limit=3)
            notas = [float("nan")] * len(linhas)

        for linha, nota in zip(linhas, notas):
            linha[nome] = float(nota) if not eh_nan(nota) else float("nan")
    return linhas


# ---------------------------------------------------------------------------
# Análise
# ---------------------------------------------------------------------------

def resumir(df):
    """Médias por configuração de chunk."""
    resumo = {}
    for tam in COLECOES:
        d = df[df["chunk"] == tam]
        resumo[tam] = {
            "faithfulness": d["faithfulness"].mean(),
            "answer_relevancy": d["answer_relevancy"].mean(),
            "contexto_medio_chars": d["contexto_medio_chars"].mean(),
            "acerto_fonte": d["acerto_fonte"].mean(),
            "nan": int(d[METRICAS].isna().sum().sum()),
        }
    return resumo


def escolher_melhor(resumo):
    """Faithfulness decide; se empatar (margem), answer_relevancy desempata."""
    a, b = list(COLECOES)
    if eh_nan(resumo[a]["faithfulness"]) or eh_nan(resumo[b]["faithfulness"]):
        return None, "dados insuficientes (faithfulness n/d em alguma configuração)"
    dif_f = resumo[a]["faithfulness"] - resumo[b]["faithfulness"]
    if abs(dif_f) > MARGEM_EMPATE:
        return (a if dif_f > 0 else b), "maior faithfulness média"
    if eh_nan(resumo[a]["answer_relevancy"]) or eh_nan(resumo[b]["answer_relevancy"]):
        return None, "faithfulness empatada e answer_relevancy n/d"
    dif_r = resumo[a]["answer_relevancy"] - resumo[b]["answer_relevancy"]
    if abs(dif_r) > MARGEM_EMPATE:
        return (a if dif_r > 0 else b), "faithfulness empatada; desempate por answer_relevancy"
    return None, "empate técnico nas duas métricas"


def status_alvo(media):
    if eh_nan(media):
        return "sem dados"
    if media >= ZONA_IDEAL_FAITHFULNESS:
        return f"ATINGIU o alvo (≥ {ALVO_FAITHFULNESS}) e está na zona ideal (≥ {ZONA_IDEAL_FAITHFULNESS})"
    if media >= ALVO_FAITHFULNESS:
        return f"ATINGIU o alvo (≥ {ALVO_FAITHFULNESS})"
    if media >= LIMITE_ALUCINACAO:
        return f"NÃO atingiu o alvo (≥ {ALVO_FAITHFULNESS})"
    return f"NÃO atingiu o alvo e está abaixo de {LIMITE_ALUCINACAO} (alucinação significativa)"


def sim_nao(media):
    return "n/d" if eh_nan(media) else ("SIM" if media >= ALVO_FAITHFULNESS else "NÃO")


def tabela_md(cabecalho, linhas):
    saida = ["| " + " | ".join(cabecalho) + " |", "| " + " | ".join("---" for _ in cabecalho) + " |"]
    saida += ["| " + " | ".join(str(c) for c in linha) + " |" for linha in linhas]
    return "\n".join(saida)


# ---------------------------------------------------------------------------
# Saídas: CSVs e relatório
# ---------------------------------------------------------------------------

def salvar_csvs(df, resumo, melhor, motivo):
    a, b = list(COLECOES)

    # 1) formato longo: uma linha por (chunk, pergunta)
    df.drop(columns=["retrieved_contexts"]).to_csv("resultados_ragas.csv", **CSV_KW)

    # 2) comparação: uma linha por pergunta, 512 x 1024 lado a lado
    linhas = []
    for q in PERGUNTAS:
        da = df[(df["chunk"] == a) & (df["id"] == q["id"])].iloc[0]
        db = df[(df["chunk"] == b) & (df["id"] == q["id"])].iloc[0]
        linha = {"id": q["id"], "categoria": q["categoria"], "pergunta": q["pergunta"]}
        for m in METRICAS:
            linha[f"{m}_{a}"] = da[m]
            linha[f"{m}_{b}"] = db[m]
        linha[f"fonte_esperada_{a}"] = "SIM" if da["acerto_fonte"] else "NÃO"
        linha[f"fonte_esperada_{b}"] = "SIM" if db["acerto_fonte"] else "NÃO"
        linha[f"resposta_{a}"] = da["response"]
        linha[f"resposta_{b}"] = db["response"]
        linhas.append(linha)

    media = {"id": "MEDIA", "categoria": "", "pergunta": ""}
    for m in METRICAS:
        media[f"{m}_{a}"] = resumo[a][m]
        media[f"{m}_{b}"] = resumo[b][m]
    linhas.append(media)
    pd.DataFrame(linhas).round(4).to_csv("comparacao_ragas.csv", **CSV_KW)

    # 3) resumo por configuração
    resumo_linhas = []
    for t in COLECOES:
        resumo_linhas.append({
            "configuracao": f"chunk {t}",
            "faithfulness_media": resumo[t]["faithfulness"],
            "answer_relevancy_media": resumo[t]["answer_relevancy"],
            "faithfulness_ge_0_7": sim_nao(resumo[t]["faithfulness"]),
            "fonte_esperada_recuperada": resumo[t]["acerto_fonte"],
            "contexto_medio_chars": resumo[t]["contexto_medio_chars"],
            "metricas_n_d": resumo[t]["nan"],
            "melhor_configuracao": "SIM" if melhor == t else ("EMPATE/INCONCLUSIVO" if melhor is None else "NÃO"),
        })
    pd.DataFrame(resumo_linhas).round(4).to_csv("resumo_ragas.csv", **CSV_KW)


def montar_relatorio(df, resumo, melhor, motivo, k):
    a, b = list(COLECOES)

    por_pergunta = []
    for q in PERGUNTAS:
        da = df[(df["chunk"] == a) & (df["id"] == q["id"])].iloc[0]
        db = df[(df["chunk"] == b) & (df["id"] == q["id"])].iloc[0]
        por_pergunta.append([
            q["id"], q["categoria"], q["pergunta"],
            fmt(da["faithfulness"]), fmt(db["faithfulness"]),
            fmt(da["answer_relevancy"]), fmt(db["answer_relevancy"]),
        ])
    t_perguntas = tabela_md(
        ["ID", "Categoria", "Pergunta", f"Faith. {a}", f"Faith. {b}", f"Relev. {a}", f"Relev. {b}"],
        por_pergunta,
    )

    t_medias = tabela_md(
        ["Configuração", "Faithfulness (média)", "Answer relevancy (média)",
         "Fonte esperada recuperada", "Tamanho médio do contexto (chars)", "Faithfulness ≥ 0,7?"],
        [[f"chunk {t}", fmt(resumo[t]["faithfulness"]), fmt(resumo[t]["answer_relevancy"]),
          f"{resumo[t]['acerto_fonte']:.0%}", f"{resumo[t]['contexto_medio_chars']:.0f}",
          sim_nao(resumo[t]["faithfulness"])] for t in COLECOES],
    )

    f_a, f_b = resumo[a]["faithfulness"], resumo[b]["faithfulness"]
    r_a, r_b = resumo[a]["answer_relevancy"], resumo[b]["answer_relevancy"]
    interp = [
        f"- **Faithfulness:** chunks de {a} obtiveram média {fmt(f_a)} e chunks de {b} média {fmt(f_b)}.",
        f"- **Answer relevancy:** chunks de {a} obtiveram média {fmt(r_a)} e chunks de {b} média {fmt(r_b)}.",
        f"- **Alvo de faithfulness ≥ {ALVO_FAITHFULNESS}:** chunk {a} {status_alvo(f_a)}; chunk {b} {status_alvo(f_b)}.",
    ]
    if melhor is None:
        interp.append(f"- **Melhor configuração:** inconclusivo — {motivo}.")
    else:
        interp.append(f"- **Melhor configuração: chunks de {melhor}** ({motivo}).")
        if melhor == b:
            hip = ("chunks maiores entregam mais contexto contínuo ao LLM, o que tende a reduzir "
                   "afirmações sem suporte; em contrapartida, trazem mais texto irrelevante junto.")
        else:
            hip = ("chunks menores são mais focados no assunto da pergunta, o que tende a reduzir "
                   "ruído; em contrapartida, podem cortar a explicação no meio.")
        interp.append(f"- **Por quê (hipótese a confirmar nas respostas):** {hip} "
                      f"Contexto médio recuperado: {resumo[a]['contexto_medio_chars']:.0f} chars ({a}) "
                      f"x {resumo[b]['contexto_medio_chars']:.0f} chars ({b}).")
    for t in COLECOES:
        d = df[df["chunk"] == t].dropna(subset=["faithfulness"])
        if not d.empty:
            pior = d.loc[d["faithfulness"].idxmin()]
            interp.append(f"- Pior faithfulness no chunk {t}: {pior['id']} ({pior['categoria']}) = "
                          f"{fmt(pior['faithfulness'])}. Vale ler essa resposta e os trechos recuperados.")
    if any(resumo[t]["nan"] for t in COLECOES):
        interp.append("- Atenção: algumas métricas vieram como n/d (falha do juiz, geralmente 429/503 da "
                      "Ollama Cloud ou saída fora do formato). Rode de novo antes de reportar as médias.")
    interp.append(f"- Limitações: {len(PERGUNTAS)} perguntas, k={k}, e o juiz do RAGAS é o mesmo modelo "
                  "que gera as respostas (gemma4:cloud), o que pode favorecer as notas.")

    return "\n\n".join([
        "## Resultados RAGAS", "### Por pergunta", t_perguntas,
        "### Médias por configuração", t_medias, "### Interpretação", "\n".join(interp),
    ]) + "\n"


def avaliar_e_salvar(linhas, k):
    preparar_asyncio()
    llm_juiz, embeddings_juiz = criar_juiz()
    metricas = criar_metricas()

    avaliadas = []
    for tamanho in COLECOES:
        grupo = [l for l in linhas if l["chunk"] == tamanho]
        print(f"Avaliando com RAGAS (chunk {tamanho})...")
        avaliadas.extend(avaliar_linhas(grupo, llm_juiz, embeddings_juiz, metricas))

    df = pd.DataFrame(avaliadas)
    resumo = resumir(df)
    melhor, motivo = escolher_melhor(resumo)

    salvar_csvs(df, resumo, melhor, motivo)
    relatorio = montar_relatorio(df, resumo, melhor, motivo, k)
    with open("resultados_ragas.md", "w", encoding="utf-8") as arquivo:
        arquivo.write(relatorio)

    print("\n" + relatorio)
    print("Arquivos gerados: comparacao_ragas.csv, resumo_ragas.csv, resultados_ragas.csv, resultados_ragas.md")


# ---------------------------------------------------------------------------
# Diagnóstico: testa cada peça isolada e mostra qual falha
# ---------------------------------------------------------------------------

def diagnostico():
    from importlib.metadata import version

    def passo(nome, funcao):
        try:
            detalhe = funcao()
            print(f"[OK]     {nome}" + (f" -> {detalhe}" if detalhe else ""))
            return True
        except Exception as erro:
            print(f"[FALHOU] {nome}: {type(erro).__name__}: {erro}")
            return False

    def versoes():
        pacotes = ["ragas", "langchain-ollama", "langchain-chroma", "chromadb", "pandas"]
        saida = []
        for p in pacotes:
            try:
                saida.append(f"{p}=={version(p)}")
            except Exception:
                saida.append(f"{p}=NÃO INSTALADO")
        return ", ".join(saida)

    def colecoes():
        import chromadb
        from app.vector_store import PASTA_CHROMA
        cliente = chromadb.PersistentClient(path=PASTA_CHROMA)
        existentes = [c.name for c in cliente.list_collections()]
        partes = []
        for tam, nome in COLECOES.items():
            if nome not in existentes:
                raise RuntimeError(f"coleção '{nome}' não existe. Rode: python -m app.vector_store")
            partes.append(f"{nome}={cliente.get_collection(nome).count()} chunks")
        return ", ".join(partes)

    def embeddings_locais():
        from app.vector_store import criar_embeddings
        return f"vetor com {len(criar_embeddings().embed_query('teste'))} números"

    def busca():
        from app.vector_store import buscar
        return f"{len(buscar('orçamento pessoal', 512, 1))} resultado(s)"

    def llm_cloud():
        from app.rag import criar_llm
        return criar_llm().invoke("Responda apenas: ok").content.strip()[:40]

    def imports_ragas():
        criar_metricas()
        from ragas import EvaluationDataset, evaluate  # noqa: F401
        from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: F401
        from ragas.llms import LangchainLLMWrapper  # noqa: F401
        from ragas.run_config import RunConfig  # noqa: F401
        return "imports ok"

    def juiz():
        llm, _ = criar_juiz()
        return "juiz criado"

    print(f"Pasta atual: {os.getcwd()} (rode da raiz do projeto)\n")
    passo("Versões instaladas", versoes)
    passo("OLLAMA_API_KEY", lambda: "encontrada" if obter_api_key() else "vazia")
    passo("ChromaDB: coleções 512 e 1024", colecoes)
    passo("Ollama LOCAL: embeddings nomic-embed-text", embeddings_locais)
    passo("Busca semântica (buscar)", busca)
    passo("LLM gemma4:cloud responde", llm_cloud)
    passo("Imports do ragas", imports_ragas)
    passo("Criação do juiz RAGAS", juiz)


# ---------------------------------------------------------------------------
# Execução
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Avaliação RAGAS do pipeline RAG (M6)")
    parser.add_argument("--k", type=int, default=3, help="quantidade de chunks recuperados")
    parser.add_argument("--diagnostico", action="store_true", help="testa cada peça e mostra qual falha")
    parser.add_argument("--so-gerar", action="store_true", help="só gera as respostas do RAG (sem ragas)")
    parser.add_argument("--reaproveitar", action="store_true", help=f"reusa {ARQUIVO_RESPOSTAS} em vez de gerar de novo")
    args = parser.parse_args()

    if args.diagnostico:
        diagnostico()
    else:
        if args.reaproveitar and os.path.exists(ARQUIVO_RESPOSTAS):
            with open(ARQUIVO_RESPOSTAS, encoding="utf-8") as arquivo:
                linhas_geradas = json.load(arquivo)
            print(f"Reaproveitando {len(linhas_geradas)} respostas de {ARQUIVO_RESPOSTAS}")
        else:
            linhas_geradas = gerar_todas(args.k)

        if not args.so_gerar:
            avaliar_e_salvar(linhas_geradas, args.k)
