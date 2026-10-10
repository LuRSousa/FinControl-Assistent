# Teste do M5: roda o RAG completo (retrieve -> generate) com 512 e 1024 e mostra resposta + fontes.
# Rodar da raiz do projeto, depois de gerar o banco (M4):
#     python -m app.vector_store
#     python -m app.teste_rag                 # as duas configurações (512 e 1024)
#     python -m app.teste_rag --chunk 1024    # só uma configuração
#     python -m app.teste_rag --salvar        # grava resultados_rag_teste.json (útil para o M6)

import argparse
import json

from app.rag import RESPOSTA_SEM_CONTEXTO, formatar_resposta_com_fontes, responder
from app.vector_store import COLECOES

# Perguntas respondíveis pela base (as 5 primeiras) + 1 fora do escopo para checar se o RAG não inventa
PERGUNTAS = [
    "Quanto do meu salário eu deveria guardar todo mês?",
    "Quanto dinheiro devo ter separado pra imprevistos?",
    "Como eu descubro se estou ficando mais rico ou mais pobre?",
    "Qual a diferença entre despesas fixas e variáveis?",
    "O que fazer quando as despesas são maiores que a receita?",
]
PERGUNTA_FORA_DO_ESCOPO = "Qual a capital da Austrália?"


def testar(tamanhos, salvar=False):
    resultados = []

    for pergunta in PERGUNTAS + [PERGUNTA_FORA_DO_ESCOPO]:
        for tamanho in tamanhos:
            print("=" * 78)
            print(f"Pergunta: {pergunta}  [chunks de {tamanho}]")
            print("-" * 78)

            resultado = responder(pergunta, tamanho_chunk=tamanho)
            print(formatar_resposta_com_fontes(resultado))
            resultados.append(resultado)

            # Checagens automáticas simples
            if pergunta == PERGUNTA_FORA_DO_ESCOPO:
                ok = RESPOSTA_SEM_CONTEXTO in resultado["resposta"]
                print(f"\n>> Recusou corretamente (fora da base)? {'SIM' if ok else 'NÃO - revisar o prompt'}")
            else:
                cita = "[" in resultado["resposta"]
                print(f"\n>> Resposta cita os trechos [n]? {'SIM' if cita else 'NÃO - revisar o prompt'}")

    if salvar:
        with open("resultados_rag_teste.json", "w", encoding="utf-8") as arquivo:
            json.dump(resultados, arquivo, ensure_ascii=False, indent=2)
        print("\nResultados salvos em resultados_rag_teste.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Teste do pipeline RAG (M5)")
    parser.add_argument("--chunk", type=int, choices=sorted(COLECOES), help="testar só um tamanho de chunk")
    parser.add_argument("--salvar", action="store_true", help="salvar as respostas em JSON")
    args = parser.parse_args()

    testar([args.chunk] if args.chunk else list(COLECOES), salvar=args.salvar)
