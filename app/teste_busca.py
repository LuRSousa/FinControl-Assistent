# Teste do M4: confere o ChromaDB e faz buscas semânticas nas duas configurações de chunk.
# Rodar da raiz do projeto, depois de gerar o banco:
#     python -m app.vector_store
#     python -m app.teste_busca

import re
import chromadb
from app.vector_store import COLECOES, PASTA_CHROMA, buscar, nome_do_arquivo

# Perguntas do jeito que um usuário escreveria, sem copiar os termos dos PDFs
PERGUNTAS = [
    "Quanto do meu salário eu deveria guardar todo mês?",
    "Quanto dinheiro devo ter separado pra imprevistos?",
    "Como eu descubro se estou ficando mais rico ou mais pobre?",
]

# Palavras comuns que não dizem nada sobre o assunto
PALAVRAS_VAZIAS = {"a", "o", "as", "os", "de", "da", "do", "das", "dos", "e", "em", "no", "na",
                   "um", "uma", "eu", "me", "meu", "minha", "que", "se", "pra", "para", "por",
                   "com", "como", "quanto", "devo", "deveria", "ter", "estou", "ou", "mais"}


def conferir_colecoes():
    # Mostra quantos chunks cada coleção tem e o começo de um vetor salvo
    chroma_client = chromadb.PersistentClient(path=PASTA_CHROMA)
    existentes = [colecao.name for colecao in chroma_client.list_collections()]

    for tamanho, nome in COLECOES.items():
        if nome not in existentes:
            print(f"Coleção '{nome}' não encontrada. Rode antes: python -m app.vector_store")
            return False

        colecao = chroma_client.get_collection(nome)
        amostra = colecao.get(limit=1, include=["embeddings", "metadatas"])
        vetor = amostra["embeddings"][0]
        inicio = [round(float(numero), 3) for numero in vetor[:5]]
        print(f"{nome}: {colecao.count()} chunks | cada vetor tem {len(vetor)} números | começo de um vetor: {inicio}")
    return True


def palavras_em_comum(pergunta, texto):
    # Palavras inteiras da pergunta que também aparecem no trecho encontrado
    palavras_pergunta = {p.strip("?,.!").lower() for p in pergunta.split()} - PALAVRAS_VAZIAS
    palavras_trecho = set(re.findall(r"\w+", texto.lower()))
    return sorted(palavras_pergunta & palavras_trecho)


def mostrar_busca(pergunta, tamanho_chunk):
    print(f"\nPergunta: {pergunta}  [chunks de {tamanho_chunk}]")
    resultados = buscar(pergunta, tamanho_chunk=tamanho_chunk, k=3)

    for posicao, (documento, similaridade) in enumerate(resultados, start=1):
        pagina = documento.metadata.get("page", 0) + 1  # o PyPDFLoader conta a partir de 0
        trecho = " ".join(documento.page_content.split())[:180]
        comuns = palavras_em_comum(pergunta, documento.page_content)

        print(f"  {posicao}. similaridade {similaridade:.2f} | {nome_do_arquivo(documento)} | página {pagina}")
        print(f"     \"{trecho}...\"")
        print(f"     palavras da pergunta no trecho: {', '.join(comuns) if comuns else 'nenhuma'}")


if __name__ == "__main__":
    if conferir_colecoes():
        for pergunta in PERGUNTAS:
            for tamanho in COLECOES:
                mostrar_busca(pergunta, tamanho)
