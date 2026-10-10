import os
import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# Modelo de embeddings: roda no Ollama local (antes: ollama pull nomic-embed-text)
MODELO_EMBEDDING = "nomic-embed-text"

# Banco local na raiz do projeto
PASTA_CHROMA = "./chroma_db"

# Uma coleção para cada configuração de chunk (o M6 compara as duas)
COLECOES = {
    512: "fin_control_512",
    1024: "fin_control_1024",
}


def criar_embeddings():
    # O mesmo modelo transforma os chunks e as perguntas em vetores.
    # Se trocar o modelo, é preciso gerar o banco de novo.
    return OllamaEmbeddings(model=MODELO_EMBEDDING)


def salvar_na_colecao(chroma_client, chunks, nome_colecao, embeddings):
    # Apaga a coleção antiga antes de salvar, senão rodar de novo duplica os chunks
    if nome_colecao in [colecao.name for colecao in chroma_client.list_collections()]:
        chroma_client.delete_collection(nome_colecao)

    # Gera o embedding de cada chunk e guarda junto o texto e os metadados (arquivo e página)
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        client=chroma_client,
        collection_name=nome_colecao,
        collection_metadata={"hnsw:space": "cosine"}  # compara os vetores por cosseno
    )
    return chroma_client.get_collection(nome_colecao).count()


def processar_e_armazenar_documentos():
    # Lendo os arquivos da pasta data/
    arquivos_pdf = [
        "data/DOC01_BCB_Caderno_Educacao_Financeira.pdf",
        "data/DOC02_CVM_TOP_Planejamento_Financeiro_Pessoal.pdf",
        "data/DOC03_CVM_Guia_Planejamento_Financeiro.pdf",
        "data/DOC04_CAIXA_Receitas_e_Despesas.pdf",
        "data/DOC05_SUSEP_Orcamento.pdf"
    ]

    documentos_totais = []
    for caminho in arquivos_pdf:
        loader = PyPDFLoader(caminho)
        documentos_totais.extend(loader.load())
        print(f"Carregado: {caminho}")

    # Configuração 1 (Tamanho 512)
    text_splitter_512 = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=51,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks_512 = text_splitter_512.split_documents(documentos_totais)

    # Configuração 2 (Tamanho 1024)
    text_splitter_1024 = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=102,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks_1024 = text_splitter_1024.split_documents(documentos_totais)

    print("Iniciando embeddings via Ollama (nomic-embed-text)...")
    embeddings = criar_embeddings()

    # Conectando ao banco local na raiz do projeto
    chroma_client = chromadb.PersistentClient(path=PASTA_CHROMA)

    # Armazenando a configuração 1 na coleção exclusiva para 512
    print(f"Armazenando chunks de 512 na coleção '{COLECOES[512]}'...")
    total_512 = salvar_na_colecao(chroma_client, chunks_512, COLECOES[512], embeddings)

    # Armazenando a configuração 2 na coleção exclusiva para 1024
    print(f"Armazenando chunks de 1024 na coleção '{COLECOES[1024]}'...")
    total_1024 = salvar_na_colecao(chroma_client, chunks_1024, COLECOES[1024], embeddings)

    print(f"Sistemas prontos! {total_512} chunks (512) e {total_1024} chunks (1024) foram armazenados com sucesso nas suas respectivas coleções.")


def carregar_vector_store(tamanho_chunk=512):
    # Abre uma coleção que já existe, sem gerar os embeddings de novo.
    # É por aqui que o retriever do M5 e os testes do M6 acessam o banco.
    if tamanho_chunk not in COLECOES:
        raise ValueError(f"Tamanho de chunk inválido: {tamanho_chunk}. Use 512 ou 1024.")

    chroma_client = chromadb.PersistentClient(path=PASTA_CHROMA)
    nome_colecao = COLECOES[tamanho_chunk]
    if nome_colecao not in [colecao.name for colecao in chroma_client.list_collections()]:
        raise RuntimeError(f"A coleção '{nome_colecao}' ainda não existe. Rode antes: python -m app.vector_store")

    return Chroma(
        client=chroma_client,
        collection_name=nome_colecao,
        embedding_function=criar_embeddings()
    )


def buscar(pergunta, tamanho_chunk=512, k=3):
    # Busca semântica: a pergunta vira um vetor e o Chroma devolve os k chunks mais próximos
    vector_store = carregar_vector_store(tamanho_chunk)
    resultados = vector_store.similarity_search_with_score(pergunta, k=k)

    # O Chroma devolve a distância de cosseno (0 = mesmo sentido).
    # Convertendo para similaridade: quanto mais perto de 1, mais parecido.
    return [(documento, 1 - distancia) for documento, distancia in resultados]


def nome_do_arquivo(documento):
    # Ex.: "data/DOC05_SUSEP_Orcamento.pdf" -> "DOC05_SUSEP_Orcamento.pdf"
    return os.path.basename(documento.metadata.get("source", "desconhecido"))


if __name__ == "__main__":
    processar_e_armazenar_documentos()
