from __future__ import annotations

import csv
import re
import time
from pathlib import Path
from typing import Optional, List

import tiktoken
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from ollama import ResponseError

# Módulos do pacote do projeto FIAP (app/)
from app.chain import llm, fincontrol_prompt
from app.memory_manager import criar_memoria

# ============================================================================
# 1. SCHEMA PYDANTIC V2 (>= 4 CAMPOS TIPADOS - EXIGÊNCIA CHECKPOINT 01)
# ============================================================================
class AnaliseContextRot(BaseModel):
    """Schema Pydantic v2 para validação e extração estruturada no teste de context rot."""
    turno: int = Field(description="Número do turno atual da conversa")
    meta_mencionada: bool = Field(description="Indica se a meta mensal de economia foi identificada")
    valor_observado: Optional[float] = Field(
        default=None, 
        description="Valor numérico absoluto da meta de economia identificada (ex: 850.0)"
    )
    resumo_resposta: str = Field(description="Resumo claro da resposta fornecida pelo modelo")

# Parser Pydantic da Aula 03
parser_pydantic = PydanticOutputParser(pydantic_object=AnaliseContextRot)

# ============================================================================
# 2. CONTEXTO E CONFIGURAÇÕES DO EXPERIMENTO
# ============================================================================
contexto_financeiro = { 
    "saldo_atual": 2450.30, 
    "transacoes": [ 
        {"id_transacao": "t1", "data": "2026-09-02", "valor": 450.00, "tipo": "despesa", "categoria": "alimentação", "descricao": "Supermercado"}, 
        {"id_transacao": "t2", "data": "2026-09-10", "valor": 120.00, "tipo": "despesa", "categoria": "alimentação", "descricao": "Delivery"}, 
        {"id_transacao": "t3", "data": "2026-09-05", "valor": 800.00, "tipo": "despesa", "categoria": "moradia", "descricao": "Aluguel"}, 
        {"id_transacao": "t4", "data": "2026-09-08", "valor": 300.00, "tipo": "despesa", "categoria": "transporte", "descricao": "Combustível"}, 
    ], 
    "orcamentos": [ 
        {"categoria": "alimentação", "limite_mensal_estipulado": 600.00, "valor_gasto_atual": 570.00, "percentual_consumido": 95.0} 
    ], 
    "faturas": [ 
        {"descricao_conta": "Cartão de Crédito", "data_vencimento": "2026-09-20", "valor": 1250.00, "status": "pendente", "nivel_alerta": "critico"} 
    ], 
}

mensagem_situacao_inicial = ( 
    "Quero registrar uma informação importante para esta conversa. " 
    "Minha meta mensal de economia é de R$ 850,00. " 
    "Essa é uma meta pessoal que defini para este mês e quero que você " 
    "considere esse valor quando eu voltar a falar sobre minha meta."
)

mensagens_preenchimento = [ 
    "Quero analisar melhor meus gastos com alimentação. Tenho despesas com supermercado e delivery, quero entender o impacto.",
    "Também estou tentando entender minhas despesas de moradia. Quero separar custo fixo de variável para organizar meu orçamento.",
    "Sobre transporte, gostaria de acompanhar os gastos e identificar mudanças relevantes em relação ao meu histórico.",
    "Quero acompanhar minhas contas com vencimento próximo para evitar atrasos e juros.",
    "Estou tentando entender a diferença entre despesas já registradas e valores de faturas pendentes no cartão.",
    "Quero acompanhar a evolução dos meus gastos ao longo do mês com comparações por categoria.",
    "Prefiro que os valores sejam apresentados em reais (R$) e que avise se alguma informação não estiver no contexto.",
    "Se houver alguma categoria próxima do limite do orçamento, destaque antes de ultrapassar o valor planejado."
]

pergunta_padrao = "Qual era exatamente a minha meta mensal de economia que mencionei no início da conversa?"

# Parâmetros exigidos no PDF (Janelas de 800-1500 tokens)
JANELAS_TOKENS = [800, 1000, 1200, 1500]
TOTAL_TURNOS = 35
CHECKPOINTS = {5, 10, 15, 20, 25, 30, 35}
VALOR_ESPERADO = 850.0

# ============================================================================
# 3. METRICAS DE TOKENS (TIKTOKEN - DIFERENCIAL DO PDF)
# ============================================================================
def contar_tokens_tiktoken(texto: str, modelo: str = "cl100k_base") -> int:
    """Conta tokens usando tiktoken como diferencial de precisão técnica."""
    try:
        encoding = tiktoken.get_encoding(modelo)
        return len(encoding.encode(texto))
    except Exception:
        # Fallback de contagem via LLM se o tiktoken falhar
        return llm.get_num_tokens(texto) if hasattr(llm, "get_num_tokens") else len(texto.split())

def tokens_da_memoria(memoria) -> int:
    variaveis = memoria.load_memory_variables({}) 
    mensagens = variaveis.get("history", []) 
    texto = "\n".join(str(getattr(msg, "content", msg)) for msg in mensagens)
    return contar_tokens_tiktoken(texto)

# ============================================================================
# 4. EXTRAÇÃO RESILIENTE DE VALORES
# ============================================================================
def extrair_valores_observados(resposta: str) -> list[float]:
    """Extrai valores monetários/numéricos da resposta de forma resiliente."""
    if not isinstance(resposta, str):
        resposta = str(resposta)

    valores: list[float] = []
    padroes = [
        r"R\$\s*([\d\.\,]+)",          # ex: R$ 850,00 ou R$ 850
        r"([\d\.\,]+)\s*(?:reais|BRL)", # ex: 850 reais
        r"\b(\d{3,4}(?:[\,\.]\d{2})?)\b"# ex: 850 ou 850.00 isolados
    ]

    for padrao in padroes:
        resultados = re.findall(padrao, resposta, flags=re.IGNORECASE)
        for val_str in resultados:
            try:
                clean_val = val_str.strip()
                if "." in clean_val and "," in clean_val:
                    clean_val = clean_val.replace(".", "").replace(",", ".")
                elif "," in clean_val:
                    clean_val = clean_val.replace(",", ".")
                
                val_float = float(clean_val)
                if val_float not in valores:
                    valores.append(val_float)
            except ValueError:
                continue

    return valores

def avaliar_resposta(resposta: str) -> int:
    valores = extrair_valores_observados(resposta)
    for valor in valores:
        if abs(valor - VALOR_ESPERADO) <= 0.1:
            return 1
    return 0

# ============================================================================
# 5. EXECUÇÃO DO EXPERIMENTO COM RETRY E LCEL
# ============================================================================
def invocar_com_retry(chain, dados, tentativas=3, espera=5):
    for tentativa in range(1, tentativas + 1):
        try:
            return chain.invoke(dados)
        except ResponseError as erro:
            if getattr(erro, "status_code", None) == 503 and tentativa < tentativas:
                print(f"    503 Ollama Cloud. Tentativa {tentativa}/{tentativas}. Aguardando {espera}s...")
                time.sleep(espera)
            else:
                raise

def executar_turno(memoria, mensagem: str) -> str:
    variaveis = memoria.load_memory_variables({"user_input": mensagem})
    cadeia_experimental = fincontrol_prompt | llm

    resposta = invocar_com_retry(
        cadeia_experimental,
        {
            "history": variaveis.get("history", []),
            "context": contexto_financeiro,
            "user_input": mensagem,
        }
    )

    resposta_texto = str(getattr(resposta, "content", resposta))

    memoria.save_context(
        {"user_input": mensagem},
        {"response": resposta_texto},
    )

    return resposta_texto

def rodar_experimento() -> list[dict]:
    resultados: list[dict] = []

    for janela in JANELAS_TOKENS:
        print("\n" + "=" * 72)
        print(f"TESTE DE CONTEXT ROT - JANELA DE MEMÓRIA: {janela} TOKENS")
        print("=" * 72)

        memoria = criar_memoria(llm=llm, max_token_limit=janela)
        executar_turno(memoria, mensagem_situacao_inicial)

        idx_preenchimento = 0

        for turno in range(2, TOTAL_TURNOS + 1):
            if turno in CHECKPOINTS:
                pergunta = pergunta_padrao
            else:
                pergunta = mensagens_preenchimento[idx_preenchimento % len(mensagens_preenchimento)]
                idx_preenchimento += 1

            tokens_hist_antes = tokens_da_memoria(memoria)
            resposta = executar_turno(memoria, pergunta)
            tokens_resposta = contar_tokens_tiktoken(resposta)

            if turno in CHECKPOINTS:
                qualidade = avaliar_resposta(resposta)
                valores_observados = extrair_valores_observados(resposta)

                resultado = {
                    "window_tokens": janela,
                    "turno": turno,
                    "history_tokens": tokens_hist_antes,
                    "response_tokens": tokens_resposta,
                    "question": pergunta,
                    "response": resposta,
                    "expected_value": VALOR_ESPERADO,
                    "observed_values": valores_observados,
                    "quality_score": qualidade,
                }

                resultados.append(resultado)
                print(f"[Checkpoint — Turno {turno:>2}] Hist Tokens: {tokens_hist_antes:>4} | Observados: {valores_observados} | Qualidade: {qualidade}")

    return resultados

# ============================================================================
# 6. EXPORTAÇÃO DE RESULTADOS E RESUMO DE DESEMPENHO
# ============================================================================
def salvar_csv(resultados: list[dict], caminho: str = "resultados_context_rot.csv") -> None:
    caminho_path = Path(caminho)
    campos = [
        "window_tokens", "turno", "history_tokens", "response_tokens",
        "question", "response", "expected_value", "observed_values", "quality_score"
    ]

    with caminho_path.open("w", newline="", encoding="utf-8") as arquivo:
        writer = csv.DictWriter(arquivo, fieldnames=campos)
        writer.writeheader()
        for res in resultados:
            linha = res.copy()
            linha["observed_values"] = ", ".join(f"{v:.2f}" for v in res["observed_values"])
            writer.writerow(linha)

    print(f"\n[+] Resultados exportados para: {caminho_path.resolve()}")

def imprimir_resumo(resultados: list[dict]) -> None:
    print("\n" + "=" * 72)
    print("TABELA COMPARATIVA DE DEGRADAÇÃO (CONTEXT ROT)")
    print("=" * 72)

    for janela in JANELAS_TOKENS:
        resultados_janela = [r for r in resultados if r["window_tokens"] == janela]
        acertos = sum(r["quality_score"] for r in resultados_janela)
        total = len(resultados_janela)
        precisao = (acertos / total) * 100 if total else 0.0

        print(
            f"Janela: {janela:>4} tokens | "
            f"Acertos nos Checkpoints: {acertos}/{total} | "
            f"Taxa de Retenção: {precisao:.1f}%"
        )

if __name__ == "__main__":
    resultados = rodar_experimento()
    salvar_csv(resultados)
    imprimir_resumo(resultados)