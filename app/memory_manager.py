from __future__ import annotations

import json
from typing import Any

from langchain_classic.memory import ConversationTokenBufferMemory

CHAVE_HISTORICO = "history"
CHAVE_CONTEXTO = "context"
CHAVE_ENTRADA = "user_input"
CHAVE_SAIDA = "response"

# Estratégia adotada e limite de tokens do histórico.
NOME_ESTRATEGIA = "token_buffer"
LIMITE_TOKENS_PADRAO = 1000

# Faixa exigida para memória gerenciada.
FAIXA_LIMITE_TOKENS_CKP = (800, 1500)

JUSTIFICATIVA_DA_MEMORIA = """\
Estratégia escolhida: ConversationTokenBufferMemory com max_token_limit=1000.

Por quê, no domínio de organização financeira pessoal:
- A conversa é do tipo consulta curta e encadeada ("quanto gastei com
  alimentação?", "e dá pra pedir delivery hoje?"). O que importa são os
  últimos turnos, para resolver referências como "aquela fatura" ou "esse
  gasto". O usuário raramente volta à primeira mensagem da sessão.
- Os fatos financeiros não dependem da memória: saldo, transações,
  orçamentos e faturas são reinjetados a cada chamada na variável {context}
  do prompt. Então a memória só precisa cobrir o diálogo, e perder o início
  da conversa não faz o assistente errar um valor.

Por que não BufferMemory:
- O custo cresce a cada turno, sem teto. Numa sessão longa de acompanhamento
  de gastos, o histórico reenviado encareceria cada chamada progressivamente.

Por que não SummaryMemory:
- Faria uma chamada extra ao LLM por turno só para resumir, e resumo é
  aproximado: pode trocar "R$ 450" por "cerca de R$ 400". Em um assistente
  financeiro, valor aproximado é pior do que valor ausente.
- Ela também não expõe parâmetro de limite de tokens, então não permitiria
  declarar nem demonstrar o teto pedido pelo checkpoint.

Por que 1000 tokens:
- Fica no meio da faixa 800–1500 exigida pelo checkpoint e mantém uma 
  janela recente do diálogo, limitada a 1000 tokens.
"""

def serializar_contexto_financeiro(context: dict | str | None) -> str:
    if context is None:
        return "Nenhum dado financeiro foi fornecido para esta interação."
    if isinstance(context, str):
        return context
    
    return json.dumps(context, ensure_ascii=False, indent=2, default=str)

class MemoriaFinanceira(ConversationTokenBufferMemory):
    current_context: str = ""

    @property
    def memory_variables(self) -> list[str]:
        # super() é a ConversationTokenBufferMemory, que devolve ["history"].
        # Resultado final: ["history", "context"].
        return [*super().memory_variables, CHAVE_CONTEXTO]

    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
        variaveis = super().load_memory_variables(inputs)
        variaveis[CHAVE_CONTEXTO] = self.current_context
        return variaveis


def criar_memoria(
    llm: Any,
    max_token_limit: int = LIMITE_TOKENS_PADRAO,
) -> MemoriaFinanceira:
    return MemoriaFinanceira(
        llm=llm,
        max_token_limit=max_token_limit,
        memory_key=CHAVE_HISTORICO,   # nome do MessagesPlaceholder no prompt
        input_key=CHAVE_ENTRADA,      # o que a memória deve gravar como pergunta
        output_key=CHAVE_SAIDA,       # o que a memória deve gravar como resposta
        return_messages=True,         # o prompt usa MessagesPlaceholder
    )

class GerenciadorDeMemoria:
    def __init__(
        self,
        llm: Any,
        max_token_limit: int = LIMITE_TOKENS_PADRAO,
    ) -> None:
        self.llm = llm
        self.max_token_limit = max_token_limit
        self.memory = criar_memoria(llm, max_token_limit=max_token_limit)

    def definir_contexto_financeiro(self, context: dict | str | None) -> str:
        texto = serializar_contexto_financeiro(context)
        self.memory.current_context = texto
        return texto

    def registrar_turno(self, user_input: str, resposta: str) -> None:
        self.memory.save_context(
            {CHAVE_ENTRADA: user_input},
            {CHAVE_SAIDA: resposta},
        )

    def limpar(self) -> None:
        self.memory.clear()
        self.memory.current_context = ""

    def carregar_variaveis(self) -> dict[str, Any]:
        return self.memory.load_memory_variables({})

    def mensagens(self) -> list:
        return list(self.memory.chat_memory.messages)

    def historico_em_texto(self) -> str:
        historico = self.carregar_variaveis().get(CHAVE_HISTORICO, [])
        if isinstance(historico, str):
            return historico
        return "\n".join(
            getattr(mensagem, "content", str(mensagem)) for mensagem in historico
        )

    def contar_tokens(self, texto: str) -> int:
        try:
            return int(self.llm.get_num_tokens(texto))
        except Exception:
            return max(1, len(texto) // 4)

    def tokens_no_historico(self) -> int:
        return self.contar_tokens(self.historico_em_texto())

    def diagnostico(self) -> dict[str, Any]:
        return {
            "estrategia": NOME_ESTRATEGIA,
            "max_token_limit": self.max_token_limit,
            "dentro_da_faixa_exigida": (
                FAIXA_LIMITE_TOKENS_CKP[0]
                <= self.max_token_limit
                <= FAIXA_LIMITE_TOKENS_CKP[1]
            ),
            "mensagens_no_historico": len(self.mensagens()),
            "tokens_no_historico": self.tokens_no_historico(),
            "tem_contexto_financeiro": bool(self.memory.current_context),
            "tokens_no_contexto_financeiro": self.contar_tokens(
                self.memory.current_context
            ),
        }

    def resumo_legivel(self) -> str:
        d = self.diagnostico()
        return (
            f"[memória: {d['estrategia']} | limite {d['max_token_limit']} tokens] "
            f"{d['mensagens_no_historico']} mensagens, "
            f"{d['tokens_no_historico']} tokens de histórico, "
            f"{d['tokens_no_contexto_financeiro']} tokens de dados financeiros"
        )


__all__ = [
    "CHAVE_CONTEXTO",
    "CHAVE_ENTRADA",
    "CHAVE_HISTORICO",
    "CHAVE_SAIDA",
    "FAIXA_LIMITE_TOKENS_CKP",
    "GerenciadorDeMemoria",
    "JUSTIFICATIVA_DA_MEMORIA",
    "LIMITE_TOKENS_PADRAO",
    "MemoriaFinanceira",
    "NOME_ESTRATEGIA",
    "criar_memoria",
    "serializar_contexto_financeiro",
]