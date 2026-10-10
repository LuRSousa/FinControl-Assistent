from __future__ import annotations

import json
import os

import tiktoken
from dotenv import load_dotenv

from langchain_classic.chains import ConversationChain
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from app.memory_manager import criar_memoria
from app.prompt import fincontrol_prompt
from app.schemas.Historico import Historico

load_dotenv()

OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")
if not OLLAMA_API_KEY:
    raise RuntimeError(
        "OLLAMA_API_KEY não encontrada. Copie .env para .env e "
        "preencha sua chave da Ollama Cloud antes de rodar o projeto."
    )

try:
    _token_encoding = tiktoken.get_encoding("cl100k_base")
except Exception:
    _token_encoding = None  # sem internet/cache: usa aproximação abaixo


class FinControlChatOllama(ChatOllama):
    """ChatOllama com contagem de tokens via tiktoken (Aula 04 — diferencial
    de context engineering), usada tanto para gerar respostas quanto para a
    memória TokenBuffer calcular o tamanho do histórico."""

    def get_num_tokens(self, text: str) -> int:
        if _token_encoding is not None:
            return len(_token_encoding.encode(text))
        return max(1, len(text) // 4)  # aproximação (~4 caracteres/token)


# LLM único, compartilhado pelas duas chains — exclusivamente gemma4:cloud
# via Ollama Cloud (nunca um modelo local/deprecated).
llm = FinControlChatOllama(
    model="gemma4:cloud",
    base_url="https://ollama.com",
    client_kwargs={"headers": {"Authorization": f"Bearer {OLLAMA_API_KEY}"}},
    temperature=0.4,
)


def _to_context_str(context: dict | str | None) -> str:
    """Serializa os dados financeiros (dict) para texto legível pelo LLM."""
    if context is None:
        return "Nenhum dado financeiro foi fornecido para esta interação."
    if isinstance(context, str):
        return context
    return json.dumps(context, ensure_ascii=False, indent=2, default=str)


# ==========================================================================
# CHAIN 1 — Conversa com memória gerenciada (ConversationChain, Aula 03)
# ==========================================================================
memory = criar_memoria(llm)


# Chain 1: ConversationChain "de fato" — obrigatória pelo checkpoint.
chat_chain = ConversationChain(
    llm=llm,
    memory=memory,
    prompt=fincontrol_prompt,   # definido em app/prompt.py (Aula 04, XML tags)
    input_key="user_input",
    output_key="response",
    verbose=False,
)


def conversar(user_input: str, context: dict | str | None = None) -> str:
    """
    Executa um turno de conversa livre com o FinControl Assistant,
    mantendo o histórico via memória gerenciada (Chain 1 — ConversationChain).
    """
    memory.current_context = _to_context_str(context)
    resultado = chat_chain.invoke({"user_input": user_input})
    return resultado["response"]


# ==========================================================================
# CHAIN 2 — Pipeline LCEL para saída estruturada (Pydantic v2)
# ==========================================================================
historico_parser = PydanticOutputParser(pydantic_object=Historico)

analise_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Você é o módulo analítico do FinControl Assistant.\n"
        "A partir dos DADOS FINANCEIROS BRUTOS abaixo e da consulta do "
        "usuário, monte um resumo estruturado do histórico financeiro do "
        "período pedido.\n\n"
        "Regras:\n"
        "- Use SOMENTE os dados fornecidos. Nunca invente transações, "
        "valores ou datas que não estejam no contexto.\n"
        "- Se um campo do schema não puder ser determinado a partir dos "
        "dados e ele for opcional, deixe-o vazio/nulo.\n\n"
        "DADOS FINANCEIROS BRUTOS:\n{context}\n\n"
        "{format_instructions}",
    ),
    ("human", "{user_input}"),
]).partial(format_instructions=historico_parser.get_format_instructions())

# Pipeline LCEL puro, com o operador | (obrigatório pelo checkpoint):
# prompt | llm | PydanticOutputParser
analise_chain = analise_prompt | llm | historico_parser


def analisar_historico(user_input: str, context: dict | str) -> Historico:
    """Executa a Chain 2 (LCEL) e devolve um `Historico` validado por Pydantic v2."""
    return analise_chain.invoke({
        "user_input": user_input,
        "context": _to_context_str(context),
    })