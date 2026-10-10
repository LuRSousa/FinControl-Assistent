import json
import gradio as gr
from app.chain import conversar, analisar_historico, memory

CONTEXTO_FINANCEIRO = {
    "saldo_atual": 2450.30,
    "renda_mensal": 6000.00,

    "transacoes": [
        {
            "id_transacao": "t1",
            "data": "2026-09-02",
            "valor": 450.00,
            "tipo": "despesa",
            "categoria": "alimentação",
            "descricao": "Supermercado",
        },
        {
            "id_transacao": "t2",
            "data": "2026-09-10",
            "valor": 120.00,
            "tipo": "despesa",
            "categoria": "alimentação",
            "descricao": "Delivery",
        },
        {
            "id_transacao": "t3",
            "data": "2026-09-12",
            "valor": 800.00,
            "tipo": "despesa",
            "categoria": "moradia",
            "descricao": "Aluguel",
        },
        {
            "id_transacao": "t4",
            "data": "2026-09-15",
            "valor": 300.00,
            "tipo": "despesa",
            "categoria": "transporte",
            "descricao": "Combustível",
        },
    ],

    "orcamentos": [
        {
            "categoria": "alimentação",
            "limite_mensal_estipulado": 600.00,
            "valor_gasto_atual": 570.00,
            "percentual_consumido": 95.0,
        }
    ],

    "faturas": [
        {
            "descricao_conta": "Cartão de Crédito",
            "data_vencimento": "2026-09-20",
            "valor": 1250.00,
            "status": "pendente",
            "nivel_alerta": "critico",
        }
    ],
}

def responder(user_input, history):
    """
    Recebe a mensagem da interface e envia para a
    ConversationChain com memória gerenciada.
    """

    if not user_input or not user_input.strip():
        return history, ""

    resposta = conversar(
        user_input=user_input,
        context=CONTEXTO_FINANCEIRO,
    )

    history = history or []

    history.append({
        "role": "user",
        "content": user_input,
    })

    history.append({
        "role": "assistant",
        "content": resposta,
    })

    return history, ""


def limpar_conversa():
    """
    Limpa a memória do chatbot e a conversa exibida no Gradio.
    """

    memory.clear()

    return []


def executar_analise():
    """
    Executa a segunda chain:
    LCEL + PydanticOutputParser.
    """

    historico = analisar_historico(
        user_input="Monte um resumo do meu histórico financeiro de setembro.",
        context=CONTEXTO_FINANCEIRO,
    )

    return json.dumps(
        historico.model_dump(),
        ensure_ascii=False,
        indent=2,
        default=str,
    )

with gr.Blocks(
    title="FinControl Assistant"
) as demo:

    gr.Markdown(
        """
        # FinControl Assistant

        Assistente de organização financeira pessoal.

        O chatbot utiliza **ConversationChain com memória TokenBuffer**
        e uma segunda pipeline **LCEL + Pydantic v2** para geração
        de dados estruturados.
        """
    )

    # --------------------------------------------------------
    # CHAT
    # --------------------------------------------------------

    chatbot = gr.Chatbot(
        label="Conversa",
        height=500,
    )

    with gr.Row():

        user_input = gr.Textbox(
            label="Mensagem",
            placeholder="Digite sua pergunta sobre suas finanças...",
            scale=4,
        )

        enviar = gr.Button(
            "Enviar",
            variant="primary",
            scale=1,
        )

    limpar = gr.Button("Limpar conversa")

    #Análise estruturada
    gr.Markdown("## Análise estruturada")

    gr.Markdown(
        """
        Executa a segunda pipeline LCEL e demonstra a validação
        da resposta utilizando **Pydantic v2**.
        """
    )

    analisar = gr.Button(
        "Analisar histórico"
    )

    resultado_analise = gr.Code(
        label="Saída Pydantic",
        language="json",
        interactive=False,
    )

    #Eventos
    enviar.click(
        fn=responder,
        inputs=[user_input, chatbot],
        outputs=[chatbot, user_input],
    )

    user_input.submit(
        fn=responder,
        inputs=[user_input, chatbot],
        outputs=[chatbot, user_input],
    )

    limpar.click(
        fn=limpar_conversa,
        inputs=[],
        outputs=[chatbot],
    )

    analisar.click(
        fn=executar_analise,
        inputs=[],
        outputs=[resultado_analise],
    )

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
    )