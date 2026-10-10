from pydantic import BaseModel, Field
from typing import List, Optional

from .Transacao import Transacao

class Historico(BaseModel):
    
  saldo_atual:            float | None       = Field(default = None, description="Saldo atual do usuário, quando relevante para a consulta")
  periodo:                str                = Field(description="Período de consulta do histórico")
  valor_movimentado:      float              = Field(description="Valor movimentado no período avaliado")
  quantidade_transacoes:  int                = Field(description="Total de transações realizadas no período avaliado")
  transacoes:             List[Transacao]    = Field(description="Lista de objetos referentes às transações efetuadas")