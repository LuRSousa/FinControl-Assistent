from pydantic import BaseModel, Field
from typing import Literal, Optional

from datetime import date

class Transacao(BaseModel):
  data:           date                            = Field(description="Data da transação")
  valor:          float                           = Field(description="Valor da transação em R$")
  tipo:           Literal["despesa", "receita"]   = Field(description="Tipo da transação")
  categoria:      str                             = Field(description="Categoria da transação") 
  descricao:      Optional[str]                   = Field(default = None, description="Descrição opcional breve da transação")
  id_transacao:   str                             = Field(description="ID único da transação")
  saldo_restante: Optional[float]                 = Field(default=None, description="Saldo restante após transação")