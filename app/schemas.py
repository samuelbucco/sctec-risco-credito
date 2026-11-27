"""Contratos de entrada e saída apresentados também no Swagger UI."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class SolicitacaoCredito(BaseModel):
    model_config = ConfigDict(
        extra="forbid", strict=True, allow_inf_nan=False, str_strip_whitespace=True,
        json_schema_extra={"examples": [{
            "person_age": 35, "person_income": 80000, "person_emp_length": 10,
            "loan_amnt": 6000, "loan_int_rate": 8,
            "cb_person_cred_hist_length": 10, "person_home_ownership": "OWN",
            "loan_intent": "PERSONAL", "loan_grade": "A", "cb_person_default_on_file": "N",
        }]},
    )

    person_age: int = Field(ge=18, le=100, description="Idade em anos; intervalo adotado no projeto.")
    person_income: float = Field(gt=0, description="Renda anual, na mesma moeda do empréstimo.")
    loan_amnt: float = Field(gt=0, description="Valor total do empréstimo.")
    person_emp_length: float | None = Field(default=None, ge=0, description="Tempo de emprego; ausente usa a imputação salva.")
    loan_int_rate: float | None = Field(default=None, gt=0, description="Taxa na escala da base; unidade e período não confirmados.")
    cb_person_cred_hist_length: float | None = Field(default=None, ge=0, description="Tempo de histórico de crédito, interpretado em anos.")
    person_home_ownership: str | None = Field(default=None, min_length=1, description="Ex.: RENT, OWN, MORTGAGE ou OTHER; categorias novas são aceitas.")
    loan_intent: str | None = Field(default=None, min_length=1, description="Ex.: PERSONAL, MEDICAL, EDUCATION; categorias novas são aceitas.")
    loan_grade: str | None = Field(default=None, min_length=1, description="Nota do empréstimo; ex.: A, B, C. Categorias novas são aceitas.")
    cb_person_default_on_file: Literal["Y", "N"] | None = Field(default=None, description="Histórico de inadimplência: Y ou N.")

    @field_validator("person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file", mode="before")
    @classmethod
    def normalizar_categorias(cls, valor):
        return valor.strip().upper() if isinstance(valor, str) else valor

    @model_validator(mode="after")
    def validar_tempos(self):
        for campo in ["person_emp_length", "cb_person_cred_hist_length"]:
            valor = getattr(self, campo)
            if valor is not None and valor > self.person_age:
                raise ValueError(f"{campo} não pode ser maior que person_age.")
        return self


class PrevisaoCredito(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    probabilidade_inadimplencia: float = Field(ge=0, le=1, description="Score estimado por predict_proba, sem calibração probabilística.")
    classe_prevista: Literal[0, 1]
    classificacao_risco: Literal["baixo", "elevado"]
    acao_recomendada: str
    threshold: float = Field(ge=0, le=1)
    modelo: str


class SaudeAPI(BaseModel):
    status: Literal["ok"]
    modelo: str
    threshold: float = Field(ge=0, le=1)
