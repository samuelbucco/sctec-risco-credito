"""Servidor: python -m uvicorn app.main:app --reload."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.prediction import CAMINHO_MODELO, carregar_preditor
from app.schemas import PrevisaoCredito, SaudeAPI, SolicitacaoCredito


def criar_app(caminho_modelo: Path = CAMINHO_MODELO) -> FastAPI:
    @asynccontextmanager
    async def lifespan(aplicacao: FastAPI):
        aplicacao.state.preditor = carregar_preditor(caminho_modelo)
        try:
            yield
        finally:
            aplicacao.state.preditor = None

    aplicacao = FastAPI(
        title="Risco de Crédito SCTec", version="1.0.0", lifespan=lifespan,
        description=("Previsão educacional de inadimplência. O pipeline é carregado na inicialização "
                     "e usa o corte salvo, escolhido com custos fictícios. Os scores não foram "
                     "calibrados probabilisticamente. A documentação da base está no README."),
    )

    @aplicacao.exception_handler(RequestValidationError)
    async def erro_validacao(request: Request, erro: RequestValidationError):
        # Não devolva o valor bruto: NaN/Infinity não são serializáveis em JSON.
        detalhes = []
        for detalhe in erro.errors():
            detalhes.append({chave: detalhe[chave] for chave in ["loc", "msg", "type"]})
        return JSONResponse(status_code=422, content={"detail": detalhes})

    def obter_preditor(request: Request):
        preditor = getattr(request.app.state, "preditor", None)
        if preditor is None:
            raise HTTPException(status_code=503, detail="Modelo indisponível.")
        return preditor

    @aplicacao.get("/health", response_model=SaudeAPI, summary="Conferir se o modelo está carregado")
    def health(request: Request):
        preditor = obter_preditor(request)
        return SaudeAPI(status="ok", modelo=preditor.configuracao, threshold=preditor.threshold)

    @aplicacao.post("/predict", response_model=PrevisaoCredito, summary="Estimar risco de inadimplência")
    def predict(solicitacao: SolicitacaoCredito, request: Request):
        preditor = obter_preditor(request)
        try:
            return preditor.prever(solicitacao)
        except ValueError as erro:
            raise HTTPException(status_code=422, detail="Os valores informados não puderam ser processados pelo modelo.") from erro

    return aplicacao


app = criar_app()
