from fastapi import FastAPI
from mangum import Mangum

from src.routes import EtfSimulationRouter, PortfolioSimulationRouter

app = FastAPI(title="AnthonyFinance Backend")

app.include_router(EtfSimulationRouter.router, prefix="/tools", tags=["ETF Calculator"])
app.include_router(PortfolioSimulationRouter.router, prefix="/tools", tags=["Portfolio Simulator"])

@app.get("/")
def read_root():
    return {"Health": "OK"}

handler = Mangum(app)
