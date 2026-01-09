from fastapi import FastAPI
from mangum import Mangum

from src.routes import portfolio_calculator_router

app = FastAPI(title="AnthonyFinance Backend")

app.include_router(portfolio_calculator_router.router, prefix="/tools", tags=["Portfolio Calculator"])

@app.get("/")
def read_root():
    return {"Hello": "World"}

handler = Mangum(app)
