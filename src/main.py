from fastapi import FastAPI
from typing import Union

from src.routes import portfolio_calculator_router

app = FastAPI(title="AnthonyFinance Backend")

app.include_router(portfolio_calculator_router.router, prefix="/tools", tags=["Portfolio Calculator"])

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}