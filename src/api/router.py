from fastapi import APIRouter
from src.api.v1 import auth
from src.api.v1 import chembot

api_router = APIRouter()

api_router.include_router(
    auth.router,
    prefix="/auth",
    tags=["Auth V1"]
)


api_router.include_router(
    chembot.router,
    prefix="/chembot",
    tags=["Chembot V1"]
)