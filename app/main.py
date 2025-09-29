from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from app.routes.solve_routes import router as solve_router
from app.services.openai_service import init_openai_client, shutdown_openai_client
from app.middleware.logging import add_logging_middleware


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    await init_openai_client()
    try:
        yield
    finally:
        await shutdown_openai_client()


def create_app() -> FastAPI:
    application = FastAPI(title="Math Solver API", version="0.1.0", lifespan=lifespan)
    add_logging_middleware(application)
    application.include_router(solve_router, prefix="/api")
    return application


app = create_app()



