from contextlib import asynccontextmanager
from typing import AsyncIterator, List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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
    
    # Configure CORS middleware
    origins = [
        "http://localhost:5174",
        "http://localhost:5173",
        "https://risolutorematematico.it",
        "https://www.risolutorematematico.it",
    ]
    
    application.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    add_logging_middleware(application)
    application.include_router(solve_router, prefix="/api")
    return application


app = create_app()



