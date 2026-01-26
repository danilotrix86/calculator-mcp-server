from contextlib import asynccontextmanager
from typing import AsyncIterator, List
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.solve_routes import router as solve_router
from app.routes.health_routes import router as health_router
from app.routes.blog_routes import router as blog_router
from app.routes.admin_routes import router as admin_router
from app.routes.sitemap_routes import router as sitemap_router
from app.routes.matrix_routes import router as matrix_router
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
        "http://localhost:3000",  # Vike/Vercel dev server
        "http://localhost:5174",
        "http://localhost:5173",
        "https://risolutorematematico.it",
        "https://www.risolutorematematico.it",
    ]
    
    # Add Cloud Run service URL if provided via environment variable
    cloud_run_url = os.environ.get("CLOUD_RUN_URL")
    if cloud_run_url:
        origins.append(cloud_run_url)
    
    # Also add the specific Cloud Run URL (can be removed if using env var)
    origins.append("https://rismat-979168412861.europe-west1.run.app")
    
    application.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    add_logging_middleware(application)
    application.include_router(solve_router, prefix="/api")
    application.include_router(health_router, prefix="/api")
    application.include_router(blog_router, prefix="/api")
    application.include_router(admin_router, prefix="/api")
    application.include_router(matrix_router, prefix="/api")
    application.include_router(sitemap_router)  # No prefix - serves at root level
    return application


app = create_app()



