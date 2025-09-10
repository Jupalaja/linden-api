import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from src.api.chatflow.router import router as chatflow_router
from src.config import settings
from src.database.db import engine, test_db_connection
from src.shared.schemas import HealthResponse

log_level = settings.LOG_LEVEL.upper()
logging.basicConfig(
    level=log_level,
    format="%(levelname)s:%(name)s: [%(funcName)s] - %(message)s",
)

# httpx logs at INFO level for requests, which is noisy for production.
# We set it to WARNING to silence it, unless we are in DEBUG mode.
if log_level != "DEBUG":
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.debug("Starting up application...")
    if not await test_db_connection():
        logger.warning(
            "Database connection could not be established on startup."
        )
    else:
        logger.debug("Database connection successful.")

    yield
    # Shutdown
    logger.info("Shutting down application...")
    await engine.dispose()


app = FastAPI(title=settings.PROJECT_NAME, lifespan=lifespan)

app.include_router(chatflow_router, prefix="/api/v1", tags=["Chatflow"])


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check(request: Request):
    """
    Checks the health of the application and its database connection.
    """
    db_ok = await test_db_connection()
    return HealthResponse(
        status="ok",
        db_connection="ok" if db_ok else "failed",
    )
