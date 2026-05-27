from __future__ import annotations

import logging.config
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.exception_handlers import http_exception_handler
from starlette.exceptions import HTTPException as StarletteHTTPException
from ai_server.calorie.food_calorie import load_food_calorie_map
from ai_server.config import APP_NAME
from ai_server.controllers.dance_controller import router as dance_router
from ai_server.controllers.food_controller import router as food_router
from ai_server.food_ai.food_classifier import load_classifier_model


LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        },
    },
    "handlers": {
        "default": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "stream": "ext://sys.stdout",
        },
    },
    "root": {
        "level": "INFO",
        "handlers": ["default"],
    },
    "loggers": {
        "uvicorn": {
            "level": "INFO",
            "handlers": ["default"],
            "propagate": False,
        },
        "uvicorn.error": {
            "level": "INFO",
            "handlers": ["default"],
            "propagate": False,
        },
        "uvicorn.access": {
            "level": "INFO",
            "handlers": ["default"],
            "propagate": False,
        },
    },
}

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("initializing application resources")
    try:
        app.state.classifier_model = load_classifier_model()
        logger.info("food classification model loaded")

        app.state.pose_detectors = {}

        app.state.food_calorie_map = load_food_calorie_map()
        logger.info("food calorie map loaded")
    except Exception:
        logger.exception("failed to initialize application resources")
        raise

    logger.info("application resources initialized")
    try:
        yield
    finally:
        logger.info("shutting down application resources")

app = FastAPI(
    title=APP_NAME,
    version="1.0.0",
    description="AI server for Dance and Food analysis.",
    lifespan=lifespan,
)

app.include_router(dance_router)
app.include_router(food_router)


def get_http_route_summary() -> str:
    http_routes: list[str] = []

    for route in app.routes:
        methods = getattr(route, "methods", None)
        path = getattr(route, "path", None)
        if not methods or not path:
            continue

        http_routes.append(f"{','.join(sorted(methods))} {path}")

    return "; ".join(http_routes)


@app.exception_handler(StarletteHTTPException)
async def log_http_exception(request: Request, exc: StarletteHTTPException):
    if exc.status_code == 404:
        logger.warning(
            "route not found: method=%s path=%s registered_http_routes=%s",
            request.method,
            request.url.path,
            get_http_route_summary(),
        )

    return await http_exception_handler(request, exc)
