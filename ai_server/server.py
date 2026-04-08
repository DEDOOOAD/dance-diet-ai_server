from __future__ import annotations

import uvicorn
from fastapi import FastAPI

from ai_server.config import APP_NAME, HOST, PORT
from ai_server.controllers.dance_controller import router as dance_router
from ai_server.controllers.food_controller import router as food_router


app = FastAPI(
    title=APP_NAME,
    version="1.0.0",
    description="AI server for Dance and Food analysis.",
)

app.include_router(dance_router)
app.include_router(food_router)

if __name__ == "__main__":
    uvicorn.run("ai_server.server:app", host=HOST, port=PORT, reload=True)
