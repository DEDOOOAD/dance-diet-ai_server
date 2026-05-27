from __future__ import annotations

import uvicorn
from ai_server.config import HOST, PORT


if __name__ == "__main__":
    uvicorn.run("ai_server.server:app", host=HOST, port=PORT, reload=False)
