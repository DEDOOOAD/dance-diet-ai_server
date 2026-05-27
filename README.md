# Dance-diet-AI_Server
AI Server 

Run the server from the project root so Python can resolve the `ai_server` package:

```powershell
uv run python main.py
```

For development with FastAPI reload:

```powershell
uv run uvicorn ai_server.server:app --host 0.0.0.0 --port 8001 --reload
```
