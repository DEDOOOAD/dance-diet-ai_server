# Dance-diet-AI_Server
AI Server 

Run the server from the project root so Python can resolve the `ai_server` package:

- 실행 명령어
uv run python main.py

- .env 파일 구성
GEMINI_API_KEY=
GEMINI_FOOD_VISION_MODEL=gemini-2.5-flash-lite

For development with FastAPI reload:

```powershell
uv run uvicorn ai_server.server:app --host 0.0.0.0 --port 8001 --reload
```
