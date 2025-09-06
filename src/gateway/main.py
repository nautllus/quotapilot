from fastapi import FastAPI

app = FastAPI(title="QuotaPilot Gateway", version="0.1.0")


@app.get("/health", tags=["health"])  # minimal health endpoint
async def health():
    return {"status": "ok"}

