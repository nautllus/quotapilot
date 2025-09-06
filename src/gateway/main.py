import json
import logging
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi import Request as FastAPIRequest
from fastapi.responses import JSONResponse, StreamingResponse

from gateway.schemas import ChatRequest, ChatResponse
from router import ProviderRegistry, Router, NoCapableProviderError
from state.mongo import init_mongo, close_mongo, get_db
from state.budget import BudgetManager

logger = logging.getLogger(__name__)

app = FastAPI(title="QuotaPilot Gateway", version="0.1.0")


@app.on_event("startup")
async def startup_event() -> None:
    # Initialize Mongo and indexes
    await init_mongo()

    # Initialize registry and router once
    registry = ProviderRegistry()
    budget = BudgetManager(db=get_db())
    router = Router(registry, budget=budget)

    app.state.registry = registry
    app.state.router = router
    app.state.budget = budget

    logger.info("Gateway initialized with %d providers", len(registry.get_providers()))


@app.get("/health", tags=["health"])  # minimal health endpoint
async def health():
    return {"status": "ok"}


@app.get("/v1/router/state")
async def router_state():
    registry: ProviderRegistry = app.state.registry
    budget: BudgetManager = app.state.budget

    result = {}
    for provider in registry.get_providers():
        try:
            models = await provider.models()
        except Exception as e:  # pragma: no cover
            logger.warning("provider.models() failed for %s: %s", provider.name, e)
            models = []

        pstate = {"health": None, "models": {}}
        try:
            pst = await provider.state()
            pstate["health"] = pst
        except Exception as e:  # pragma: no cover
            pstate["health"] = {"status": "unknown", "error": str(e)}

        for m in models:
            mname = m.get("name")
            if not mname:
                continue
            stats = await budget.get_usage_stats(provider.name, mname)
            # Headroom without estimated tokens
            head = await budget.check_headroom(provider.name, mname)
            pstate["models"][mname] = {"usage": stats, "headroom": head.remaining}
        result[provider.name] = pstate

    return JSONResponse(result)


@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completions(req: ChatRequest, request: FastAPIRequest):
    router: Router = app.state.router

    if not isinstance(req, ChatRequest):
        raise HTTPException(status_code=400, detail="Invalid request body")

    try:
        if req.stream:
            # Basic streaming: get full response and stream as a single SSE event + DONE
            # Force non-streaming downstream to simplify
            non_stream_req = req.copy()
            non_stream_req.stream = False
            resp = await router.route_request(non_stream_req)
            return _sse_response(resp)
        else:
            resp = await router.route_request(req)
            return JSONResponse(status_code=200, content=json.loads(resp.model_dump_json()))
    except NoCapableProviderError as e:
        logger.warning("No capable provider: %s", e)
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        # Map known provider HTTP errors if they expose status_code
        status = getattr(e, "status_code", 500)
        detail = getattr(e, "message", None) or str(e)
        logger.exception("Chat completions failed: %s", e)
        raise HTTPException(status_code=int(status), detail=detail)


@app.on_event("shutdown")
async def shutdown_event() -> None:
    try:
        await close_mongo()
    except Exception:  # pragma: no cover
        pass


def _sse_response(chat: ChatResponse) -> StreamingResponse:
    async def gen() -> AsyncIterator[bytes]:
        payload = json.loads(chat.model_dump_json())
        yield f"data: {json.dumps(payload)}\n\n".encode("utf-8")
        yield b"data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")

