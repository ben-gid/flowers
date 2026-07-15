import time
import logging

from fastapi import Request
from fastapi.responses import Response

logger = logging.getLogger("api_v2")


async def log_responses(request: Request, call_next):
    """Log method, path, status code, duration, and response body for every request."""
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000

    # Consume the stream so we can log it, then rebuild the response
    body = b"".join([chunk async for chunk in response.body_iterator])

    logger.info(
        "%s %s → %d (%.1fms) body=%s",
        request.method, request.url.path,
        response.status_code, duration_ms,
        body.decode("utf-8", errors="replace"),
    )

    return Response(
        content=body,
        status_code=response.status_code,
        headers=dict(response.headers),
        media_type=response.media_type,
    )
