"""Trace inspection endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request

from polyportia.observability.store import TraceStore, get_default_store

router = APIRouter()


def _store(request: Request) -> TraceStore:
    return getattr(request.app.state, "trace_store", get_default_store())


@router.get("/v1/traces/{trace_id}")
async def get_trace(trace_id: str, request: Request) -> dict[str, Any]:
    rec = _store(request).get(trace_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"trace '{trace_id}' not found")
    return rec.to_dict()


@router.get("/v1/traces")
async def list_traces(request: Request, limit: int = 50) -> dict[str, Any]:
    return {
        "data": [
            {
                "trace_id": rec.trace_id,
                "created_at": rec.created_at.isoformat(),
                "final_status": rec.final_status,
                "request_summary": rec.request_summary,
            }
            for rec in _store(request).list(limit=limit)
        ]
    }
