"""Integration tests for FastAPI endpoints, request validation, and WebSockets."""

import asyncio
import functools
import json
from collections.abc import Callable, MutableMapping
from typing import Any

import pytest

from src.placement.api import app


def async_test(func: Callable[..., Any]) -> Callable[..., Any]:
    """Helper decorator running coroutine tests via asyncio.run without extra plugins."""
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return asyncio.run(func(*args, **kwargs))
    return wrapper


async def asgi_request(
    method: str,
    path: str,
    json_body: dict[str, Any] | None = None,
) -> tuple[int, dict[str, Any]]:
    """Execute an HTTP request directly against the ASGI app without external network."""
    body_bytes = json.dumps(json_body).encode("utf-8") if json_body is not None else b""
    headers = [
        (b"host", b"testserver"),
        (b"content-type", b"application/json"),
        (b"content-length", str(len(body_bytes)).encode("ascii")),
    ]

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": b"",
        "headers": headers,
    }

    status_code = 500
    response_body = bytearray()

    async def receive() -> dict[str, Any]:
        return {
            "type": "http.request",
            "body": body_bytes,
            "more_body": False,
        }

    async def send(message: MutableMapping[str, Any]) -> None:
        nonlocal status_code
        if message["type"] == "http.response.start":
            status_code = message["status"]
        elif message["type"] == "http.response.body":
            response_body.extend(message.get("body", b""))

    await app(scope, receive, send)
    parsed = json.loads(response_body.decode("utf-8")) if response_body else {}
    return status_code, parsed


@pytest.fixture
def valid_netlist_payload() -> dict[str, Any]:
    """Return a valid netlist dictionary payload matching NetlistSchema."""
    return {
        "version": "1.0",
        "grid_w": 20,
        "grid_h": 20,
        "macros": [
            {"id": 0, "w": 4, "h": 4},
            {"id": 1, "w": 3, "h": 5},
            {"id": 2, "w": 2, "h": 2},
        ],
        "nets": [
            {"id": 0, "macro_ids": [0, 1], "weight": 1.0},
            {"id": 1, "macro_ids": [1, 2], "weight": 2.5},
        ],
    }


@async_test
async def test_health_check() -> None:
    """GET /health returns 200 and ok status."""
    status_code, body = await asgi_request("GET", "/health")
    assert status_code == 200
    assert body["status"] == "ok"
    assert "version" in body


@async_test
async def test_validate_netlist_success(valid_netlist_payload: dict[str, Any]) -> None:
    """POST /api/netlist/validate reports valid netlist."""
    status_code, body = await asgi_request(
        "POST",
        "/api/netlist/validate",
        json_body=valid_netlist_payload,
    )
    assert status_code == 200
    assert body["is_valid"] is True
    assert body["macro_count"] == 3
    assert body["net_count"] == 2
    assert len(body["errors"]) == 0


@async_test
async def test_validate_netlist_errors() -> None:
    """POST /api/netlist/validate reports macro size and net connectivity errors."""
    invalid_payload = {
        "version": "1.0",
        "grid_w": 10,
        "grid_h": 10,
        "macros": [
            {"id": 0, "w": 12, "h": 5},  # Exceeds grid width
            {"id": 1, "w": 2, "h": 2},
        ],
        "nets": [
            {"id": 0, "macro_ids": [0]},        # Fewer than 2 macros
            {"id": 1, "macro_ids": [1, 99]},     # Non-existent macro 99
        ],
    }
    status_code, body = await asgi_request(
        "POST",
        "/api/netlist/validate",
        json_body=invalid_payload,
    )
    assert status_code == 200
    assert body["is_valid"] is False
    assert len(body["errors"]) >= 3


@async_test
async def test_start_placement_run(valid_netlist_payload: dict[str, Any]) -> None:
    """POST /api/placements/run validates, starts job, and returns 202."""
    payload = {
        "netlist": valid_netlist_payload,
        "algorithm": "random",
        "seed": 42,
    }
    status_code, body = await asgi_request(
        "POST",
        "/api/placements/run",
        json_body=payload,
    )
    assert status_code == 202
    assert "run_id" in body
    assert body["status"] == "pending"

    run_id = body["run_id"]

    # Poll status
    status_body: dict[str, Any] = {}
    for _ in range(50):
        s_code, status_body = await asgi_request("GET", f"/api/placements/{run_id}")
        assert s_code == 200
        if status_body["status"] in ("completed", "failed"):
            break
        await asyncio.sleep(0.02)

    assert status_body["status"] == "completed"
    assert status_body["progress"] == 1.0
    assert len(status_body["placement"]) == 3
    assert status_body["current_hpwl"] is not None


@async_test
async def test_start_placement_run_invalid_inputs(
    valid_netlist_payload: dict[str, Any],
) -> None:
    """Invalid netlist or unregistered algorithm returns HTTP 400."""
    # Unregistered algorithm
    bad_algo_payload = {
        "netlist": valid_netlist_payload,
        "algorithm": "unknown_future_algo",
        "seed": 1,
    }
    status_code, _ = await asgi_request(
        "POST",
        "/api/placements/run",
        json_body=bad_algo_payload,
    )
    assert status_code == 400

    # Invalid netlist
    bad_netlist_payload = dict(valid_netlist_payload)
    bad_netlist_payload["macros"] = [{"id": 0, "w": 999, "h": 999}]
    status_code_net, _ = await asgi_request(
        "POST",
        "/api/placements/run",
        json_body={"netlist": bad_netlist_payload, "algorithm": "random"},
    )
    assert status_code_net == 400


@async_test
async def test_get_unknown_job_404() -> None:
    """GET /api/placements/{run_id} returns 404 for unknown job."""
    status_code, body = await asgi_request("GET", "/api/placements/non-existent-id")
    assert status_code == 404
    assert "detail" in body


@async_test
async def test_list_placements(valid_netlist_payload: dict[str, Any]) -> None:
    """GET /api/placements returns list of active and completed jobs."""
    status_code, runs = await asgi_request("GET", "/api/placements")
    assert status_code == 200
    assert isinstance(runs, list)


@async_test
async def test_websocket_progress_stream(valid_netlist_payload: dict[str, Any]) -> None:
    """WebSocket /api/placements/{run_id}/ws receives real-time progress updates."""
    # Launch job
    payload = {
        "netlist": valid_netlist_payload,
        "algorithm": "random",
        "seed": 55,
    }
    _, start_body = await asgi_request("POST", "/api/placements/run", json_body=payload)
    run_id = start_body["run_id"]

    in_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    out_queue: asyncio.Queue[MutableMapping[str, Any]] = asyncio.Queue()
    await in_queue.put({"type": "websocket.connect"})

    scope = {
        "type": "websocket",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "scheme": "ws",
        "path": f"/api/placements/{run_id}/ws",
        "raw_path": f"/api/placements/{run_id}/ws".encode("ascii"),
        "query_string": b"",
        "headers": [],
        "subprotocols": [],
    }

    async def receive() -> dict[str, Any]:
        return await in_queue.get()

    async def send(msg: MutableMapping[str, Any]) -> None:
        await out_queue.put(msg)

    ws_task = asyncio.create_task(app(scope, receive, send))

    # Expect accept
    accept_msg = await asyncio.wait_for(out_queue.get(), timeout=1.0)
    assert accept_msg["type"] == "websocket.accept"

    messages = []
    # Collect streamed progress messages until terminal status
    for _ in range(50):
        try:
            msg = await asyncio.wait_for(out_queue.get(), timeout=0.2)
            if msg["type"] == "websocket.send":
                data = json.loads(msg["text"])
                messages.append(data)
                if data.get("status") in ("completed", "failed"):
                    break
        except TimeoutError:
            break

    await in_queue.put({"type": "websocket.disconnect", "code": 1000})
    await ws_task

    assert len(messages) >= 1
    assert any(m["status"] in ("running", "completed") for m in messages)
