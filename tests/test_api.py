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
    try:
        parsed = json.loads(response_body.decode("utf-8")) if response_body else {}
    except Exception:
        parsed = {"raw": response_body.decode("utf-8", errors="replace")}
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


@async_test
async def test_run_greedy_via_api(valid_netlist_payload: dict[str, Any]) -> None:
    """POST /api/placements/run executes Greedy algorithm to completion."""
    payload = {
        "netlist": valid_netlist_payload,
        "algorithm": "greedy",
        "seed": 42,
        "params": {"alpha": 1.0, "beta": 1.0},
    }
    status_code, start_body = await asgi_request("POST", "/api/placements/run", json_body=payload)
    assert status_code == 202
    run_id = start_body["run_id"]

    for _ in range(50):
        _, job = await asgi_request("GET", f"/api/placements/{run_id}")
        if job.get("status") in ("completed", "failed"):
            break
        await asyncio.sleep(0.02)

    assert job["status"] == "completed"
    assert job["current_hpwl"] is not None
    assert len(job["placement"]) == 3


@async_test
async def test_run_sa_via_api(valid_netlist_payload: dict[str, Any]) -> None:
    """POST /api/placements/run executes SA algorithm to completion."""
    payload = {
        "netlist": valid_netlist_payload,
        "algorithm": "sa",
        "seed": 42,
        "params": {"max_iterations": 10, "cooling_rate": 0.8},
    }
    status_code, start_body = await asgi_request("POST", "/api/placements/run", json_body=payload)
    assert status_code == 202
    run_id = start_body["run_id"]

    for _ in range(50):
        _, job = await asgi_request("GET", f"/api/placements/{run_id}")
        if job.get("status") in ("completed", "failed"):
            break
        await asyncio.sleep(0.02)

    assert job["status"] == "completed"
    assert job["current_hpwl"] is not None
    assert len(job["placement"]) == 3


@async_test
async def test_run_simulated_annealing_alias_via_api(valid_netlist_payload: dict[str, Any]) -> None:
    """POST /api/placements/run executes with 'simulated_annealing' alias."""
    payload = {
        "netlist": valid_netlist_payload,
        "algorithm": "simulated_annealing",
        "seed": 42,
        "params": {"max_iterations": 5},
    }
    status_code, start_body = await asgi_request("POST", "/api/placements/run", json_body=payload)
    assert status_code == 202
    run_id = start_body["run_id"]

    for _ in range(50):
        _, job = await asgi_request("GET", f"/api/placements/{run_id}")
        if job.get("status") in ("completed", "failed"):
            break
        await asyncio.sleep(0.02)

    assert job["status"] == "completed"


@async_test
async def test_list_algorithms() -> None:
    """GET /api/algorithms returns all registered algorithms."""
    status_code, body = await asgi_request("GET", "/api/algorithms")
    assert status_code == 200
    assert "algorithms" in body
    algs = set(body["algorithms"])
    assert {"random", "greedy", "sa", "simulated_annealing", "rl"}.issubset(algs)


@async_test
async def test_list_and_get_benchmarks() -> None:
    """GET /api/benchmarks and GET /api/benchmarks/{name} return valid benchmarks."""
    status_code, body = await asgi_request("GET", "/api/benchmarks")
    assert status_code == 200
    assert "benchmarks" in body
    names = {b["name"] for b in body["benchmarks"]}
    assert {"toy_4", "medium_8", "dense_10"}.issubset(names)

    status_code, toy_nl = await asgi_request("GET", "/api/benchmarks/toy_4")
    assert status_code == 200
    assert toy_nl["grid_w"] == 15
    assert len(toy_nl["macros"]) == 4

    status_code, _ = await asgi_request("GET", "/api/benchmarks/nonexistent")
    assert status_code == 404


@async_test
async def test_rudy_endpoints(valid_netlist_payload: dict[str, Any]) -> None:
    """Test POST /api/netlist/rudy and GET /api/placements/{run_id}/rudy."""
    placement = [
        {"macro_id": 0, "x": 0, "y": 0},
        {"macro_id": 1, "x": 5, "y": 5},
        {"macro_id": 2, "x": 10, "y": 10},
    ]
    status_code, body = await asgi_request(
        "POST",
        "/api/netlist/rudy",
        json_body={"netlist": valid_netlist_payload, "placement": placement},
    )
    assert status_code == 200
    assert body["grid_w"] == 20
    assert body["grid_h"] == 20
    assert len(body["demand"]) == 20
    assert len(body["demand"][0]) == 20
    assert "congestion_score" in body
    assert body["congestion_score"] >= 0.0


@async_test
async def test_completed_job_has_congestion_and_objective(valid_netlist_payload: dict[str, Any]) -> None:
    """Verify completed job status includes congestion and objective scores."""
    payload = {
        "netlist": valid_netlist_payload,
        "algorithm": "greedy",
        "seed": 42,
        "params": {"alpha": 1.0, "beta": 1.0},
    }
    status_code, start_body = await asgi_request("POST", "/api/placements/run", json_body=payload)
    assert status_code == 202
    run_id = start_body["run_id"]

    for _ in range(50):
        _, job = await asgi_request("GET", f"/api/placements/{run_id}")
        if job.get("status") in ("completed", "failed"):
            break
        await asyncio.sleep(0.02)

    assert job["status"] == "completed"
    assert job["current_hpwl"] is not None
    assert job["current_congestion"] is not None
    assert job["current_objective"] is not None
    assert job["congestion"] == job["current_congestion"]
    assert job["objective"] == job["current_objective"]

    # Also test GET /api/placements/{run_id}/rudy
    rudy_status, rudy_body = await asgi_request("GET", f"/api/placements/{run_id}/rudy")
    assert rudy_status == 200
    assert rudy_body["grid_w"] == 20
    assert rudy_body["congestion_score"] == job["current_congestion"]


# ---------------------------------------------------------------------------
# Phase E Tests: Experiments & Reports Endpoints
# ---------------------------------------------------------------------------


@async_test
async def test_experiments_index_endpoint() -> None:
    """GET /api/experiments returns summary for Experiments A, B, and D with archival metadata."""
    status_code, body = await asgi_request("GET", "/api/experiments")
    assert status_code == 200
    assert "experiments" in body
    assert "research_summary" in body

    exp_ids = {e["id"] for e in body["experiments"]}
    assert exp_ids == {"a", "b", "d"}

    exp_map = {e["id"]: e for e in body["experiments"]}
    assert exp_map["a"]["is_archival"] is False
    assert exp_map["b"]["is_archival"] is True
    assert exp_map["b"]["source_type"] == "verified_archival_reconstruction"
    assert exp_map["d"]["is_archival"] is True
    assert exp_map["d"]["source_type"] == "verified_archival_reconstruction"

    takeaways = body["research_summary"]["key_takeaways"]
    assert len(takeaways) >= 4
    assert any("Greedy" in t for t in takeaways)
    assert any("RL" in t for t in takeaways)


@async_test
async def test_experiment_a_endpoint() -> None:
    """GET /api/experiments/a returns canonical scaling data and figures."""
    status_code, body = await asgi_request("GET", "/api/experiments/a")
    assert status_code == 200
    assert body["experiment"] == "A"
    assert len(body["rows"]) == 3

    row_map = {r["benchmark"]: r for r in body["rows"]}
    assert "dense_10" in row_map
    assert "scaling_20" in row_map
    assert "scaling_30" in row_map

    # scaling_30 must be 48x48
    assert row_map["scaling_30"]["grid_w"] == 48
    assert row_map["scaling_30"]["grid_h"] == 48

    # Greedy has lowest objective on all
    for r in body["rows"]:
        assert r["best_algorithm"] == "Greedy"
        assert r["greedy_obj"] < r["random_obj"]

    # RL Improvement vs Random is positive (canonical improvement)
    assert row_map["dense_10"]["rl_improvement_pct"] == pytest.approx(28.95, abs=0.05)
    assert row_map["scaling_20"]["rl_improvement_pct"] == pytest.approx(8.98, abs=0.05)
    assert row_map["scaling_30"]["rl_improvement_pct"] == pytest.approx(11.40, abs=0.05)


@async_test
async def test_experiment_b_endpoint() -> None:
    """GET /api/experiments/b returns connectivity sensitivities and archival label."""
    status_code, body = await asgi_request("GET", "/api/experiments/b")
    assert status_code == 200
    assert body["experiment"] == "B"
    assert body["is_archival"] is True
    assert "Verified archival reconstruction" in body["provenance_label"]
    assert len(body["topologies"]) == 2
    assert len(body["sensitivities"]) == 4

    sens_map = {s["algorithm"]: s for s in body["sensitivities"]}
    assert sens_map["Simulated Annealing"]["pct_change"] == pytest.approx(-29.34, abs=0.05)
    assert sens_map["Greedy"]["pct_change"] == pytest.approx(-15.79, abs=0.05)
    assert sens_map["RL (Maskable PPO)"]["pct_change"] == pytest.approx(-15.15, abs=0.05)


@async_test
async def test_experiment_d_canonical_values() -> None:
    """GET /api/experiments/d returns canonical verified means, gaps, and rankings."""
    status_code, body = await asgi_request("GET", "/api/experiments/d")
    assert status_code == 200
    assert body["experiment"] == "D"
    assert body["is_archival"] is True
    assert "12 macros, 28×28" in body["controlled_setup"]

    cond_map = {c["condition"]: c for c in body["conditions"]}
    train = cond_map["training"]
    unseen_inst = cond_map["unseen_instance"]
    unseen_topo = cond_map["unseen_topology"]

    # Canonical means
    assert train["random_obj"] == pytest.approx(513.57, abs=0.05)
    assert train["greedy_obj"] == pytest.approx(225.48, abs=0.05)
    assert train["sa_obj"] == pytest.approx(305.04, abs=0.05)
    assert train["rl_obj"] == pytest.approx(335.42, abs=0.05)
    assert train["ranking"] == "Greedy < SA < RL < Random"

    assert unseen_inst["random_obj"] == pytest.approx(535.44, abs=0.05)
    assert unseen_inst["greedy_obj"] == pytest.approx(305.76, abs=0.05)
    assert unseen_inst["sa_obj"] == pytest.approx(348.53, abs=0.05)
    assert unseen_inst["rl_obj"] == pytest.approx(441.64, abs=0.05)
    assert unseen_inst["ranking"] == "Greedy < SA < RL < Random"

    assert unseen_topo["random_obj"] == pytest.approx(568.86, abs=0.05)
    assert unseen_topo["greedy_obj"] == pytest.approx(282.70, abs=0.05)
    assert unseen_topo["sa_obj"] == pytest.approx(280.87, abs=0.05)
    assert unseen_topo["rl_obj"] == pytest.approx(460.45, abs=0.05)
    # Correct ranking: SA < Greedy < RL < Random
    assert unseen_topo["ranking"] == "SA < Greedy < RL < Random"
    assert unseen_topo["best_algorithm"] == "Simulated Annealing"

    # Generalization gaps
    gap_map = {g["algorithm"]: g for g in body["gaps"]}
    rl_gap = gap_map["RL (Maskable PPO)"]
    assert rl_gap["inst_delta"] == pytest.approx(106.22, abs=0.1)
    assert rl_gap["inst_gap_pct"] == pytest.approx(31.67, abs=0.1)
    assert rl_gap["topo_delta"] == pytest.approx(125.04, abs=0.1)
    assert rl_gap["topo_gap_pct"] == pytest.approx(37.28, abs=0.1)


@async_test
async def test_reports_endpoints() -> None:
    """GET /api/reports and download endpoints serve report artifacts and handle missing files."""
    status_code, body = await asgi_request("GET", "/api/reports")
    assert status_code == 200
    assert body["html_available"] is True
    assert body["html_url"] is not None
    assert body["pdf_status"] in ("Available", "Not generated")
    assert body["figures_count"] >= 4

    # Download existing HTML report
    dl_status, _ = await asgi_request("GET", "/api/reports/download/macro_placement_report.html")
    assert dl_status == 200

    # Non-existent report returns 404
    missing_status, _ = await asgi_request("GET", "/api/reports/download/does_not_exist.html")
    assert missing_status == 404
