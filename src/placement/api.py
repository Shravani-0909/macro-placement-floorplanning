"""FastAPI application providing REST endpoints and WebSocket streams for macro placement."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .benchmarks import get_default_benchmark_suite
from .jobs import ALGORITHM_REGISTRY, JobManager, get_registered_algorithms
from .models import Macro, Net, Netlist, PlacedMacro
from .objective import congestion_score, rudy_map


class MacroSchema(BaseModel):
    """Schema for a macro definition."""

    id: int = Field(..., ge=0, description="Unique macro ID")
    w: int = Field(..., gt=0, description="Macro width in grid cells")
    h: int = Field(..., gt=0, description="Macro height in grid cells")


class NetSchema(BaseModel):
    """Schema for a net hyperedge connecting macros."""

    id: int = Field(..., ge=0, description="Unique net ID")
    macro_ids: list[int] = Field(..., description="List of connected macro IDs")
    weight: float = Field(default=1.0, gt=0.0, description="Net weight")


class NetlistSchema(BaseModel):
    """Schema for a complete circuit netlist."""

    version: str = Field(default="1.0", description="Netlist format version")
    grid_w: int = Field(..., gt=0, description="Placement grid width")
    grid_h: int = Field(..., gt=0, description="Placement grid height")
    macros: list[MacroSchema]
    nets: list[NetSchema]

    def to_domain_model(self) -> Netlist:
        """Convert Pydantic schema to immutable Netlist dataclass."""
        return Netlist(
            version=self.version,
            grid_w=self.grid_w,
            grid_h=self.grid_h,
            macros=tuple(Macro(id=m.id, w=m.w, h=m.h) for m in self.macros),
            nets=tuple(
                Net(id=n.id, macro_ids=tuple(n.macro_ids), weight=n.weight)
                for n in self.nets
            ),
        )


class PlacedMacroSchema(BaseModel):
    """Schema for a placed macro's coordinates."""

    macro_id: int
    x: int
    y: int


class PlacementRunRequest(BaseModel):
    """Payload to initiate an asynchronous placement run."""

    netlist: NetlistSchema
    algorithm: str = Field(default="random", description="Placement algorithm name")
    seed: int = Field(default=42, description="Random seed for deterministic runs")
    params: dict[str, Any] = Field(
        default_factory=dict, description="Optional algorithm parameters"
    )


class PlacementRunResponse(BaseModel):
    """Initial response acknowledging placement run creation."""

    run_id: str
    status: str
    algorithm: str
    seed: int


class JobStatusResponse(BaseModel):
    """Status and results response for a placement run."""

    run_id: str
    status: str
    progress: float
    step: int
    total_steps: int
    current_hpwl: float | None
    current_congestion: float | None = None
    current_objective: float | None = None
    hpwl: float | None = None
    congestion: float | None = None
    objective: float | None = None
    placement: list[PlacedMacroSchema]
    error: str | None = None


class RudyRequest(BaseModel):
    """Payload to compute RUDY congestion map for a placement."""

    netlist: NetlistSchema
    placement: list[PlacedMacroSchema]


class RudyMapResponse(BaseModel):
    """RUDY routing demand density grid and congestion metrics."""

    grid_w: int
    grid_h: int
    demand: list[list[float]]
    congestion_score: float
    max_demand: float


# Application-level job manager instance
job_manager = JobManager()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """FastAPI application lifespan management."""
    yield


app = FastAPI(
    title="Macro Placement Engine API",
    description="Backend API for graph-theoretic macro placement and floorplanning",
    version="0.1.0",
    lifespan=lifespan,
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["Health"])
async def health_check() -> dict[str, str]:
    """Health check endpoint confirming API service readiness."""
    return {"status": "ok", "version": "0.1.0"}


@app.post("/api/netlist/validate", tags=["Netlist"])
async def validate_netlist_endpoint(schema: NetlistSchema) -> dict[str, Any]:
    """Validate a netlist specification and verify macro and net integrity."""
    errors: list[str] = []
    macro_ids = {m.id for m in schema.macros}

    # Verify unique macro IDs
    if len(macro_ids) != len(schema.macros):
        errors.append("Duplicate macro IDs found in netlist.")

    # Verify macros fit on grid
    for m in schema.macros:
        if m.w > schema.grid_w or m.h > schema.grid_h:
            errors.append(
                f"Macro {m.id} ({m.w}x{m.h}) exceeds grid dimensions ({schema.grid_w}x{schema.grid_h})."
            )

    # Verify nets connect existing macros
    for net in schema.nets:
        if len(net.macro_ids) < 2:
            errors.append(f"Net {net.id} must connect at least 2 macros.")
        for mid in net.macro_ids:
            if mid not in macro_ids:
                errors.append(f"Net {net.id} references undefined macro ID {mid}.")

    return {
        "is_valid": len(errors) == 0,
        "grid_w": schema.grid_w,
        "grid_h": schema.grid_h,
        "macro_count": len(schema.macros),
        "net_count": len(schema.nets),
        "errors": errors,
    }


@app.post(
    "/api/placements/run",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=PlacementRunResponse,
    tags=["Placements"],
)
async def start_placement_run(
    request: PlacementRunRequest,
) -> PlacementRunResponse:
    """Validate netlist and dispatch placement run as a background task."""
    validation_res = await validate_netlist_endpoint(request.netlist)
    if not validation_res["is_valid"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "message": "Invalid netlist specification.",
                "errors": validation_res["errors"],
            },
        )

    if request.algorithm not in ALGORITHM_REGISTRY:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Algorithm '{request.algorithm}' is not registered. "
                f"Available: {list(ALGORITHM_REGISTRY.keys())}"
            ),
        )

    netlist = request.netlist.to_domain_model()
    run_id = job_manager.create_and_start_job(
        netlist=netlist,
        algorithm=request.algorithm,
        seed=request.seed,
        params=request.params,
    )

    return PlacementRunResponse(
        run_id=run_id,
        status="pending",
        algorithm=request.algorithm,
        seed=request.seed,
    )


@app.get(
    "/api/placements/{run_id}",
    response_model=JobStatusResponse,
    tags=["Placements"],
)
async def get_placement_status(run_id: str) -> JobStatusResponse:
    """Retrieve the current progress and results of a placement run."""
    job = job_manager.get_job(run_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Placement run '{run_id}' not found.",
        )
    return JobStatusResponse(**job.to_dict())


@app.get(
    "/api/placements",
    response_model=list[JobStatusResponse],
    tags=["Placements"],
)
async def list_placements() -> list[JobStatusResponse]:
    """List all tracked placement runs and their current statuses."""
    jobs = job_manager.list_jobs()
    return [JobStatusResponse(**j.to_dict()) for j in jobs]


@app.websocket("/api/placements/{run_id}/ws")
async def placement_progress_websocket(websocket: WebSocket, run_id: str) -> None:
    """Stream live placement progress updates for run_id via WebSocket."""
    await websocket.accept()

    job = job_manager.get_job(run_id)
    if job is None:
        await websocket.send_json(
            {"error": f"Placement run '{run_id}' not found."}
        )
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    # Send current state immediately
    await websocket.send_json(job.to_dict())

    if job.status in ("completed", "failed"):
        await websocket.close()
        return

    queue = job_manager.subscribe(run_id)
    try:
        while True:
            data = await queue.get()
            await websocket.send_json(data)
            if data.get("status") in ("completed", "failed"):
                break
    except WebSocketDisconnect:
        pass
    finally:
        job_manager.unsubscribe(run_id, queue)
        try:
            await websocket.close()
        except RuntimeError:
            pass


@app.get("/api/algorithms", tags=["Algorithms"])
async def list_algorithms_endpoint() -> dict[str, list[str]]:
    """List all registered placement algorithms."""
    return {"algorithms": get_registered_algorithms()}


@app.get("/api/benchmarks", tags=["Benchmarks"])
async def list_benchmarks_endpoint() -> dict[str, Any]:
    """List standard reproducible benchmark suites."""
    suite = get_default_benchmark_suite()
    summaries = []
    for name, nl in suite.items():
        area = sum(m.w * m.h for m in nl.macros)
        g_area = nl.grid_w * nl.grid_h
        summaries.append({
            "name": name,
            "grid_w": nl.grid_w,
            "grid_h": nl.grid_h,
            "macro_count": len(nl.macros),
            "net_count": len(nl.nets),
            "total_macro_area": area,
            "grid_area": g_area,
            "area_utilization": round(area / g_area, 4) if g_area > 0 else 0.0,
        })
    return {"benchmarks": summaries}


@app.get("/api/benchmarks/{name}", response_model=NetlistSchema, tags=["Benchmarks"])
async def get_benchmark_endpoint(name: str) -> NetlistSchema:
    """Retrieve full Netlist specification for a named standard benchmark."""
    suite = get_default_benchmark_suite()
    if name not in suite:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Benchmark '{name}' not found. Available: {list(suite.keys())}",
        )
    nl = suite[name]
    return NetlistSchema(
        version=nl.version,
        grid_w=nl.grid_w,
        grid_h=nl.grid_h,
        macros=[MacroSchema(id=m.id, w=m.w, h=m.h) for m in nl.macros],
        nets=[
            NetSchema(id=n.id, macro_ids=list(n.macro_ids), weight=n.weight)
            for n in nl.nets
        ],
    )


@app.get(
    "/api/placements/{run_id}/rudy",
    response_model=RudyMapResponse,
    tags=["Placements"],
)
async def get_placement_rudy_endpoint(run_id: str) -> RudyMapResponse:
    """Compute and retrieve the 2D RUDY wire density map for a placement run."""
    job = job_manager.get_job(run_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Placement run '{run_id}' not found.",
        )
    if not job.placement:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Placement run '{run_id}' has no placed macros.",
        )
    netlist = job_manager.get_netlist(run_id)
    if netlist is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Netlist for run '{run_id}' not found.",
        )
    demand = rudy_map(netlist, job.placement)
    c_score = congestion_score(netlist, job.placement)
    return RudyMapResponse(
        grid_w=netlist.grid_w,
        grid_h=netlist.grid_h,
        demand=demand.tolist(),
        congestion_score=c_score,
        max_demand=float(np.max(demand)) if demand.size > 0 else 0.0,
    )


@app.post(
    "/api/netlist/rudy",
    response_model=RudyMapResponse,
    tags=["Netlist"],
)
async def compute_rudy_endpoint(req: RudyRequest) -> RudyMapResponse:
    """Compute 2D RUDY wire density map for arbitrary netlist and placement."""
    netlist = req.netlist.to_domain_model()
    placements = [
        PlacedMacro(macro_id=p.macro_id, x=p.x, y=p.y)
        for p in req.placement
    ]
    demand = rudy_map(netlist, placements)
    c_score = congestion_score(netlist, placements)
    return RudyMapResponse(
        grid_w=netlist.grid_w,
        grid_h=netlist.grid_h,
        demand=demand.tolist(),
        congestion_score=c_score,
        max_demand=float(np.max(demand)) if demand.size > 0 else 0.0,
    )
