"""Job orchestration and background task management for placement runs."""

import asyncio
import random
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from .constraints import PlacementError, validate_placement
from .engine import PlacementEngine
from .hpwl import calculate_hpwl
from .models import Netlist, PlacedMacro, Placement


@dataclass
class JobProgress:
    """State and progress snapshot of a placement job."""

    run_id: str
    status: str  # "pending", "running", "completed", "failed"
    progress: float  # 0.0 to 1.0
    step: int
    total_steps: int
    current_hpwl: float | None
    placement: list[PlacedMacro] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert snapshot to JSON-serializable dictionary."""
        return {
            "run_id": self.run_id,
            "status": self.status,
            "progress": self.progress,
            "step": self.step,
            "total_steps": self.total_steps,
            "current_hpwl": self.current_hpwl,
            "placement": [
                {"macro_id": pm.macro_id, "x": pm.x, "y": pm.y}
                for pm in self.placement
            ],
            "error": self.error,
        }


ProgressCallback = Callable[[JobProgress], Awaitable[None]]
AlgorithmRunner = Callable[
    [Netlist, int, dict[str, Any], ProgressCallback],
    Awaitable[Placement],
]

# Shared registry for placement algorithm implementations.
# Teammates Shravani (Random, SA, Greedy) and Ishita (RL) can register algorithms here.
ALGORITHM_REGISTRY: dict[str, AlgorithmRunner] = {}


async def default_random_placement_runner(
    netlist: Netlist,
    seed: int,
    params: dict[str, Any],
    progress_cb: ProgressCallback,
) -> Placement:
    """Deterministic random placement baseline runner.

    Places macros sequentially at randomly chosen legal coordinates.
    """
    rng = random.Random(seed)
    engine = PlacementEngine(netlist)
    total_steps = len(netlist.macros)

    for step, macro in enumerate(netlist.macros, start=1):
        legal_positions = engine.get_legal_positions(macro.id)
        if not legal_positions:
            raise PlacementError(
                f"No legal position available for macro {macro.id}."
            )

        chosen_x, chosen_y = rng.choice(legal_positions)
        engine.place(macro.id, chosen_x, chosen_y)

        current_hpwl = engine.current_hpwl(allow_incomplete=True)
        await progress_cb(
            JobProgress(
                run_id="",
                status="running",
                progress=float(step / total_steps),
                step=step,
                total_steps=total_steps,
                current_hpwl=current_hpwl,
                placement=engine.get_placement(),
            )
        )
        # Yield control to prevent blocking event loop
        await asyncio.sleep(0.001)

    return engine.get_placement()


def register_algorithm(name: str, runner: AlgorithmRunner) -> None:
    """Register an algorithm runner with the engine registry.

    Parameters:
        name: Unique lowercase name identifying the algorithm (e.g. 'greedy', 'sa', 'ppo').
        runner: An async function matching AlgorithmRunner signature.
    """
    clean_name = name.strip().lower()
    if not clean_name:
        raise ValueError("Algorithm name must be a non-empty string.")
    if not callable(runner):
        raise TypeError(f"Runner for '{clean_name}' must be callable.")
    ALGORITHM_REGISTRY[clean_name] = runner


def get_registered_algorithms() -> list[str]:
    """Return a sorted list of all registered algorithm names."""
    return sorted(ALGORITHM_REGISTRY.keys())


# Register default random algorithm runner
register_algorithm("random", default_random_placement_runner)


class JobManager:
    """Manages placement run lifecycles, execution tasks, and WebSocket subscribers."""

    def __init__(self) -> None:
        """Initialize in-memory job store and subscriber queues."""
        self.jobs: dict[str, JobProgress] = {}
        self.subscribers: dict[str, set[asyncio.Queue[dict[str, Any]]]] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}

    def get_job(self, run_id: str) -> JobProgress | None:
        """Retrieve job progress by ID."""
        return self.jobs.get(run_id)

    def list_jobs(self) -> list[JobProgress]:
        """List all tracked jobs."""
        return list(self.jobs.values())

    async def _notify_subscribers(self, run_id: str, snapshot: JobProgress) -> None:
        """Push progress update to all active WebSocket subscriber queues for a job."""
        data = snapshot.to_dict()
        queues = self.subscribers.get(run_id, set())
        for q in list(queues):
            try:
                q.put_nowait(data)
            except asyncio.QueueFull:
                pass

    def subscribe(self, run_id: str) -> asyncio.Queue[dict[str, Any]]:
        """Register a subscriber queue for a job's progress stream."""
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=100)
        self.subscribers.setdefault(run_id, set()).add(q)
        return q

    def unsubscribe(self, run_id: str, q: asyncio.Queue[dict[str, Any]]) -> None:
        """Unregister a subscriber queue."""
        if run_id in self.subscribers:
            self.subscribers[run_id].discard(q)
            if not self.subscribers[run_id]:
                del self.subscribers[run_id]

    async def _execute_job(
        self,
        run_id: str,
        netlist: Netlist,
        algorithm: str,
        seed: int,
        params: dict[str, Any],
    ) -> None:
        """Run the placement job asynchronously in background."""
        runner = ALGORITHM_REGISTRY.get(algorithm)
        if runner is None:
            self.jobs[run_id].status = "failed"
            self.jobs[run_id].error = (
                f"Unknown algorithm: '{algorithm}'. "
                f"Available: {list(ALGORITHM_REGISTRY.keys())}"
            )
            await self._notify_subscribers(run_id, self.jobs[run_id])
            return

        async def on_progress(p: JobProgress) -> None:
            p.run_id = run_id
            self.jobs[run_id] = p
            await self._notify_subscribers(run_id, p)

        try:
            self.jobs[run_id].status = "running"
            await self._notify_subscribers(run_id, self.jobs[run_id])

            final_placement = await runner(netlist, seed, params, on_progress)

            # Validate that the returned placement is legal and complete
            report = validate_placement(netlist, final_placement)
            if not report.is_valid or not report.is_complete:
                raise PlacementError(
                    f"Runner '{algorithm}' produced an invalid or incomplete placement. "
                    f"Errors: {list(report.errors)}. Placed: {report.placed_count}/{report.total_count}."
                )

            final_hpwl = calculate_hpwl(netlist, final_placement, allow_incomplete=False)

            completed_state = JobProgress(
                run_id=run_id,
                status="completed",
                progress=1.0,
                step=len(netlist.macros),
                total_steps=len(netlist.macros),
                current_hpwl=final_hpwl,
                placement=final_placement,
                error=None,
            )
            self.jobs[run_id] = completed_state
            await self._notify_subscribers(run_id, completed_state)
        except Exception as exc:  # noqa: BLE001
            # Top-level background task exception barrier to capture any solver failure
            failed_state = JobProgress(
                run_id=run_id,
                status="failed",
                progress=self.jobs[run_id].progress,
                step=self.jobs[run_id].step,
                total_steps=len(netlist.macros),
                current_hpwl=self.jobs[run_id].current_hpwl,
                placement=self.jobs[run_id].placement,
                error=str(exc),
            )
            self.jobs[run_id] = failed_state
            await self._notify_subscribers(run_id, failed_state)

    def create_and_start_job(
        self,
        netlist: Netlist,
        algorithm: str = "random",
        seed: int = 42,
        params: dict[str, Any] | None = None,
    ) -> str:
        """Create a job entry and launch its execution task."""
        run_id = str(uuid.uuid4())
        initial_progress = JobProgress(
            run_id=run_id,
            status="pending",
            progress=0.0,
            step=0,
            total_steps=len(netlist.macros),
            current_hpwl=None,
            placement=[],
        )
        self.jobs[run_id] = initial_progress

        task = asyncio.create_task(
            self._execute_job(
                run_id=run_id,
                netlist=netlist,
                algorithm=algorithm,
                seed=seed,
                params=params or {},
            )
        )
        self._tasks[run_id] = task
        return run_id
