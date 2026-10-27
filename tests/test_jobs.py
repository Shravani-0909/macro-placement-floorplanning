"""Tests for job manager, background tasks, and placement orchestration."""

import asyncio
import functools
from collections.abc import Callable
from typing import Any

import pytest

from src.placement.constraints import validate_placement
from src.placement.jobs import JobManager, default_random_placement_runner
from src.placement.models import Macro, Net, Netlist


def async_test(func: Callable[..., Any]) -> Callable[..., Any]:
    """Helper decorator running coroutine tests via asyncio.run without extra plugins."""
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return asyncio.run(func(*args, **kwargs))
    return wrapper


@pytest.fixture
def small_netlist() -> Netlist:
    """Return a small feasible netlist."""
    return Netlist(
        version="1.0",
        grid_w=15,
        grid_h=15,
        macros=(
            Macro(id=0, w=3, h=3),
            Macro(id=1, w=4, h=2),
            Macro(id=2, w=2, h=4),
        ),
        nets=(
            Net(id=0, macro_ids=(0, 1), weight=1.0),
            Net(id=1, macro_ids=(1, 2), weight=2.0),
        ),
    )


@async_test
async def test_random_placement_runner(small_netlist: Netlist) -> None:
    """Deterministic random runner produces complete and valid placement."""
    progress_snapshots = []

    async def on_progress(p: Any) -> None:
        progress_snapshots.append(p)

    placement = await default_random_placement_runner(
        netlist=small_netlist,
        seed=123,
        params={},
        progress_cb=on_progress,
    )

    assert len(placement) == 3
    assert len(progress_snapshots) == 3
    assert progress_snapshots[-1].step == 3

    # Placements must be legal according to constraints
    report = validate_placement(small_netlist, placement)
    assert report.is_valid
    assert report.is_complete


@async_test
async def test_job_manager_lifecycle(small_netlist: Netlist) -> None:
    """JobManager starts background job and progresses to completion."""
    manager = JobManager()
    run_id = manager.create_and_start_job(
        netlist=small_netlist,
        algorithm="random",
        seed=42,
    )

    # Initial state is tracked
    initial_job = manager.get_job(run_id)
    assert initial_job is not None
    assert initial_job.run_id == run_id

    # Wait for completion
    for _ in range(50):
        job = manager.get_job(run_id)
        if job and job.status in ("completed", "failed"):
            break
        await asyncio.sleep(0.02)

    final_job = manager.get_job(run_id)
    assert final_job is not None
    assert final_job.status == "completed"
    assert final_job.progress == 1.0
    assert final_job.current_hpwl is not None
    assert final_job.current_hpwl > 0.0
    assert len(final_job.placement) == 3


@async_test
async def test_job_manager_unknown_algorithm(small_netlist: Netlist) -> None:
    """Requesting an unregistered algorithm marks the job as failed."""
    manager = JobManager()
    run_id = manager.create_and_start_job(
        netlist=small_netlist,
        algorithm="non_existent_algo",
        seed=1,
    )

    for _ in range(50):
        job = manager.get_job(run_id)
        if job and job.status == "failed":
            break
        await asyncio.sleep(0.02)

    job = manager.get_job(run_id)
    assert job is not None
    assert job.status == "failed"
    assert "Unknown algorithm" in (job.error or "")


@async_test
async def test_job_manager_subscribers(small_netlist: Netlist) -> None:
    """Subscribers receive streamed progress notifications."""
    manager = JobManager()
    run_id = manager.create_and_start_job(
        netlist=small_netlist,
        algorithm="random",
        seed=99,
    )
    queue = manager.subscribe(run_id)

    events = []
    for _ in range(50):
        try:
            event = await asyncio.wait_for(queue.get(), timeout=0.1)
            events.append(event)
            if event.get("status") in ("completed", "failed"):
                break
        except TimeoutError:
            cur = manager.get_job(run_id)
            if cur and cur.status in ("completed", "failed"):
                break

    manager.unsubscribe(run_id, queue)
    assert len(events) >= 1
    assert any(e["status"] in ("running", "completed") for e in events)


@async_test
async def test_job_manager_impossible_placement() -> None:
    """If placement cannot be completed legally, the job fails gracefully."""
    # 4x4 grid with two 3x3 macros - second macro cannot fit
    netlist = Netlist(
        version="1.0",
        grid_w=4,
        grid_h=4,
        macros=(
            Macro(id=0, w=3, h=3),
            Macro(id=1, w=3, h=3),
        ),
        nets=(Net(id=0, macro_ids=(0, 1)),),
    )
    manager = JobManager()
    run_id = manager.create_and_start_job(netlist=netlist, algorithm="random", seed=42)

    for _ in range(50):
        job = manager.get_job(run_id)
        if job and job.status in ("completed", "failed"):
            break
        await asyncio.sleep(0.02)

    job = manager.get_job(run_id)
    assert job is not None
    assert job.status == "failed"
    assert "No legal position available" in (job.error or "")


def test_register_algorithm_validation() -> None:
    """register_algorithm rejects blank names or non-callable runners."""
    from src.placement.jobs import get_registered_algorithms, register_algorithm

    async def dummy(n: Any, s: int, p: dict[str, Any], cb: Any) -> list[Any]:
        return []

    with pytest.raises(ValueError, match="non-empty string"):
        register_algorithm("", dummy)

    with pytest.raises(TypeError, match="must be callable"):
        register_algorithm("invalid", "not-a-function")  # type: ignore[arg-type]

    assert "random" in get_registered_algorithms()


@async_test
async def test_runner_returning_invalid_placement_fails_job(small_netlist: Netlist) -> None:
    """A runner returning overlapping macros is trapped and marked failed."""
    from src.placement.jobs import register_algorithm
    from src.placement.models import PlacedMacro

    # Dummy runner that illegally overlaps all macros at (0, 0)
    async def buggy_runner(netlist, seed, params, progress_cb):
        return [PlacedMacro(macro_id=m.id, x=0, y=0) for m in netlist.macros]

    register_algorithm("buggy_test", buggy_runner)

    manager = JobManager()
    run_id = manager.create_and_start_job(netlist=small_netlist, algorithm="buggy_test")

    for _ in range(50):
        job = manager.get_job(run_id)
        if job and job.status in ("completed", "failed"):
            break
        await asyncio.sleep(0.02)

    job = manager.get_job(run_id)
    assert job is not None
    assert job.status == "failed"
    assert "produced an invalid or incomplete placement" in (job.error or "")


