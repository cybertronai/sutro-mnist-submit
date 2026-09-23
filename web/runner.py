"""Run one submission job through run_modal.evaluate() and write the payload.

Started by the A100 worker in its own session (process group) so that the
worker can kill everything the evaluation spawned when the job's deadline
passes. Deliberately stdlib-only: no modal, no fastapi.

    python runner.py job.json payload.json
"""

import json
import shutil
import sys
from pathlib import Path

HARNESS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HARNESS))

import run_modal  # noqa: E402


def lift_baked_pool(baked: Path) -> dict | None:
    """Read the image's baked dataset files into RAM and delete them from disk.

    Mirrors run_modal.remote_evaluate: the evaluator re-verifies every MD5 and
    consumes its copy before any submission process exists, so the public
    labels are never on the container's disk while a kernel runs. Returns None
    when there is no baked pool (a local dry run downloads instead).
    """
    if not baked.is_dir():
        return None
    pool = {path.name: path.read_bytes() for path in sorted(baked.glob("*.gz"))}
    shutil.rmtree(baked, ignore_errors=True)
    return pool or None


def sources_for(job: dict, band_dir: Path) -> tuple[dict, dict]:
    task = run_modal.load_task(band_dir)
    sources = {}
    for entry in task["files"]:
        if entry["source"] == "@SUBMISSION@":
            sources[entry["name"]] = job["source"]
        else:
            sources[entry["name"]] = (band_dir / entry["source"]).read_text()
    return task, sources


def main(job_path: str, out_path: str, work_root: str = "/tmp") -> int:
    job = json.loads(Path(job_path).read_text())
    task, sources = sources_for(job, HARNESS / job["band"])
    pool_bytes = lift_baked_pool(Path(run_modal.BAKED_POOL))
    payload = run_modal.evaluate(sources, task, job["mode"], {}, int(job["seed"]), {},
                                 work_root=work_root, pool_bytes=pool_bytes)
    payload["job_id"] = job["id"]
    Path(out_path).write_text(json.dumps(payload))
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
