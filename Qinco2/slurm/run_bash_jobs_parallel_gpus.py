#!/usr/bin/env python3
"""
Run a flat bash script produced by generate_slurm_cmds.py (alternating ``cd ...``
and ``python ...`` lines) with many concurrent subprocesses.

Jobs are assigned ``CUDA_VISIBLE_DEVICES`` in round-robin order across ``--gpu-ids``
(so multiple jobs can share one GPU — useful when each task is CPU-bound and VRAM is tiny).

Examples:
  python slurm/run_bash_jobs_parallel_gpus.py jobs.sh --workers 16 --gpu-ids 0,1
  PARALLEL_WORKERS=12 python slurm/run_bash_jobs_parallel_gpus.py jobs.sh

Legacy (one worker per GPU, no oversubscription):
  python slurm/run_bash_jobs_parallel_gpus.py jobs.sh --gpus 2
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed


def parse_cd_python_pairs(path: str) -> list[tuple[str, str]]:
    lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
    jobs: list[tuple[str, str]] = []
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if s.startswith("cd ") and i + 1 < len(lines):
            nxt = lines[i + 1].strip()
            if nxt.startswith("python "):
                jobs.append((s, nxt))
                i += 2
                continue
        i += 1
    return jobs


def resolve_gpu_ids(arg: str | None) -> list[int]:
    if arg:
        return [int(x.strip()) for x in arg.split(",") if x.strip()]
    try:
        import torch

        n = int(torch.cuda.device_count()) if torch.cuda.is_available() else 1
    except Exception:
        n = 1
    return list(range(max(1, n)))


def default_workers(gpu_ids: list[int]) -> int:
    env = os.environ.get("PARALLEL_WORKERS")
    if env is not None and env.strip():
        return max(1, int(env.strip()))
    # Aggressive default: AQ training is mostly CPU; GPUs stay idle otherwise.
    return max(8, len(gpu_ids) * 8)


def run_one(gpu_id: int, cd_line: str, py_line: str, env_base: dict[str, str]) -> tuple[int, int]:
    env = dict(env_base)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    cmd = f"{cd_line} && {py_line}"
    r = subprocess.run(cmd, shell=True, executable="/bin/bash", env=env)
    return gpu_id, r.returncode


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("bash_script", help="Path to bash file with cd/python pairs")
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Max concurrent subprocesses (default: env PARALLEL_WORKERS or 8× num GPUs)",
    )
    p.add_argument(
        "--gpu-ids",
        type=str,
        default=None,
        help="Comma-separated physical GPU ids for round-robin (default: 0..N-1 from torch)",
    )
    p.add_argument(
        "--gpus",
        type=int,
        default=None,
        help="Deprecated: use --workers N with --gpu-ids 0,1,... Same as workers=N on GPUs 0..N-1.",
    )
    p.add_argument("--dry-run", action="store_true", help="Print job count and exit")
    args = p.parse_args()

    jobs = parse_cd_python_pairs(args.bash_script)
    if args.dry_run:
        gids = resolve_gpu_ids(args.gpu_ids)
        w = args.workers if args.workers is not None else (
            max(1, args.gpus) if args.gpus is not None else default_workers(gids)
        )
        print(f"Parsed {len(jobs)} jobs from {args.bash_script} (would use workers={w}, gpu_ids={gids})")
        return 0
    if not jobs:
        print(f"No cd/python job pairs found in {args.bash_script}", file=sys.stderr)
        return 2

    if args.gpus is not None and args.workers is None:
        gpu_ids = list(range(max(1, args.gpus)))
        workers = len(gpu_ids)
    else:
        gpu_ids = resolve_gpu_ids(args.gpu_ids)
        workers = args.workers if args.workers is not None else default_workers(gpu_ids)

    workers = max(1, workers)
    env_base = dict(os.environ)

    if workers == 1:
        print("Running serially (workers=1).")
        for i, (cd_l, py_l) in enumerate(jobs):
            gid = gpu_ids[i % len(gpu_ids)]
            _, rc = run_one(gid, cd_l, py_l, env_base)
            if rc != 0:
                print(f"Job {i + 1}/{len(jobs)} failed with exit {rc}", file=sys.stderr)
                return rc
        return 0

    print(
        f"Running {len(jobs)} jobs with workers={workers}, gpu_ids={gpu_ids} "
        f"(round-robin CUDA_VISIBLE_DEVICES; multiple jobs may share a GPU)."
    )
    indexed = [(gpu_ids[i % len(gpu_ids)], cd, py) for i, (cd, py) in enumerate(jobs)]

    failures: list[tuple[int, int]] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(run_one, gid, cd, py, env_base): (k, gid)
            for k, (gid, cd, py) in enumerate(indexed)
        }
        for fut in as_completed(futs):
            k, gid = futs[fut]
            try:
                _, rc = fut.result()
                if rc != 0:
                    failures.append((k, rc))
                    print(f"Job index {k} (CUDA_VISIBLE_DEVICES={gid}) exit {rc}", file=sys.stderr)
            except Exception as e:
                failures.append((k, -1))
                print(f"Job index {k} exception: {e}", file=sys.stderr)

    if failures:
        print(f"Finished with {len(failures)} failing jobs.", file=sys.stderr)
        return 1
    print("All jobs completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
