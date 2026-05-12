#!/usr/bin/env python3
"""Check whether the current machine state is suitable for stable ADC timing runs."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
from pathlib import Path


def _read_text(path: str) -> str | None:
    p = Path(path)
    if not p.is_file():
        return None
    try:
        return p.read_text().strip()
    except OSError:
        return None


def _parse_meminfo() -> dict[str, int]:
    out: dict[str, int] = {}
    meminfo = Path("/proc/meminfo")
    if not meminfo.is_file():
        return out
    for line in meminfo.read_text().splitlines():
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        parts = raw.strip().split()
        if not parts:
            continue
        try:
            out[key] = int(parts[0])
        except ValueError:
            continue
    return out


def _fmt_gib(kib: int | None) -> str | None:
    if kib is None:
        return None
    return f"{kib / 1024 / 1024:.1f} GiB"


def _module_version(name: str) -> str | None:
    try:
        mod = importlib.import_module(name)
    except Exception:
        return None
    return getattr(mod, "__version__", "present")


def build_report() -> dict:
    load1, load5, load15 = os.getloadavg()
    cpu_count = os.cpu_count() or 0
    meminfo = _parse_meminfo()
    total_mem_kib = meminfo.get("MemTotal")
    avail_mem_kib = meminfo.get("MemAvailable")

    packages = {
        "faiss": _module_version("faiss"),
        "numba": _module_version("numba"),
        "numpy": _module_version("numpy"),
        "scipy": _module_version("scipy"),
        "pandas": _module_version("pandas"),
    }

    governor = _read_text("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    turbo_disabled = _read_text("/sys/devices/system/cpu/intel_pstate/no_turbo")
    smt_active = _read_text("/sys/devices/system/cpu/smt/active")

    env_threads = {
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        "NUMBA_NUM_THREADS": os.environ.get("NUMBA_NUM_THREADS"),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
    }

    warnings: list[str] = []
    blockers: list[str] = []

    if packages["faiss"] is None:
        blockers.append("Python package `faiss` is not available in the current shell.")
    if governor and governor != "performance":
        warnings.append(
            f"CPU scaling governor is `{governor}`; `performance` is preferred for stable timing."
        )
    if turbo_disabled == "0":
        warnings.append(
            "Intel turbo is enabled; results may vary more between runs under thermal or boost changes."
        )
    if smt_active == "1":
        warnings.append(
            "SMT/Hyper-Threading is enabled; pinning to physical cores is safer for microbenchmarks."
        )
    if cpu_count and load1 > max(1.0, cpu_count * 0.25):
        warnings.append(
            f"1-minute load average is {load1:.2f} on {cpu_count} CPUs; background load may distort timings."
        )
    if avail_mem_kib is not None and total_mem_kib is not None:
        if avail_mem_kib / total_mem_kib < 0.10:
            warnings.append("Available memory is below 10% of total RAM.")

    return {
        "host": platform.node(),
        "platform": platform.platform(),
        "cpu_count": cpu_count,
        "load_average": {"1m": load1, "5m": load5, "15m": load15},
        "memory": {
            "total_kib": total_mem_kib,
            "available_kib": avail_mem_kib,
            "total_human": _fmt_gib(total_mem_kib),
            "available_human": _fmt_gib(avail_mem_kib),
        },
        "cpu_controls": {
            "scaling_governor": governor,
            "intel_pstate_no_turbo": turbo_disabled,
            "smt_active": smt_active,
        },
        "python_packages": packages,
        "thread_env": env_threads,
        "warnings": warnings,
        "blockers": blockers,
        "ready": not blockers and not warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assess machine readiness for ADC timing runs."
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON only.")
    args = parser.parse_args()

    report = build_report()
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    print("Benchmark environment check")
    print(f"Host: {report['host']}")
    print(f"Platform: {report['platform']}")
    print(f"CPUs: {report['cpu_count']}")
    la = report["load_average"]
    print(f"Load average: {la['1m']:.2f} {la['5m']:.2f} {la['15m']:.2f}")
    mem = report["memory"]
    print(f"Memory: {mem['available_human']} available / {mem['total_human']} total")

    ctrl = report["cpu_controls"]
    print(f"Governor: {ctrl['scaling_governor'] or 'unknown'}")
    print(f"Turbo disabled: {ctrl['intel_pstate_no_turbo'] or 'unknown'}")
    print(f"SMT active: {ctrl['smt_active'] or 'unknown'}")

    print("Python packages:")
    for name, version in report["python_packages"].items():
        print(f"  {name}: {version or 'missing'}")

    print("Thread env:")
    for name, value in report["thread_env"].items():
        print(f"  {name}: {value or '<unset>'}")

    if report["blockers"]:
        print("Blockers:")
        for item in report["blockers"]:
            print(f"  - {item}")

    if report["warnings"]:
        print("Warnings:")
        for item in report["warnings"]:
            print(f"  - {item}")

    if report["ready"]:
        print("Status: ready")
    elif report["blockers"]:
        print("Status: not ready")
    else:
        print("Status: usable with caveats")


if __name__ == "__main__":
    main()
