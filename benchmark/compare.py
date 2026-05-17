#!/usr/bin/env python3
"""Compare Quadtrix C++ and Python benchmark JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_RESULTS = Path(__file__).resolve().parent / "results"


def load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def index_rows(result: dict[str, Any]) -> dict[tuple[str, str, int, int], dict[str, Any]]:
    indexed = {}
    for row in result.get("results", []):
        key = (
            row.get("suite", ""),
            row.get("name", ""),
            int(row.get("batch_size") or 0),
            int(row.get("sequence_length") or 0),
        )
        indexed[key] = row
    return indexed


def pct(new: float, old: float) -> float:
    if old == 0:
        return 0.0
    return (new - old) / old * 100.0


def compare_backends(cpp_path: Path, python_path: Path) -> int:
    missing = [str(path) for path in (cpp_path, python_path) if not path.exists()]
    if missing:
        print("Missing benchmark result file(s):")
        for path in missing:
            print(f"  {path}")
        print("Run benchmark/run_all.py first, or pass explicit --cpp/--python paths.")
        return 1

    cpp = load(cpp_path)
    py = load(python_path)
    cpp_rows = index_rows(cpp)
    py_rows = index_rows(py)

    common = sorted(set(cpp_rows) & set(py_rows))
    if not common:
        print("No matching benchmark rows found.")
        return 1

    print("Quadtrix C++ vs Python Benchmark Comparison")
    print(f"C++:    {cpp_path}")
    print(f"Python: {python_path}")
    print()
    print(f"{'suite':<12} {'name':<24} {'shape':<10} {'cpp ms':>10} {'py ms':>10} {'cpp tok/s':>12} {'py tok/s':>12} {'latency':>10}")
    print("-" * 110)

    for key in common:
        suite, name, batch, seq = key
        c = cpp_rows[key]
        p = py_rows[key]
        cpp_ms = float(c.get("avg_ms") or 0.0)
        py_ms = float(p.get("avg_ms") or 0.0)
        cpp_tps = float(c.get("tokens_per_sec") or 0.0)
        py_tps = float(p.get("tokens_per_sec") or 0.0)
        shape = f"{batch}x{seq}" if batch or seq else "-"
        delta = pct(cpp_ms, py_ms)
        print(
            f"{suite:<12} {name:<24} {shape:<10} "
            f"{cpp_ms:10.3f} {py_ms:10.3f} {cpp_tps:12.1f} {py_tps:12.1f} {delta:+9.1f}%"
        )
    return 0


def compare_baseline(current_path: Path, baseline_path: Path, threshold_pct: float) -> int:
    missing = [str(path) for path in (current_path, baseline_path) if not path.exists()]
    if missing:
        print("Missing benchmark result file(s):")
        for path in missing:
            print(f"  {path}")
        return 1

    current = load(current_path)
    baseline = load(baseline_path)
    current_rows = index_rows(current)
    baseline_rows = index_rows(baseline)
    common = sorted(set(current_rows) & set(baseline_rows))

    print("Quadtrix Benchmark Baseline Comparison")
    print(f"Current:  {current_path}")
    print(f"Baseline: {baseline_path}")
    print()

    regressions = []
    for key in common:
        c = current_rows[key]
        b = baseline_rows[key]
        delta = pct(float(c.get("avg_ms") or 0.0), float(b.get("avg_ms") or 0.0))
        if delta > threshold_pct:
            regressions.append((key, delta, b, c))

    if not regressions:
        print(f"No latency regressions over {threshold_pct:.1f}%.")
        return 0

    print(f"Latency regressions over {threshold_pct:.1f}%:")
    for key, delta, b, c in regressions:
        suite, name, batch, seq = key
        print(
            f"  {suite}/{name} {batch}x{seq}: "
            f"{float(b.get('avg_ms') or 0.0):.3f} ms -> {float(c.get('avg_ms') or 0.0):.3f} ms ({delta:+.1f}%)"
        )
    return 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Quadtrix benchmark results.")
    parser.add_argument("--cpp", type=Path, default=DEFAULT_RESULTS / "cpp_benchmark.json")
    parser.add_argument("--python", type=Path, default=DEFAULT_RESULTS / "python_benchmark.json")
    parser.add_argument("--current", type=Path, default=None)
    parser.add_argument("--baseline", type=Path, default=None)
    parser.add_argument("--threshold-pct", type=float, default=10.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.current and args.baseline:
        return compare_baseline(args.current, args.baseline, args.threshold_pct)
    return compare_backends(args.cpp, args.python)


if __name__ == "__main__":
    raise SystemExit(main())
