#!/usr/bin/env python3
"""Keep a CUDA GPU busy at a target duty cycle."""

from __future__ import annotations

import argparse
import signal
import sys
import time


DEFAULT_UTILIZATION = 0.40
DEFAULT_PERIOD_SECONDS = 1.0
DEFAULT_MATRIX_SIZE = 4096
torch = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run periodic CUDA work to keep GPU utilization near a target value."
    )
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device id to use.")
    parser.add_argument(
        "--utilization",
        "--fraction",
        dest="utilization",
        type=float,
        default=DEFAULT_UTILIZATION,
        help="Target duty cycle, e.g. 0.40 for about 40%% GPU utilization.",
    )
    parser.add_argument(
        "--period",
        type=float,
        default=DEFAULT_PERIOD_SECONDS,
        help="Control period in seconds. Each period runs compute then sleeps.",
    )
    parser.add_argument(
        "--matrix-size",
        type=int,
        default=DEFAULT_MATRIX_SIZE,
        help="Square matrix size used for CUDA matmul workload.",
    )
    parser.add_argument(
        "--memory-fraction",
        type=float,
        default=0.0,
        help="Optional extra fraction of total GPU memory to reserve.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please run in an environment with GPU.")
    if args.gpu < 0 or args.gpu >= torch.cuda.device_count():
        raise ValueError(f"Invalid GPU id {args.gpu}; found {torch.cuda.device_count()} GPU(s).")
    if not 0 < args.utilization < 1:
        raise ValueError("--utilization must be between 0 and 1.")
    if args.period <= 0:
        raise ValueError("--period must be positive.")
    if args.matrix_size <= 0:
        raise ValueError("--matrix-size must be positive.")
    if not 0 <= args.memory_fraction < 1:
        raise ValueError("--memory-fraction must be between 0 and 1.")


def reserve_memory(gpu: int, memory_fraction: float) -> torch.Tensor | None:
    if memory_fraction <= 0:
        return None

    free_bytes, total_bytes = torch.cuda.mem_get_info(gpu)
    target_bytes = int(total_bytes * memory_fraction)
    already_used = total_bytes - free_bytes
    bytes_to_allocate = max(0, target_bytes - already_used)

    if bytes_to_allocate <= 0:
        return None

    elements = bytes_to_allocate // torch.empty((), dtype=torch.float32, device=gpu).element_size()
    if elements <= 0:
        return None
    return torch.empty(elements, dtype=torch.float32, device=gpu)


def make_workload(gpu: int, matrix_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    device = torch.device(f"cuda:{gpu}")
    a = torch.randn((matrix_size, matrix_size), device=device, dtype=torch.float32)
    b = torch.randn((matrix_size, matrix_size), device=device, dtype=torch.float32)
    torch.cuda.synchronize(gpu)
    return a, b


def run_busy_loop(args: argparse.Namespace) -> None:
    torch.cuda.set_device(args.gpu)
    memory_holder = reserve_memory(args.gpu, args.memory_fraction)
    a, b = make_workload(args.gpu, args.matrix_size)

    stopping = False

    def stop(_signum: int, _frame: object) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    active_seconds = args.period * args.utilization
    print(
        f"GPU {args.gpu}: target utilization {args.utilization:.0%}, "
        f"period {args.period:.2f}s, matrix {args.matrix_size}x{args.matrix_size}. "
        "Press Ctrl+C to stop.",
        flush=True,
    )

    while not stopping:
        period_start = time.monotonic()
        active_until = period_start + active_seconds

        while not stopping and time.monotonic() < active_until:
            c = torch.mm(a, b)
            torch.cuda.synchronize(args.gpu)
            a, b = b, c

        sleep_seconds = args.period - (time.monotonic() - period_start)
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    del a, b, memory_holder
    torch.cuda.empty_cache()
    print("Stopped and released GPU memory.", flush=True)


def main() -> int:
    global torch

    args = parse_args()

    import torch as torch_module

    torch = torch_module
    validate_args(args)
    run_busy_loop(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
