"""Auto-tune dataloader parameters based on system resources and model characteristics."""
import os
import gc
import time
import torch


def _get_system_resources():
    """Query GPU, CPU, and RAM resources."""
    gpu_total_mb = torch.cuda.mem_get_info()[1] / 1024**2
    cpu_count = os.cpu_count() or 4

    # Parse /proc/meminfo for RAM (avoids psutil dependency)
    ram_total_mb = 0
    ram_available_mb = 0
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    ram_total_mb = int(line.split()[1]) / 1024
                elif line.startswith("MemAvailable:"):
                    ram_available_mb = int(line.split()[1]) / 1024
    except (FileNotFoundError, ValueError):
        ram_total_mb = 64 * 1024
        ram_available_mb = 32 * 1024

    return dict(
        gpu_total_mb=gpu_total_mb,
        cpu_count=cpu_count,
        ram_total_mb=ram_total_mb,
        ram_available_mb=ram_available_mb,
    )


def _measure_gpu_memory(model, dataset, cfg):
    """Measure per-sample GPU activation memory with test forward passes.

    Returns:
        (base_memory_mb, per_sample_mb, input_tensor_per_sample_mb)
    """
    from trace_tad.datasets.builder import collate

    model.eval()

    # Forward with batch_size=1
    sample_0 = dataset[0]
    batch_1 = {
        k: v.unsqueeze(0).cuda() if isinstance(v, torch.Tensor) else [v]
        for k, v in sample_0.items()
    }

    torch.cuda.reset_peak_memory_stats()
    with torch.amp.autocast("cuda", dtype=torch.float16, enabled=True):
        with torch.inference_mode():
            _ = model.forward_test(batch_1["inputs"], batch_1["masks"], batch_1.get("metas"), cfg.inference)
    torch.cuda.synchronize()
    peak_bs1 = torch.cuda.max_memory_allocated() / 1024**2

    # Forward with batch_size=2 to measure per-sample memory
    del batch_1, _
    gc.collect()
    torch.cuda.empty_cache()

    input_bytes_per_sample = sample_0["inputs"].nelement() * sample_0["inputs"].element_size()

    try:
        samples = [dataset[0], dataset[1 % len(dataset)]]
        batch_2 = collate(samples)
        for k, v in batch_2.items():
            if isinstance(v, torch.Tensor):
                batch_2[k] = v.cuda()
            elif isinstance(v, list) and v and isinstance(v[0], torch.Tensor):
                batch_2[k] = [vv.cuda() for vv in v]

        torch.cuda.reset_peak_memory_stats()
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=True):
            with torch.inference_mode():
                _ = model.forward_test(batch_2["inputs"], batch_2["masks"], batch_2.get("metas"), cfg.inference)
        torch.cuda.synchronize()
        peak_bs2 = torch.cuda.max_memory_allocated() / 1024**2

        per_sample_mb = peak_bs2 - peak_bs1
        base_mb = peak_bs1 - per_sample_mb

        del batch_2, _
    except RuntimeError as e:
        if "out of memory" not in str(e).lower():
            raise
        # OOM at batch_size=2: estimate per_sample from bs=1
        # Assume ~50% of bs=1 peak is per-sample activation
        per_sample_mb = peak_bs1 * 0.5
        base_mb = peak_bs1 - per_sample_mb

    gc.collect()
    torch.cuda.empty_cache()

    return base_mb, per_sample_mb, input_bytes_per_sample / 1024**2


def _benchmark_batch_size(model, dataset, cfg, batch_size, num_runs=2):
    """Benchmark forward pass throughput at a given batch_size.

    Returns:
        per_sample_time_ms or None if OOM.
    """
    from trace_tad.datasets.builder import collate

    n_samples = min(batch_size, len(dataset))
    samples = [dataset[i % len(dataset)] for i in range(n_samples)]
    batch = collate(samples)
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.cuda()
        elif isinstance(v, list) and v and isinstance(v[0], torch.Tensor):
            batch[k] = [vv.cuda() for vv in v]

    result = None
    try:
        # Warmup
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=True):
            with torch.inference_mode():
                model.forward_test(batch["inputs"], batch["masks"], batch.get("metas"), cfg.inference)
        torch.cuda.synchronize()

        # Benchmark
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(num_runs):
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=True):
                with torch.inference_mode():
                    model.forward_test(batch["inputs"], batch["masks"], batch.get("metas"), cfg.inference)
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) / num_runs

        per_sample_ms = (elapsed / n_samples) * 1000
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            per_sample_ms = None
        else:
            raise
    finally:
        del batch
        gc.collect()
        torch.cuda.empty_cache()

    return per_sample_ms


def auto_tune_inference(model, dataset, cfg, logger=None):
    """Auto-tune inference parameters based on system resources and measured throughput.

    Benchmarks multiple batch sizes to find the optimal one (not just the maximum).
    Modifies cfg.solver.test in-place.

    Returns:
        dict with the tuned parameters.
    """
    log = logger.info if logger else print

    log("Auto-tuning inference parameters...")

    # 1. Get system resources
    sys_res = _get_system_resources()
    gpu_total = sys_res["gpu_total_mb"]
    cpu_count = sys_res["cpu_count"]
    ram_available = sys_res["ram_available_mb"]

    log(f"  GPU: {torch.cuda.get_device_name(0)}, {gpu_total:.0f} MB")
    log(f"  CPU: {cpu_count} cores")
    log(f"  RAM: {ram_available:.0f} MB available")

    # 2. Measure GPU memory per sample
    base_mem, per_sample_mem, input_per_sample_mb = _measure_gpu_memory(model, dataset, cfg)
    log(f"  GPU base memory (model+overhead): {base_mem:.0f} MB")
    log(f"  GPU per-sample activation: {per_sample_mem:.0f} MB")

    # 3. Calculate max batch_size from memory (with 15% safety buffer)
    safety_buffer = gpu_total * 0.15
    max_batch_size = int((gpu_total - base_mem - safety_buffer) / per_sample_mem)
    max_batch_size = max(1, min(max_batch_size, len(dataset)))
    log(f"  Max batch_size (memory-limited): {max_batch_size}")

    # 4. Benchmark candidate batch sizes to find optimal throughput
    # Test powers-of-2-ish candidates up to max
    candidates = sorted(set([1, 2, 4, 8] + [max_batch_size]) & set(range(1, max_batch_size + 1)))
    # Also include the current config batch_size
    current_bs = cfg.solver.test.get("batch_size", 4)
    if current_bs not in candidates and 1 <= current_bs <= max_batch_size:
        candidates.append(current_bs)
        candidates.sort()

    log(f"  Benchmarking batch sizes: {candidates}")
    best_bs = 1
    best_throughput = float("inf")

    for bs in candidates:
        per_sample_ms = _benchmark_batch_size(model, dataset, cfg, bs)
        if per_sample_ms is None:
            log(f"    batch_size={bs}: OOM")
            break
        log(f"    batch_size={bs}: {per_sample_ms:.1f} ms/sample")
        if per_sample_ms < best_throughput:
            best_throughput = per_sample_ms
            best_bs = bs

    log(f"  Optimal batch_size: {best_bs} ({best_throughput:.1f} ms/sample)")

    # 5. Calculate optimal num_workers
    # Rule: min(cpu_cores - 2, batch_size * 4, 32)
    # - Leave 2 cores for main thread + post-processing thread
    # - More workers than batch_size*4 gives diminishing returns
    # - Cap at 32 to avoid OS overhead
    num_workers = min(cpu_count - 2, best_bs * 4, 32)
    num_workers = max(2, num_workers)

    # 6. Calculate optimal prefetch_factor
    # Total prefetch memory = num_workers * prefetch_factor * batch_size * input_per_sample_mb
    # Cap at 25% of available RAM
    ram_budget_mb = ram_available * 0.25
    per_batch_cpu_mb = best_bs * input_per_sample_mb
    if per_batch_cpu_mb > 0 and num_workers > 0:
        max_prefetch = int(ram_budget_mb / (num_workers * per_batch_cpu_mb))
        prefetch_factor = max(2, min(max_prefetch, 8))
    else:
        prefetch_factor = 2

    tuned = dict(
        batch_size=best_bs,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=True,
    )

    # Log comparison
    current = dict(cfg.solver.test) if hasattr(cfg.solver, "test") else {}
    log(f"  Tuned parameters:")
    for k, v in tuned.items():
        cur = current.get(k, "N/A")
        changed = " (changed)" if cur != "N/A" and cur != v else ""
        log(f"    {k}: {cur} → {v}{changed}")

    # Apply to config
    for k, v in tuned.items():
        cfg.solver.test[k] = v

    return tuned
