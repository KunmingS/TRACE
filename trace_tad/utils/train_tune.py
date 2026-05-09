"""Training dataloader resource tuning."""
import copy
import gc
import json
import os
import time

from trace_tad.config import Config
from trace_tad.datasets import build_dataset, build_dataloader


TRAIN_RESOURCE_PROFILES = [
    dict(name="Low", num_workers=2, decode_threads=1, prefetch_factor=2),
    dict(name="Balanced", num_workers=4, decode_threads=2, prefetch_factor=2),
    dict(name="High", num_workers=8, decode_threads=2, prefetch_factor=2),
]


def _clone_config(cfg):
    return Config(copy.deepcopy(dict(cfg._cfg_dict)), filename=getattr(cfg, "_filename", None))


def _apply_dataset_paths(cfg, *, annotation_path, class_map, model_dir):
    for split in ("train", "val", "test"):
        if hasattr(cfg.dataset, split):
            cfg.dataset[split].ann_file = annotation_path
            cfg.dataset[split].class_map = class_map
            cfg.dataset[split].data_path = model_dir


def _set_video_init_threads(dataset_cfg, decode_threads):
    for transform in dataset_cfg.pipeline:
        if transform.get("type") == "VideoInit":
            transform["num_threads"] = int(decode_threads)


def _apply_profile(cfg, profile):
    cfg.solver.train.num_workers = int(profile["num_workers"])
    cfg.solver.train.prefetch_factor = int(profile["prefetch_factor"])
    cfg.solver.train.persistent_workers = int(profile["num_workers"]) > 0
    _set_video_init_threads(cfg.dataset.train, int(profile["decode_threads"]))


def _rss_mb():
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except (OSError, ValueError):
        pass
    return 0.0


def _total_ram_mb():
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / 1024
    except (OSError, ValueError):
        pass
    return None


def _guardrail_reason(result, cpu_count, total_ram_mb):
    if cpu_count and result["num_workers"] > max(2, cpu_count):
        return f"{result['name']} uses {result['num_workers']} workers on a {cpu_count}-CPU machine"
    if total_ram_mb and result["peak_rss_mb"] > total_ram_mb * 0.75:
        return f"{result['name']} used more than 75% of system RAM estimate"
    return None


def _detect_cache_mode(annotation_path):
    try:
        with open(annotation_path, "r", encoding="utf-8") as f:
            database = json.load(f).get("database", {})
    except (OSError, json.JSONDecodeError):
        return "unknown"
    if any(info.get("cached_video") for info in database.values()):
        return "cached_video"
    if any(info.get("source_video") for info in database.values()):
        return "virtual"
    return "physical"


def _shutdown_loader(loader):
    try:
        iterator = getattr(loader, "_iterator", None)
        if iterator is not None and hasattr(iterator, "_shutdown_workers"):
            iterator._shutdown_workers()
    except Exception:
        pass


def benchmark_train_profile(base_cfg, profile, *, annotation_path, class_map, model_dir, max_batches=8):
    cfg = _clone_config(base_cfg)
    _apply_dataset_paths(cfg, annotation_path=annotation_path, class_map=class_map, model_dir=model_dir)
    _apply_profile(cfg, profile)

    before_rss = _rss_mb()
    loader = None
    batches = 0
    started = time.perf_counter()
    try:
        dataset = build_dataset(cfg.dataset.train)
        loader = build_dataloader(
            dataset,
            shuffle=True,
            drop_last=True,
            **cfg.solver.train,
        )
        iterator = iter(loader)
        for _ in range(max_batches):
            try:
                batch = next(iterator)
            except StopIteration:
                break
            batches += 1
            del batch
        if batches == 0:
            raise RuntimeError("No training batches were produced")
        elapsed = time.perf_counter() - started
        status = "ok"
        error = None
        avg_batch_ms = (elapsed / batches) * 1000
    except RuntimeError as exc:
        status = "oom" if "out of memory" in str(exc).lower() else "failed"
        error = str(exc)
        avg_batch_ms = None
    except Exception as exc:
        status = "failed"
        error = str(exc)
        avg_batch_ms = None
    finally:
        if loader is not None:
            _shutdown_loader(loader)
        del loader
        gc.collect()

    peak_rss_mb = max(before_rss, _rss_mb())
    result = dict(
        name=profile["name"],
        num_workers=int(profile["num_workers"]),
        decode_threads=int(profile["decode_threads"]),
        prefetch_factor=int(profile["prefetch_factor"]),
        avg_batch_ms=avg_batch_ms,
        peak_rss_mb=peak_rss_mb,
        status=status,
    )
    if error:
        result["error"] = error
    return result


def tune_train_resources(
    config_path,
    *,
    model_dir,
    annotation_path,
    class_map,
    profiles=None,
    max_batches=8,
    logger=None,
):
    log = logger.info if logger else print
    profiles = profiles or TRAIN_RESOURCE_PROFILES
    base_cfg = Config.fromfile(config_path)

    log("Auto-tuning train dataloader resources...")
    log(f"  Dataset: {annotation_path}")
    log(f"  Cache mode: {_detect_cache_mode(annotation_path)}")

    results = []
    for profile in profiles:
        log(
            "  Benchmarking {name}: workers={num_workers}, "
            "decode_threads={decode_threads}, prefetch={prefetch_factor}".format(**profile)
        )
        result = benchmark_train_profile(
            base_cfg,
            profile,
            annotation_path=annotation_path,
            class_map=class_map,
            model_dir=model_dir,
            max_batches=max_batches,
        )
        if result["status"] == "ok":
            log(f"    {result['avg_batch_ms']:.1f} ms/batch, RSS~{result['peak_rss_mb']:.0f} MB")
        else:
            log(f"    {result['status']}: {result.get('error', 'unknown error')}")
        results.append(result)

    successful = [r for r in results if r["status"] == "ok" and r["avg_batch_ms"] is not None]
    notes = []
    cpu_count = os.cpu_count() or 1
    total_ram_mb = _total_ram_mb()
    guarded_successful = []
    for result in successful:
        reason = _guardrail_reason(result, cpu_count, total_ram_mb)
        if reason:
            notes.append(f"Resource guardrail skipped {reason}.")
        else:
            guarded_successful.append(result)

    if guarded_successful:
        recommended = min(guarded_successful, key=lambda r: r["avg_batch_ms"])["name"]
    elif successful:
        recommended = min(successful, key=lambda r: r["avg_batch_ms"])["name"]
        notes.append("No successful profile passed resource guardrails; using fastest successful fallback.")
    else:
        recommended = "Low"

    if _detect_cache_mode(annotation_path) == "cached_video":
        notes.append("Benchmarked on cached video clips; results should be applied to this prepared dataset.")
    else:
        notes.append("Benchmarked on virtual clips; cached video may change the optimal profile.")
    if not successful:
        notes.append("All benchmark profiles failed; falling back to Low.")

    return dict(
        recommended_profile=recommended,
        profiles=results,
        cache_mode=_detect_cache_mode(annotation_path),
        notes=notes,
    )
