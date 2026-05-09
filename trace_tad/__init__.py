_LAZY = {
    "build_detector": ("trace_tad.models", "build_detector"),
    "build_dataset": ("trace_tad.datasets", "build_dataset"),
    "build_dataloader": ("trace_tad.datasets", "build_dataloader"),
    "build_evaluator": ("trace_tad.evaluations", "build_evaluator"),
}


def __getattr__(name):
    if name in _LAZY:
        import importlib
        mod, attr = _LAZY[name]
        return getattr(importlib.import_module(mod), attr)
    raise AttributeError(f"module 'trace_tad' has no attribute {name!r}")


__all__ = list(_LAZY)
