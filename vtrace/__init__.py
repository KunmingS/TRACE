_LAZY = {
    "build_detector": ("vtrace.models", "build_detector"),
    "build_dataset": ("vtrace.datasets", "build_dataset"),
    "build_dataloader": ("vtrace.datasets", "build_dataloader"),
    "build_evaluator": ("vtrace.evaluations", "build_evaluator"),
}


def __getattr__(name):
    if name in _LAZY:
        import importlib
        mod, attr = _LAZY[name]
        return getattr(importlib.import_module(mod), attr)
    raise AttributeError(f"module 'vtrace' has no attribute {name!r}")


__all__ = list(_LAZY)
