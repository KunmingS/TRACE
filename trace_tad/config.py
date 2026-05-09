"""
Python-file config loader with _base_ inheritance.
Replaces mmengine.config.Config and DictAction.
"""
import os
import sys
import copy
import importlib.util
import argparse
from pathlib import Path


def _load_py_file(filepath):
    """Load a Python config file as a module and return its namespace dict."""
    filepath = os.path.abspath(filepath)
    spec = importlib.util.spec_from_file_location("_config_module_", filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # Filter out built-ins
    cfg_dict = {
        k: v for k, v in mod.__dict__.items()
        if not k.startswith("__") and (not callable(v) or isinstance(v, (dict, list, tuple, str, int, float, bool, type(None))))
    }
    return cfg_dict


def _merge_dicts(base, override):
    """Recursively merge override into base (override wins)."""
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _merge_dicts(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result


def _load_config_file(filepath):
    """Load a config file, resolving _base_ inheritance recursively."""
    filepath = os.path.abspath(filepath)
    cfg_dir = os.path.dirname(filepath)

    raw = _load_py_file(filepath)

    base_paths = raw.pop("_base_", None)
    if base_paths is None:
        return raw

    if isinstance(base_paths, str):
        base_paths = [base_paths]

    merged = {}
    for base_path in base_paths:
        if not os.path.isabs(base_path):
            base_path = os.path.join(cfg_dir, base_path)
        base_cfg = _load_config_file(base_path)
        merged = _merge_dicts(merged, base_cfg)

    # Child overrides parent
    merged = _merge_dicts(merged, raw)
    return merged


def _parse_value(value_str):
    """Try to parse a string as Python literal, fallback to string."""
    import ast
    try:
        return ast.literal_eval(value_str)
    except (ValueError, SyntaxError):
        return value_str


def _set_nested(cfg_dict, key_path, value):
    """Set a value in a nested dict using dot-separated key path."""
    keys = key_path.split(".")
    d = cfg_dict
    for k in keys[:-1]:
        if isinstance(d, list):
            idx = int(k)
            d = d[idx]
        else:
            if k not in d:
                d[k] = {}
            d = d[k]
    last = keys[-1]
    if isinstance(d, list):
        d[int(last)] = value
    else:
        d[last] = value


class ConfigDict(dict):
    """A dict subclass that supports attribute access."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'ConfigDict' has no attribute '{name}'")

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError:
            raise AttributeError(f"'ConfigDict' has no attribute '{name}'")

    def copy(self):
        return copy.copy(self)


def _to_config_dict(obj):
    """Recursively convert dicts to ConfigDict."""
    if isinstance(obj, dict):
        return ConfigDict({k: _to_config_dict(v) for k, v in obj.items()})
    elif isinstance(obj, (list, tuple)):
        converted = [_to_config_dict(x) for x in obj]
        return type(obj)(converted)
    return obj


class Config:
    """Simple Python-file config loader with _base_ inheritance."""

    def __init__(self, cfg_dict=None, filename=None):
        if cfg_dict is None:
            cfg_dict = {}
        object.__setattr__(self, "_cfg_dict", _to_config_dict(cfg_dict))
        object.__setattr__(self, "_filename", filename)

    @staticmethod
    def fromfile(filepath):
        cfg_dict = _load_config_file(filepath)
        return Config(cfg_dict, filename=filepath)

    def merge_from_dict(self, options):
        """Merge a flat dict of overrides (supports dot-notation keys)."""
        cfg_dict = dict(self._cfg_dict)
        for key, value in options.items():
            _set_nested(cfg_dict, key, value)
        object.__setattr__(self, "_cfg_dict", _to_config_dict(cfg_dict))

    def __getattr__(self, name):
        return getattr(self._cfg_dict, name)

    def __setattr__(self, name, value):
        self._cfg_dict[name] = value

    def __contains__(self, key):
        return key in self._cfg_dict

    def __repr__(self):
        return f"Config(filename={self._filename})"

    @property
    def pretty_text(self):
        import pprint
        return pprint.pformat(dict(self._cfg_dict))

    def get(self, key, default=None):
        return self._cfg_dict.get(key, default)


class DictAction(argparse.Action):
    """
    argparse action for key=value pairs that get merged into a dict.
    Supports: --cfg-options key=value key2=value2 nested.key=value
    """

    def __call__(self, parser, namespace, values, option_string=None):
        options = {}
        for kv in values:
            if "=" not in kv:
                raise argparse.ArgumentTypeError(f"Expected key=value, got: {kv}")
            key, value = kv.split("=", 1)
            options[key] = _parse_value(value)
        setattr(namespace, self.dest, options)
