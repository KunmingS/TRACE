class Registry:
    def __init__(self, name):
        self.name = name
        self._module_dict = {}

    def register_module(self, name=None, force=False):
        def decorator(cls):
            key = name if name is not None else cls.__name__
            if key in self._module_dict and not force:
                raise KeyError(f"{key} is already registered in {self.name}")
            self._module_dict[key] = cls
            return cls
        return decorator

    def build(self, cfg, **kwargs):
        cfg = dict(cfg)
        obj_type = cfg.pop("type")
        if obj_type not in self._module_dict:
            raise KeyError(f"'{obj_type}' is not registered in registry '{self.name}'. "
                           f"Available: {list(self._module_dict.keys())}")
        return self._module_dict[obj_type](**cfg, **kwargs)

    def get(self, key):
        return self._module_dict.get(key, None)

    def __contains__(self, key):
        return key in self._module_dict
