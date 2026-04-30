from dataclasses import fields, is_dataclass

class SettingsDeltaTracker:
    def __init__(self):
        self.pl_set_delta = {}
        self.settings = None
        self.value_transforms = {}

    def bind_settings_to_vars(self, settings, tk_vars_obj, var_prefix="", var_postfix="", value_transforms=None, on_change_callback=None):
        self.settings = settings
        self.tk_vars_obj = tk_vars_obj
        self.var_prefix = var_prefix
        self.var_postfix = var_postfix
        self.value_transforms = value_transforms or {}
        self.on_change_callback = on_change_callback
        self._bind_recursive(settings)

    def clear_delta(self):
        self.pl_set_delta = {}
        if getattr(self, "on_change_callback", None):
            self.on_change_callback(False)

    def get_delta(self):
        return self.pl_set_delta
    
    def has_delta(self):
        return len(self.pl_set_delta) > 0

    def _get_value_by_path(self, obj, path):
        cur = obj
        for key in path:
            if isinstance(cur, dict):
                if key not in cur:
                    return None
                cur = cur[key]
            else:
                if not hasattr(cur, key):
                    return None
                cur = getattr(cur, key)
        return cur

    def _set_delta_path(self, path, value):
        cur = self.pl_set_delta
        for key in path[:-1]:
            nxt = cur.get(key)
            if not isinstance(nxt, dict):
                nxt = {}
                cur[key] = nxt
            cur = nxt
        cur[path[-1]] = value

    def _remove_delta_path(self, path):
        cur = self.pl_set_delta
        parents = []
        for key in path[:-1]:
            if key not in cur or not isinstance(cur[key], dict):
                return
            parents.append((cur, key))
            cur = cur[key]

        if path[-1] in cur:
            del cur[path[-1]]

        for parent, key in reversed(parents):
            child = parent.get(key)
            if isinstance(child, dict) and len(child) == 0:
                del parent[key]
            else:
                break

    def _make_write_callback(self, path, tk_var):
        def write_callback(*_):
            new_value = tk_var.get()
            old_value = self._get_value_by_path(self.settings, path)
            transform = self.value_transforms.get(".".join(path))

            if transform is not None:
                try:
                    new_value = transform(new_value)
                except Exception:
                    pass

            if isinstance(old_value, bool):
                new_value = bool(new_value)
            elif isinstance(old_value, int) and not isinstance(old_value, bool):
                try:
                    new_value = int(new_value)
                except Exception:
                    pass
            elif isinstance(old_value, float):
                try:
                    new_value = float(new_value)
                except Exception:
                    pass

            if new_value == old_value:
                self._remove_delta_path(path)
            else:
                self._set_delta_path(path, new_value)

            if getattr(self, "on_change_callback", None):
                self.on_change_callback(len(self.pl_set_delta) > 0)

        return write_callback

    def _bind_recursive(self, obj, path_prefix=()):
        if is_dataclass(obj):
            for f in fields(obj):
                self._bind_recursive(getattr(obj, f.name), path_prefix + (f.name,))
            return

        if isinstance(obj, dict):
            for k, v in obj.items():
                self._bind_recursive(v, path_prefix + (str(k),))
            return

        var_name = f"{self.var_prefix}{'_'.join(path_prefix)}{self.var_postfix}"
        tk_var = getattr(self.tk_vars_obj, var_name, None)
        if tk_var is not None and hasattr(tk_var, "trace_add"):
            tk_var.trace_add("write", self._make_write_callback(path_prefix, tk_var))

__all__ = [
    "SettingsDeltaTracker",
]
