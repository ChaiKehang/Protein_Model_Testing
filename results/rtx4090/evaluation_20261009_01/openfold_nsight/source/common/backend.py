"""Small device interface. Unsupported counters stay null, never zero."""
from contextlib import nullcontext
import os
import torch

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


class Backend:
    def __init__(self, name):
        if name.startswith("musa"):
            try:
                import torch_musa  # noqa: F401
            except ImportError as exc:
                raise RuntimeError("MUSA requested, but torch_musa is not installed") from exc
        self.device = torch.device(name)
        if self.device.type not in {"cpu", "cuda", "musa"}:
            raise ValueError("Supported devices: cpu, cuda[:index], musa[:index]")
        self.module = getattr(torch, self.device.type, None) if self.device.type != "cpu" else None
        if self.module is not None:
            if not self.module.is_available():
                raise RuntimeError(f"{self.device.type} is unavailable")
            if self.device.index is None:
                self.device = torch.device(self.device.type, self.module.current_device())
            if self.device.index >= self.module.device_count():
                raise ValueError(f"Device index {self.device.index} is outside visible device count")

    def context(self):
        return self.module.device(self.device) if self.module else nullcontext()

    def synchronize(self):
        if self.module:
            with self.context():
                self.module.synchronize()

    def seed(self, value):
        torch.manual_seed(value)
        if self.module and hasattr(self.module, "manual_seed"):
            with self.context():
                self.module.manual_seed(value)

    def reset_peak(self):
        if self.module and hasattr(self.module, "reset_peak_memory_stats"):
            with self.context():
                self.module.reset_peak_memory_stats()

    def memory(self):
        def read(name):
            if self.module is None or not hasattr(self.module, name):
                return None
            with self.context():
                return getattr(self.module, name)() / 1024**3
        return {"allocated_peak_gib": read("max_memory_allocated"),
                "reserved_peak_gib": read("max_memory_reserved"),
                "allocated_current_gib": read("memory_allocated")}

    def events(self):
        if self.module is None or not hasattr(self.module, "Event"):
            return None
        with self.context():
            return (self.module.Event(enable_timing=True), self.module.Event(enable_timing=True))

    def metadata(self):
        result = {"type": self.device.type, "index": self.device.index, "name": "CPU"}
        if self.module and hasattr(self.module, "get_device_name"):
            with self.context():
                result["name"] = self.module.get_device_name()
        result['visible_devices'] = os.environ.get(f'{self.device.type.upper()}_VISIBLE_DEVICES')
        if self.module and hasattr(self.module, 'get_device_properties'):
            properties = self.module.get_device_properties(self.device)
            result['properties_repr'] = str(properties)
            for name in ['total_memory', 'major', 'minor', 'multi_processor_count', 'warp_size', 'uuid']:
                value = getattr(properties, name, None)
                result[name] = str(value) if name == 'uuid' and value is not None else value
            if result['total_memory'] is not None:
                result['total_memory_gib'] = result['total_memory'] / 1024**3
        return result
