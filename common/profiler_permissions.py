"""Read-only NVIDIA counter capability check, shared by both model collectors."""
from pathlib import Path


def counter_allowed(policy, capabilities):
    if policy == '0':
        return True
    if policy == '1':
        return bool(capabilities & (1 << 21))  # CAP_SYS_ADMIN
    return None


def counter_policy():
    try:
        lines = Path('/proc/driver/nvidia/params').read_text().splitlines()
        value = next(x.split(':', 1)[1].strip() for x in lines
                     if x.startswith('RmProfilingAdminOnly:'))
    except (OSError, StopIteration):
        value = None
    try:
        cap = int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines()
                       if x.startswith('CapEff:')), 16)
    except (OSError, StopIteration, ValueError):
        cap = 0
    return {'RmProfilingAdminOnly': value, 'CAP_SYS_ADMIN': bool(cap & (1 << 21)),
            'collection_allowed': counter_allowed(value, cap)}
