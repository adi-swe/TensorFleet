"""Conservative local-worker planning; estimates are not memory guarantees."""
import os
import platform


def physical_memory():
    if platform.system() == 'Windows':
        import ctypes
        class MemoryStatus(ctypes.Structure):
            _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong) for name in ('total', 'available', 'page_total', 'page_available', 'virtual_total', 'virtual_available', 'extended')]
        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OSError('Could not inspect physical memory')
        return status.total
    return os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')


def plan(algorithm, rows, features, trees=100, max_workers=4, memory_budget=None):
    if min(rows, features, trees, max_workers) < 1:
        raise ValueError('Dataset dimensions and worker/tree budgets must be positive')
    memory_budget = int(physical_memory() * .5 if memory_budget is None else memory_budget)
    if memory_budget <= 0: raise ValueError('Memory budget must be positive')
    cpu_limit = min(os.cpu_count() or 1, max_workers, trees, 32, max(1, rows // 1000))
    estimate = lambda workers: rows * features * 32 + rows * ((trees + workers - 1)//workers) * 128 + 256 * 1024**2
    feasible = [workers for workers in range(1, cpu_limit + 1) if workers * estimate(workers) <= memory_budget]
    if not feasible:
        raise ValueError('Dataset does not fit the estimated memory budget; use a smaller dataset or an out-of-core backend')
    if algorithm == 'fleet_distributed_random_forest':
        workers = max(feasible)
        device = 'cpu'
    elif algorithm == 'fleet_accelerated_logistic':
        from accelerated_logistic import available_devices
        devices = available_devices()
        if not devices: raise ValueError('Accelerated logistic requires optional PyTorch dependencies')
        device = next(candidate for candidate in ('cuda', 'mps', 'cpu') if candidate in devices)
        workers = 1  # This backend owns one device; replicas do not create GPUs.
    else:
        raise ValueError('No local resource policy for this algorithm')
    return {'algorithm': algorithm, 'workers': workers, 'device': device,
            'physical_memory_bytes': physical_memory(), 'memory_budget_bytes': memory_budget,
            'estimated_bytes_per_worker': estimate(workers), 'cpu_count': os.cpu_count(),
            'max_workers': max_workers, 'estimate_scope': 'Heuristic for replicated in-memory datasets; not a peak-memory bound',
            'dataset_rows': rows, 'dataset_features': features}
