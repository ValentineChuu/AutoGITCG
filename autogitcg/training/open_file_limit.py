"""Raise this process's open-file limit to the system's ceiling before starting actor processes.

PyTorch hands tensors to other processes through file descriptors; with many actors, every weight broadcast
holds a few thousand at once. Rented cloud containers often start with a soft limit of 1024 ("Too many open files"
with 10 actors), even when the hard limit allows far more.
"""

import resource


def raise_open_file_limit() -> int:
    """Returns the new soft limit."""
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if hard == resource.RLIM_INFINITY or soft < hard:
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    return resource.getrlimit(resource.RLIMIT_NOFILE)[0]
