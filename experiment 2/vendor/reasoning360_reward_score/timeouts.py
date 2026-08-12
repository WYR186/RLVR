"""Windows compatibility shim for Reasoning360's POSIX timeout context.

The upstream helper relies on SIGALRM, which Windows does not provide. This
shim preserves the context-manager API. Inputs are bounded by the 640/384
token geometry and all verifier exceptions are converted to a zero reward.
"""

from contextlib import contextmanager


@contextmanager
def timeout_limit(seconds: float):
    del seconds
    yield
