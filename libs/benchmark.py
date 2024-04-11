import logging
import time

log = logging.getLogger(__name__)


class Benchmark:
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, ty, val, tb):
        elapsed = time.perf_counter() - self.start
        name = self.name if self.name else "Block"
        log.debug(f"{name} executed in {elapsed:.3f} seconds.")
        return False

    def __call__(self, func):
        def wrapped(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return wrapped
