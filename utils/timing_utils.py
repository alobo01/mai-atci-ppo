import time
import threading
from functools import wraps
from typing import Callable, Dict, TypeVar, Any

F = TypeVar('F', bound=Callable[..., Any])

class Timing:
    """Thread-safe context manager and storage for timing code blocks."""

    def __init__(self) -> None:
        # Shared across all threads:
        self.totals: Dict[str, float] = {}
        self.counts: Dict[str, int] = {}
        # Per-thread storage of active keys & start times:
        self._local = threading.local()

    def _ensure_thread_state(self):
        """Make sure this thread has its stack & start_times dict."""
        if not hasattr(self._local, "stack"):
            self._local.stack = []               # type: ignore
            self._local.start_times = {}         # type: ignore

    def __enter__(self) -> 'Timing':
        # Allows `with timer:` but doesn't start any key
        return self

    def __call__(self, key: str) -> 'Timing':
        """Start timing context for a given key."""
        self._ensure_thread_state()
        # push onto this thread's stack
        self._local.stack.append(key)           # type: ignore
        # record start time under that key
        self._local.start_times[key] = time.perf_counter()  # type: ignore
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Stop timing for the key most recently __call__'d in this thread."""
        self._ensure_thread_state()
        if not self._local.stack:               # type: ignore
            # nothing to stop
            return

        key = self._local.stack.pop()           # type: ignore
        start = self._local.start_times.pop(key)  # type: ignore
        elapsed = time.perf_counter() - start

        # update the shared totals & counts
        self.totals[key] = self.totals.get(key, 0.0) + elapsed
        self.counts[key] = self.counts.get(key, 0) + 1

    def start(self, key: str) -> None:
        """Explicitly start timing for a key (thread-local)."""
        self._ensure_thread_state()
        self._local.start_times[key] = time.perf_counter()  # type: ignore

    def stop(self, key: str) -> None:
        """Explicitly stop timing for a key (thread-local)."""
        self._ensure_thread_state()
        if key not in self._local.start_times:  # type: ignore
            print(f"Warning: Timer key '{key}' was stopped without being started.")
            return
        start = self._local.start_times.pop(key)  # type: ignore
        elapsed = time.perf_counter() - start

        self.totals[key] = self.totals.get(key, 0.0) + elapsed
        self.counts[key] = self.counts.get(key, 0) + 1

    def summary(self, reset: bool = True) -> Dict[str, Dict[str, float]]:
        """
        Returns average times per key and optionally resets the timer.

        Args:
            reset: If True, clears stored timings after generating the summary.

        Returns:
            { key: { total_ms, count, avg_ms } }
        """
        result: Dict[str, Dict[str, float]] = {}
        for key, total in self.totals.items():
            cnt = self.counts.get(key, 0)
            result[key] = {
                "total_ms": total * 1000.0,
                "count": cnt,
                "avg_ms": (total / cnt) * 1000.0 if cnt else 0.0,
            }

        if reset:
            self.totals.clear()
            self.counts.clear()
            # also clear any per-thread state
            # note: threads will re-init on next use
            for attr in ("stack", "start_times"):
                if hasattr(self._local, attr):
                    delattr(self._local, attr)

        return result

def timed(key: str, timer_instance: Timing) -> Callable[[F], F]:
    """
    Decorator to time a function using a provided Timing instance.

    Usage: 
        @timed("myfunc", my_timer)
        def myfunc(...): ...
    """
    def decorator(fn: F) -> F:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            timer_instance.start(key)
            try:
                return fn(*args, **kwargs)
            finally:
                timer_instance.stop(key)
        return wrapper  # type: ignore
    return decorator
