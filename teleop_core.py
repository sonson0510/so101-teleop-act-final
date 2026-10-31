"""Backend-independent command contract for a single six-axis robot."""
from dataclasses import dataclass
from typing import Protocol
import time
import numpy as np


@dataclass(frozen=True)
class Observation:
    state: np.ndarray
    image: np.ndarray
    acquired_at: float


@dataclass(frozen=True)
class SendResult:
    accepted: bool
    action: np.ndarray | None
    sent_at: float
    reason: str = ""
    limited: bool = False


class RobotBackend(Protocol):
    def connect(self) -> None: ...
    def observe(self) -> Observation: ...
    def send_action(self, action: np.ndarray) -> SendResult: ...
    def stop(self) -> None: ...
    def disconnect(self) -> None: ...


class CommandLimiter:
    def __init__(self, low, high, max_delta):
        self.low = np.asarray(low, dtype=float)
        self.high = np.asarray(high, dtype=float)
        self.max_delta = np.asarray(max_delta, dtype=float)
        if not (self.low.shape == self.high.shape == self.max_delta.shape == (6,)):
            raise ValueError("Limits must be six-vectors")
        if np.any(self.low >= self.high) or np.any(self.max_delta <= 0):
            raise ValueError("Invalid limits")

    def apply(self, proposed, previous):
        proposed = np.asarray(proposed, dtype=float)
        previous = np.asarray(previous, dtype=float)
        if proposed.shape != (6,) or previous.shape != (6,) or not np.isfinite(proposed).all():
            raise ValueError("Invalid command")
        bounded = np.clip(proposed, self.low, self.high)
        bounded = np.clip(bounded, previous - self.max_delta, previous + self.max_delta)
        return bounded.astype(np.float32), not np.array_equal(bounded, proposed)


class MockBackend:
    """No-reset backend for testing rejection, delay and disconnect behavior."""
    def __init__(self, delay_s=0.0):
        self.delay_s = delay_s
        self.connected = False
        self.reject_next = False
        self.state = np.zeros(6, dtype=np.float32)
        self.sent = []

    def connect(self):
        self.connected = True

    def observe(self):
        if not self.connected:
            raise ConnectionError("Mock disconnected")
        return Observation(self.state.copy(), np.zeros((240, 320, 3), dtype=np.uint8), time.perf_counter())

    def send_action(self, action):
        if not self.connected:
            return SendResult(False, None, time.perf_counter(), "disconnected")
        if self.delay_s:
            time.sleep(self.delay_s)
        if self.reject_next:
            self.reject_next = False
            return SendResult(False, None, time.perf_counter(), "injected rejection")
        self.state = np.asarray(action, dtype=np.float32).copy()
        self.sent.append(self.state.copy())
        return SendResult(True, self.state.copy(), time.perf_counter())

    def stop(self):
        pass

    def disconnect(self):
        self.connected = False


class CommandRunner:
    """Single-source command arbitration, limiting, and backend rejection handling."""
    def __init__(self, backend: RobotBackend, limiter: CommandLimiter, source="manual"):
        self.backend = backend
        self.limiter = limiter
        self.source = source
        self.connected = False
        self.stopped = False
        self.last_action = None

    def connect(self):
        self.backend.connect()
        self.last_action = self.backend.observe().state.astype(np.float32).copy()
        self.connected = True
        self.stopped = False

    def submit(self, proposed, *, source="manual"):
        now = time.perf_counter()
        if not self.connected or self.stopped or source != self.source:
            return SendResult(False, None, now, "not authorized or stopped")
        try:
            action, limited = self.limiter.apply(proposed, self.last_action)
        except ValueError as exc:
            return SendResult(False, None, now, str(exc))
        result = self.backend.send_action(action)
        if not result.accepted:
            if result.reason == "disconnected":
                self.connected = False
            return result
        self.last_action = result.action.copy()
        return SendResult(True, result.action.copy(), result.sent_at, limited=limited)

    def stop(self):
        self.stopped = True
        self.backend.stop()

    def resume(self):
        if not self.connected:
            raise ConnectionError("Backend disconnected")
        self.last_action = self.backend.observe().state.astype(np.float32).copy()
        self.stopped = False

    def disconnect(self):
        self.backend.disconnect()
        self.connected = False

