from __future__ import annotations
import numpy as np
import json
from multiprocessing import Queue
from typing import Protocol, Any, Callable, List
import queue
import c4



def np_encoder(obj):
    if isinstance(obj, (np.integer, np.int_, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)  # safe fallback for other odd types

def jprint(label, obj):
    print(f"\n{label} =")
    print(json.dumps(obj, indent=2, default=np_encoder, ensure_ascii=False))

# Hyperparameter schedules
def linear_sched(start, end, steps):
    return lambda t: start + (end - start) * min(1.0, t / steps)

class SummaryWriterLike(Protocol):
    def add_scalar(self, tag: str, value: float, step: int | None = None) -> None: ...
    def with_tag_prefix(self, tag_prefix: str) -> SummaryWriterLike: ...

# Summary writer that sends requests through a Queue
class SummaryWriterProxy:
    def __init__(self, q: Queue, global_prefix: str = None):
        self.q = q
        self.global_prefix = global_prefix

    def add_scalar(self, tag: str, value: float, step: int | None = None) -> None:
        if self.global_prefix is not None:
            tag = f"{self.global_prefix}/{tag}"
        try:
            self.q.put(("add_scalar", (tag, value, step), {}))  # (method, args, kwargs)
        except queue.Full:
            print("Could not log scalar: queue full")

    def with_tag_prefix(self, tag_prefix: str) -> SummaryWriterLike:
        if self.global_prefix is not None:
            tag_prefix = f"{self.global_prefix}/{tag_prefix}"
        return SummaryWriterProxy(self.q, tag_prefix)


# Ring buffer

class RecentValuesRingBuffer:
    def __init__(self, size: int = 100):
        self.size = size
        self.buffer: List[float] = []
        self.index = 0
        self.sum = 0.0

    def add(self, score: float) -> None:
        if len(self.buffer) < self.size:
            self.buffer.append(score)
            self.sum += score
        else:
            self.sum -= self.buffer[self.index]
            self.sum += score
            self.buffer[self.index] = score
            self.index = (self.index + 1) % self.size

    def average_if_full(self) -> float:
        if len(self.buffer) < self.size:
            return 0.0
        return self.sum / self.size

def print_transition(tr: c4.Transition) -> None:
    print(f"STATE {tr.a} {tr.r} STATE2 {tr.mask} {tr.mask2}")



