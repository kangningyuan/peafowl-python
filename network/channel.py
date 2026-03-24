import pickle
import threading
from queue import Queue, Empty
from typing import Any, Optional, List


class Channel:
    def __init__(self, name: str = ""):
        self.name = name
        self.send_queue = Queue()
        self.recv_queue = Queue()
        self._closed = False

    def send(self, message: Any):
        if self._closed:
            raise RuntimeError("Channel is closed")
        self.send_queue.put(message)

    def recv(self, timeout: Optional[float] = None) -> Any:
        if self._closed:
            raise RuntimeError("Channel is closed")
        try:
            return self.recv_queue.get(timeout=timeout)
        except Empty:
            return None

    def close(self):
        self._closed = True

    def is_closed(self) -> bool:
        return self._closed


class MessageQueue:
    def __init__(self):
        self.queues = {}
        self.lock = threading.Lock()

    def create_channel(self, name: str) -> Channel:
        with self.lock:
            ch = Channel(name)
            self.queues[name] = ch
        return ch

    def get_channel(self, name: str) -> Optional[Channel]:
        with self.lock:
            return self.queues.get(name)

    def list_channels(self) -> List[str]:
        with self.lock:
            return list(self.queues.keys())


class BroadcastChannel:
    def __init__(self, party_id: str):
        self.party_id = party_id
        self.listeners = []
        self.lock = threading.Lock()

    def broadcast(self, message: Any):
        with self.lock:
            for listener in self.listeners:
                listener.put(message)

    def subscribe(self, queue):
        with self.lock:
            self.listeners.append(queue)

    def receive(self, timeout: Optional[float] = None) -> Any:
        pass
