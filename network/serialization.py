import pickle
from typing import List, Any


def serialize_int(x: int, num_bytes: int = 16) -> bytes:
    return x.to_bytes(num_bytes, 'big')


def deserialize_int(b: bytes) -> int:
    return int.from_bytes(b, 'big')


def serialize_vector(vec: List[int], num_bytes: int = 16) -> bytes:
    serialized = []
    for x in vec:
        serialized.append(x.to_bytes(num_bytes, 'big'))
    length_bytes = len(vec).to_bytes(4, 'big')
    return length_bytes + b''.join(serialized)


def deserialize_vector(data: bytes, num_bytes: int = 16) -> List[int]:
    length = int.from_bytes(data[:4], 'big')
    result = []
    for i in range(length):
        start = 4 + i * num_bytes
        end = start + num_bytes
        result.append(int.from_bytes(data[start:end], 'big'))
    return result


def serialize_message(msg: Any) -> bytes:
    return pickle.dumps(msg)


def deserialize_message(data: bytes) -> Any:
    return pickle.loads(data)
