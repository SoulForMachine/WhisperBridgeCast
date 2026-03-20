import json
import socket
import numpy as np

MSG_TYPE_JSON = 1
MSG_TYPE_AUDIO = 2

def send_json(sock: socket.socket, obj):
    payload = json.dumps(obj).encode("utf-8")
    header = bytes([MSG_TYPE_JSON]) + len(payload).to_bytes(4, byteorder="big")
    sock.sendall(header + payload)

def send_ndarray(sock: socket.socket, arr: np.ndarray):
    payload = arr.tobytes()
    header = bytes([MSG_TYPE_AUDIO]) + len(payload).to_bytes(4, byteorder="big")
    sock.sendall(header + payload)

def recv_all(sock: socket.socket, n: int) -> bytes | None:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf

def recv_message(sock: socket.socket):
    header = recv_all(sock, 5)
    if not header:
        return None, None

    msg_type = header[0]
    msg_len = int.from_bytes(header[1:5], "big")

    payload = recv_all(sock, msg_len)
    if payload is None:
        return None, None

    if msg_type == MSG_TYPE_JSON:
        return "json", json.loads(payload.decode("utf8"))
    elif msg_type == MSG_TYPE_AUDIO:
        return "audio", np.frombuffer(payload, dtype=np.float32)
