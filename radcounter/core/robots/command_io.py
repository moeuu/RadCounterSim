"""Localhost JSON-lines command endpoint and CLI client."""

from __future__ import annotations

import argparse
import json
import socket
import socketserver
import threading
from collections.abc import Mapping
from queue import Empty, Queue

from .input import (
    command_from_wire,
    command_to_wire,
    parse_command_text,
)


class _ThreadedServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


class RobotCommandServer:
    def __init__(self, host: str = "127.0.0.1", port: int = 8766) -> None:
        if host not in {"127.0.0.1", "::1", "localhost"}:
            raise ValueError("Robot command server is intentionally localhost-only")
        self.host = host
        self.port = port
        self._queue: Queue[dict[str, object]] = Queue()
        queue = self._queue

        class Handler(socketserver.StreamRequestHandler):
            def handle(self) -> None:
                raw = self.rfile.readline(65_537)
                try:
                    if len(raw) > 65_536:
                        raise ValueError("command exceeds 64 KiB")
                    payload = json.loads(raw.decode("utf-8"))
                    if not isinstance(payload, dict):
                        raise ValueError("command must be a JSON object")
                    command_from_wire(payload)
                    queue.put(payload)
                    reply = {"ok": True}
                except Exception as exc:
                    reply = {"ok": False, "error": str(exc)}
                self.wfile.write((json.dumps(reply, separators=(",", ":")) + "\n").encode("utf-8"))

        self._server = _ThreadedServer((host, port), Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="radcounter-robot-command-server",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2.0)

    def drain(self, maximum: int = 256) -> tuple[dict[str, object], ...]:
        items = []
        for _ in range(maximum):
            try:
                items.append(self._queue.get_nowait())
            except Empty:
                break
        return tuple(items)


def send_command(
    payload: Mapping[str, object],
    host: str = "127.0.0.1",
    port: int = 8766,
    timeout_s: float = 2.0,
) -> dict[str, object]:
    encoded = (json.dumps(dict(payload), separators=(",", ":")) + "\n").encode("utf-8")
    with socket.create_connection((host, port), timeout=timeout_s) as connection:
        connection.sendall(encoded)
        response = connection.makefile("rb").readline(65_537)
    result = json.loads(response.decode("utf-8"))
    if not result.get("ok"):
        raise RuntimeError(str(result.get("error", "command rejected")))
    return result


def command_main() -> int:
    parser = argparse.ArgumentParser(
        description="Send one robot command to a running RadCounterSim GUI"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--ttl", type=float, default=0.5)
    parser.add_argument(
        "command",
        nargs="+",
        help="ROBOT twist VX VY WZ | joint MODE NAME=VALUE... | gripper NAME F | stop",
    )
    args = parser.parse_args()
    robot_id, command = parse_command_text(" ".join(args.command))
    response = send_command(
        command_to_wire(robot_id, command, ttl_s=args.ttl),
        host=args.host,
        port=args.port,
    )
    print(json.dumps(response, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(command_main())
