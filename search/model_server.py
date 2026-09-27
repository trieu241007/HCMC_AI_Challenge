"""Persistent localhost server for Vietnamese text-to-FAISS search.

Start this file once. Translation, PE-Core, FAISS, and metadata remain
loaded while the process is alive, so clients do not reload model weights.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock
from typing import Any, Protocol


SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8765
MAX_REQUEST_BYTES = 1_048_576


class SearchRuntime(Protocol):
    def search(
        self,
        query_text: str,
        object_query: dict[str, int],
        top_k: int,
        nprobe: int,
    ) -> dict[str, object]: ...


class ModelHTTPServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        runtime: SearchRuntime,
    ) -> None:
        super().__init__(server_address, ModelRequestHandler)
        self.runtime = runtime
        self.inference_lock = Lock()


class ModelRequestHandler(BaseHTTPRequestHandler):
    server: ModelHTTPServer

    def _send_json(self, status: int, document: object) -> None:
        body = json.dumps(document, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path != "/health":
            self._send_json(404, {"error": "Not found"})
            return
        runtime = self.server.runtime
        index = getattr(runtime, "index", None)
        self._send_json(
            200,
            {
                "status": "ready",
                "device": str(getattr(runtime, "device", "unknown")),
                "index_dimension": getattr(index, "d", None),
                "index_total": getattr(index, "ntotal", None),
            },
        )
    def do_POST(self) -> None:
        if self.path != "/search":
            self._send_json(404, {"error": "Not found"})
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0 or content_length > MAX_REQUEST_BYTES:
                raise ValueError("Invalid request size")
            payload = json.loads(self.rfile.read(content_length))
            query_text = payload["query_text"]
            object_query = payload.get("object", {})
            top_k = payload.get("top_k", 200)
            nprobe = payload.get("nprobe", 128)
            with self.server.inference_lock:
                result = self.server.runtime.search(
                    query_text,
                    object_query,
                    top_k,
                    nprobe,
                )
            self._send_json(200, result)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            self._send_json(400, {"error": str(error)})
        except Exception as error:
            self._send_json(500, {"error": str(error)})

    def log_message(self, format: str, *args: Any) -> None:
        print(f"Client {self.client_address[0]}: {format % args}")


def create_server(
    runtime: SearchRuntime,
    host: str,
    port: int,
) -> ModelHTTPServer:
    return ModelHTTPServer((host, port), runtime)



if __name__ == "__main__":
    import search_faiss

    search_faiss.configure_console_utf8()
    selected_device = search_faiss.select_device()
    print(f"Loading persistent search resources on {selected_device}...")
    persistent_runtime = search_faiss.PersistentSearchRuntime.load(
        device=selected_device,
        index_path=search_faiss.INDEX_PATH,
        metadata_path=search_faiss.METADATA_PATH,
        object_data_path=search_faiss.OBJECT_DATA_PATH,
    )
    http_server = create_server(
        persistent_runtime,
        SERVER_HOST,
        SERVER_PORT,
    )
    print(
        f"Server ready: http://{SERVER_HOST}:{SERVER_PORT} "
        f"| vectors={persistent_runtime.index.ntotal:,} "
        f"| dimension={persistent_runtime.index.d}"
    )
    print("Press Ctrl+C to stop and release model weights.")
    try:
        http_server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping model server...")
    finally:
        http_server.server_close()