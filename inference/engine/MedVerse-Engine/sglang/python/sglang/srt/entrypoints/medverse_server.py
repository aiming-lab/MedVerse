"""MedVerse HTTP Server entry point.

Identical to http_server.launch_server() except it wires in
MedVerseTokenizerManager + MedVerseScheduler via
_launch_medverse_subprocesses() instead of the stock _launch_subprocesses().

Usage (replaces `python -m sglang.launch_server`):
    python -m sglang.srt.entrypoints.medverse_server \
        --model-path <path> --tp-size 1 --port 30000 --trust-remote-code
"""

from __future__ import annotations

import multiprocessing
import threading
from typing import Callable, Optional

import uvicorn

from sglang.srt.entrypoints.http_server import (
    _GlobalState,
    _wait_and_warmup,
    add_api_key_middleware,
    app,
    set_global_state,
    set_uvicorn_logging_configs,
)
from sglang.srt.entrypoints.medverse_engine import _launch_medverse_subprocesses
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import configure_logger


def launch_medverse_server(
    server_args: ServerArgs,
    pipe_finish_writer: Optional[multiprocessing.connection.Connection] = None,
    launch_callback: Optional[Callable[[], None]] = None,
) -> None:
    """Launch MedVerse HTTP server with fork/join scheduler."""
    configure_logger(server_args)

    tokenizer_manager, scheduler_info = _launch_medverse_subprocesses(
        server_args=server_args
    )

    set_global_state(
        _GlobalState(
            tokenizer_manager=tokenizer_manager,
            scheduler_info=scheduler_info,
        )
    )

    if server_args.api_key:
        add_api_key_middleware(app, server_args.api_key)

    warmup_thread = threading.Thread(
        target=_wait_and_warmup,
        args=(
            server_args,
            pipe_finish_writer,
            getattr(tokenizer_manager, "image_token_id", None),
            launch_callback,
        ),
    )
    app.warmup_thread = warmup_thread

    set_uvicorn_logging_configs()
    app.server_args = server_args

    uvicorn.run(
        app,
        host=server_args.host,
        port=server_args.port,
        log_level=server_args.log_level_http or server_args.log_level,
        timeout_keep_alive=5,
        loop="uvloop",
    )


def main() -> None:
    import os
    import sys
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree

    server_args = prepare_server_args(sys.argv[1:])
    try:
        launch_medverse_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__main__":
    main()
