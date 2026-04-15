"""MedVerse Engine – entry point that wires MedVerse components into Multiverse.

Architecture (3-process, mirroring Multiverse):

    Main process
    ├─ MedVerseTokenizerManager  (ZMQ send to scheduler)
    └─ uvicorn / FastAPI HTTP server

    Subprocess 1: run_medverse_scheduler_process
    └─ MedVerseScheduler (pre-fork, DAG-weighted merge, ZMQ recv/send)

    Subprocess 2: run_detokenizer_process  [unchanged from Multiverse]
    └─ DetokenizerManager (ZMQ recv + response routing)

Key substitutions vs Multiverse's _launch_subprocesses:
  - TokenizerManager         →  MedVerseTokenizerManager
  - run_scheduler_process    →  run_medverse_scheduler_process
                                (instantiates MedVerseScheduler instead of Scheduler)
"""

from __future__ import annotations

import atexit
import logging
import multiprocessing as mp
from typing import Dict, Iterator, List, Optional, Tuple, Union

import zmq

from sglang.srt.entrypoints.EngineBase import EngineBase
from sglang.srt.entrypoints.engine import (
    _set_envs_and_config,
    load_chat_template_for_openai_api,
    guess_chat_template_name_from_model_path,
)
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.data_parallel_controller import run_data_parallel_controller_process
from sglang.srt.managers.detokenizer_manager import run_detokenizer_process
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import (
    configure_logger,
    get_zmq_socket,
    kill_itself_when_parent_died,
    launch_dummy_health_check_server,
    prepare_model_and_tokenizer,
    suppress_other_loggers,
)
from sglang.srt.torch_memory_saver_adapter import TorchMemorySaverAdapter

from sglang.srt.managers.medverse_scheduler import MedVerseScheduler
from sglang.srt.managers.medverse_tokenizer_manager import MedVerseTokenizerManager

logger = logging.getLogger(__name__)


# ── MedVerse scheduler subprocess entrypoint ─────────────────────────────────

def run_medverse_scheduler_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    gpu_id: int,
    tp_rank: int,
    pp_rank: int,
    dp_rank: Optional[int],
    pipe_writer,
) -> None:
    """Drop-in replacement for run_scheduler_process using MedVerseScheduler."""
    import faulthandler
    import os
    import setproctitle

    prefix = ""
    if dp_rank is not None:
        prefix += f" DP{dp_rank}"
    if server_args.tp_size > 1:
        prefix += f" TP{tp_rank}"
    if server_args.pp_size > 1:
        prefix += f" PP{pp_rank}"

    kill_itself_when_parent_died()
    setproctitle.setproctitle(f"medverse::scheduler{prefix.replace(' ', '_')}")
    faulthandler.enable()

    if dp_rank is None and "SGLANG_DP_RANK" in os.environ:
        dp_rank = int(os.environ["SGLANG_DP_RANK"])

    configure_logger(server_args, prefix=prefix)
    suppress_other_loggers()

    try:
        scheduler = MedVerseScheduler(
            server_args, port_args, gpu_id, tp_rank, pp_rank, dp_rank
        )
        pipe_writer.send(
            {
                "status": "ready",
                "max_total_num_tokens": scheduler.max_total_num_tokens,
                "max_req_input_len": scheduler.max_req_input_len,
            }
        )

        from sglang.srt.disaggregation.utils import DisaggregationMode

        if scheduler.disaggregation_mode == DisaggregationMode.NULL:
            if server_args.pp_size > 1:
                scheduler.event_loop_pp()
            elif scheduler.enable_overlap:
                scheduler.event_loop_overlap()
            else:
                scheduler.event_loop_normal()
        else:
            scheduler.event_loop_normal()

    except Exception:
        logger.exception("MedVerseScheduler subprocess crashed")
        raise


# ── MedVerse _launch_subprocesses ────────────────────────────────────────────

def _launch_medverse_subprocesses(
    server_args: ServerArgs,
    port_args: Optional[PortArgs] = None,
) -> Tuple[MedVerseTokenizerManager, Dict]:
    """Like Multiverse's _launch_subprocesses but with MedVerse components."""
    import os

    configure_logger(server_args)
    server_args.check_server_args()
    _set_envs_and_config(server_args)

    if port_args is None:
        port_args = PortArgs.init_new(server_args)
        logger.info(f"{server_args=}")

    server_args.model_path, server_args.tokenizer_path = prepare_model_and_tokenizer(
        server_args.model_path, server_args.tokenizer_path
    )

    scheduler_procs = []

    if server_args.dp_size == 1:
        memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=server_args.enable_memory_saver
        )
        scheduler_pipe_readers = []

        nnodes_per_tp_group = max(server_args.nnodes // server_args.pp_size, 1)
        tp_size_per_node = server_args.tp_size // nnodes_per_tp_group
        tp_rank_range = range(
            tp_size_per_node * (server_args.node_rank % nnodes_per_tp_group),
            tp_size_per_node * (server_args.node_rank % nnodes_per_tp_group + 1),
        )
        pp_size_per_node = max(server_args.pp_size // server_args.nnodes, 1)
        pp_rank_range = range(
            pp_size_per_node * (server_args.node_rank // nnodes_per_tp_group),
            pp_size_per_node * (server_args.node_rank // nnodes_per_tp_group + 1),
        )

        for pp_rank in pp_rank_range:
            for tp_rank in tp_rank_range:
                reader, writer = mp.Pipe(duplex=False)
                gpu_id = (
                    server_args.base_gpu_id
                    + ((pp_rank % pp_size_per_node) * tp_size_per_node)
                    + (tp_rank % tp_size_per_node) * server_args.gpu_id_step
                )
                proc = mp.Process(
                    target=run_medverse_scheduler_process,
                    args=(server_args, port_args, gpu_id, tp_rank, pp_rank, None, writer),
                )
                with memory_saver_adapter.configure_subprocess():
                    proc.start()
                scheduler_procs.append(proc)
                scheduler_pipe_readers.append(reader)
    else:
        reader, writer = mp.Pipe(duplex=False)
        scheduler_pipe_readers = [reader]
        proc = mp.Process(
            target=run_data_parallel_controller_process,
            args=(server_args, port_args, writer),
        )
        proc.start()
        scheduler_procs.append(proc)

    if server_args.node_rank >= 1:
        for reader in scheduler_pipe_readers:
            data = reader.recv()
            assert data["status"] == "ready"
        if __import__("os").getenv("SGLANG_BLOCK_NONZERO_RANK_CHILDREN") == "0":
            return None, None
        launch_dummy_health_check_server(server_args.host, server_args.port)
        for proc in scheduler_procs:
            proc.join()
        return None, None

    # Launch detokenizer subprocess
    detoken_proc = mp.Process(
        target=run_detokenizer_process,
        args=(server_args, port_args),
    )
    detoken_proc.start()

    # ── Key substitution: MedVerseTokenizerManager ────────────────────────────
    tokenizer_manager = MedVerseTokenizerManager(server_args, port_args)

    if server_args.chat_template:
        load_chat_template_for_openai_api(
            tokenizer_manager, server_args.chat_template, server_args.model_path
        )
    else:
        guess_chat_template_name_from_model_path(server_args.model_path)

    scheduler_info = {}
    for i, reader in enumerate(scheduler_pipe_readers):
        data = reader.recv()
        assert data["status"] == "ready", f"Scheduler {i} not ready: {data}"
        scheduler_info[f"scheduler_{i}"] = data
        logger.info(f"[MedVerse] Scheduler {i} ready: {data}")

    tokenizer_manager.max_req_input_len = min(
        d["max_req_input_len"] for d in scheduler_info.values()
    )

    return tokenizer_manager, scheduler_info


# ── MedVerseEngine (Python API) ───────────────────────────────────────────────

class MedVerseEngine(EngineBase):
    """MedVerse inference engine – clinical note → structured DAG reasoning.

    Usage:
        engine = MedVerseEngine(model_path="your-medical-model", tp_size=1)
        out = engine.generate(prompt=soap_note)
        print(out["text"])
    """

    def __init__(self, **kwargs):
        if "server_args" in kwargs:
            server_args = kwargs["server_args"]
        else:
            kwargs.setdefault("log_level", "error")
            server_args = ServerArgs(**kwargs)

        atexit.register(self.shutdown)

        port_args = PortArgs.init_new(server_args)
        logger.info(f"{server_args=}")

        tokenizer_manager, scheduler_info = _launch_medverse_subprocesses(
            server_args=server_args,
            port_args=port_args,
        )
        self.server_args = server_args
        self.tokenizer_manager = tokenizer_manager
        self.scheduler_info = scheduler_info

        context = zmq.Context(2)
        self.send_to_rpc = get_zmq_socket(
            context, zmq.DEALER, port_args.rpc_ipc_name, True
        )

    def generate(
        self,
        prompt: Optional[Union[List[str], str]] = None,
        sampling_params: Optional[Union[List[Dict], Dict]] = None,
        input_ids: Optional[Union[List[List[int]], List[int]]] = None,
        stream: bool = False,
    ) -> Union[Dict, Iterator[Dict]]:
        import asyncio

        obj = GenerateReqInput(
            text=prompt,
            input_ids=input_ids,
            sampling_params=sampling_params or {},
            stream=stream,
        )

        async def _collect():
            results = []
            async for r in self.tokenizer_manager.generate_request(obj, None):
                results.append(r)
            return results[-1] if results else {}

        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(_collect())
        finally:
            loop.close()

    def shutdown(self):
        pass
