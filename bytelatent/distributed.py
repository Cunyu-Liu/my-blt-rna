# Copyright (c) Meta Platforms, Inc. and affiliates.
import atexit
import contextlib
import logging
import multiprocessing as mp
import os
import random
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
from functools import lru_cache, partial, reduce
from itertools import chain
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

# for no recompute ops
import xformers.ops
from pydantic import BaseModel, ConfigDict
from torch import distributed as dist
from torch.distributed import ReduceOp
from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed._tensor import DTensor
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import (
    CheckpointPolicy,
    create_selective_checkpoint_contexts,
)

from bytelatent.float8 import convert_linears_to_fp8

logger = logging.getLogger()

# for selective AC
default_no_recompute_ops = {
    torch.ops.aten.mm.default,
    torch.ops.aten._scaled_mm.default,
    torch.ops.aten._scaled_dot_product_efficient_attention.default,
    torch.ops.aten._scaled_dot_product_flash_attention.default,
    torch.ops.c10d_functional.reduce_scatter_tensor.default,
    torch.ops.xformers_flash.flash_fwd.default,
}

if int(os.environ.get("BLT_ALLOW_MISSING_FLEX_ATTENTION", False)) == 0:
    default_no_recompute_ops.add(
        torch.ops.xformers.efficient_attention_forward_cutlass.default
    )


class DistributedArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dp_shard: int = (
        1  # In how many shard to split the model weight. Typically number gpu in a node.
    )
    dp_replicate: int = (
        1  # How many times to replicate the model weight. Typically number of nodes.
    )
    tp_size: int = 1
    selective_activation_checkpointing: bool = False
    compile: bool = False
    fsdp_type: str = "no_shard"
    model_dtype: str = "bf16"
    float8_recipe: str | None = None
    float8_filter: str = r"layers\.[0-9]+\."

    matmul_allow_tf32: bool = False
    allow_bf16_reduced_precision_reduction: bool = True
    detect_anomaly: bool = False

    compile_cache_size_limit: int = 8

    spawn_method: str = "forkserver"

    def configure_world(self):
        pass
        if self.dp_replicate * self.dp_shard * self.tp_size != get_world_size():
            logging.info("Modifying TrainArgs distributed config")
            assert get_world_size() % self.dp_shard == 0
            logging.info("World size: %s", get_world_size())
            logging.info(
                "Existing setting: train_args.distributed.dp_shard=%s",
                self.dp_shard,
            )
            logging.info(
                "Setting train_args.distributed.dp_replicate=%s, was dp_replicate=%s",
                get_world_size() // self.dp_shard,
                self.dp_replicate,
            )
            self.dp_replicate = get_world_size() // self.dp_shard

            logging.info(
                "Changing dp_replicate from %s to %s, to account for tp_size=%s",
                self.dp_replicate,
                self.dp_replicate // self.tp_size,
                self.tp_size,
            )
            assert self.dp_replicate % self.tp_size == 0
            self.dp_replicate = self.dp_replicate // self.tp_size

            logger.warning(
                f"Setting Data Parallel size to {self.dp_replicate * self.dp_shard}"
            )
            assert self.dp_replicate * self.dp_shard * self.tp_size == get_world_size()

            if self.fsdp_type == "no_shard":
                assert self.dp_shard == 1 and self.dp_replicate == get_world_size()


class EnvironmentArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # Use GNU openMP (GOMP) instead of Intel OpenMP [Intel Math Kernel Library (MKL)]
    MKL_SERVICE_FORCE_INTEL: str = "GNU"
    OMP_NUM_THREADS: str = "1"
    MKL_NUM_THREADS: str = "1"
    # faster intra-node collectives, seems to be a cluster specific flag
    ENABLE_INTRA_NODE_COMM: str = "1"
    # avoids OOMs with long context
    TORCH_NCCL_AVOID_RECORD_STREAMS: str = "1"
    # increasing NCCL timeout time before having some NCCL error 22 should give a 16s timeout
    NCCL_IB_TIMEOUT: str = "22"
    NCCL_DEBUG: str = "INFO"
    TORCH_NCCL_ASYNC_ERROR_HANDLING: str = "1"


def get_device_mesh(distributed_args: DistributedArgs):
    tp_size = distributed_args.tp_size
    dp_replicate = distributed_args.dp_replicate
    dp_shard = distributed_args.dp_shard

    assert (
        dp_replicate * dp_shard * tp_size == get_world_size()
    ), f"dp_replicate * dp_shard * tp_size ({dp_replicate} * {dp_shard} * {tp_size}) != world_size ({get_world_size()})"

    dims = []
    names = []
    if dp_replicate >= 1:
        dims.append(dp_replicate)
        names.append("dp_replicate")
    if dp_shard > 1 or distributed_args.fsdp_type == "no_shard":
        dims.append(dp_shard)
        names.append("dp_shard")
    if tp_size > 1:
        dims.append(tp_size)
        names.append("tp")
    dims = tuple(dims)
    names = tuple(names)

    return init_device_mesh("cuda", mesh_shape=dims, mesh_dim_names=names)


def dist_max(x: Union[int, float], mesh: DeviceMesh = None):
    tensor = torch.tensor(x).cuda()
    dist.all_reduce(tensor, op=ReduceOp.MAX, group=mesh.get_group() if mesh else None)
    return tensor


def dist_min(x: Union[int, float], mesh: DeviceMesh = None):
    tensor = torch.tensor(x).cuda()
    dist.all_reduce(tensor, op=ReduceOp.MIN, group=mesh.get_group() if mesh else None)
    return tensor


def dist_sum(
    x: Union[int, float], mesh: DeviceMesh = None, reduce_dtype: torch.dtype = None
):
    tensor = torch.tensor(x).cuda()
    if reduce_dtype is not None:
        tensor = tensor.to(reduce_dtype)
    dist.all_reduce(tensor, op=ReduceOp.SUM, group=mesh.get_group() if mesh else None)
    return tensor


def dist_mean(x: Union[int, float], mesh: DeviceMesh = None):
    tensor = torch.tensor(x).cuda()
    dist.all_reduce(tensor, op=ReduceOp.AVG, group=mesh.get_group() if mesh else None)
    return tensor


def dist_mean_dict(x):
    r = dict()
    for k in x:
        r[k] = dist_mean(x[k])
        r[k] = r[k].item() if (r[k].dim() == 0) else r[k].tolist()
    return r


def to_py_num(num: int | float | torch.Tensor | np.ndarray) -> int | float:
    if isinstance(num, (torch.Tensor, np.ndarray)):
        return num.item()
    else:
        return num


@lru_cache()
def get_is_torch_run() -> bool:
    return os.environ.get("LOCAL_RANK") is not None


@lru_cache()
def get_is_slurm_job() -> bool:
    return "SLURM_JOB_ID" in os.environ and not get_is_torch_run()


@lru_cache()
def get_global_rank() -> int:
    if get_is_torch_run():
        return int(os.environ["RANK"])
    elif get_is_slurm_job():
        return int(os.environ["SLURM_PROCID"])
    else:
        return 0


@lru_cache()
def get_local_rank() -> int:
    if get_is_torch_run():
        return int(os.environ["LOCAL_RANK"])
    elif get_is_slurm_job():
        return int(os.environ["SLURM_LOCALID"])
    else:
        return 0


@lru_cache()
def get_world_size() -> int:
    if get_is_torch_run():
        return int(os.environ["WORLD_SIZE"])
    elif get_is_slurm_job():
        return int(os.environ.get("SLURM_NTASKS", "1"))
    else:
        return 1


@lru_cache()
def get_is_master() -> bool:
    return get_global_rank() == 0


@lru_cache()
def get_master_port(job_id: int) -> int:
    if get_is_torch_run():
        return int(os.environ["MASTER_PORT"])
    else:
        MIN_MASTER_PORT, MAX_MASTER_PORT = (20000, 60000)
        rng = random.Random(job_id)
        return rng.randint(MIN_MASTER_PORT, MAX_MASTER_PORT)


@lru_cache()
def get_master_addr() -> str:
    if get_is_torch_run():
        return os.environ["MASTER_ADDR"]
    elif get_is_slurm_job():
        hostnames = subprocess.check_output(
            ["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]]
        )
        return hostnames.split()[0].decode("utf-8")
    else:
        return "127.0.0.1"


def setup_env(env_args: EnvironmentArgs):
    env_vars = env_args.model_dump()

    # When using Triton, it attempts to locate prebuilt kernels in a cache
    # located at ~/.triton/cache, but when that's backed by NFS this can fail
    # with a "OSError: [Errno 116] Stale file handle" error. If we were to set
    # it to a local directory it would belong to the first user who created it
    # and it would fail for the job of any other successive user assigned to
    # that machine. To avoid all this mess we use a temporary per-process cache.
    triton_cache_dir = tempfile.mkdtemp()
    atexit.register(shutil.rmtree, triton_cache_dir, ignore_errors=True)
    env_vars["TRITON_CACHE_DIR"] = triton_cache_dir

    # We change the tmp dir to /scratch in case it's slurm job
    # This avoids filling up the host's usually limited tmpfs
    # A full tmpfs leads to very slow creation of processes and weird bugs
    if get_is_slurm_job():
        new_tmp = f"/scratch/slurm_tmpdir/{os.environ['SLURM_JOB_ID']}"
        if os.path.exists(new_tmp):
            env_vars["TMP_DIR"] = new_tmp

    for name, value in env_vars.items():
        if os.environ.get(name) != str(value):
            os.environ[name] = str(value)
            logger.warning(f"WARNING: Setting {name} to {value}")


def setup_torch_distributed(dist_args: DistributedArgs):
    """
    Handle single and multi-GPU / multi-node / SLURM jobs.
    Initialize the following variables:
        - global_rank
        - world_size
    """
    mp.set_start_method(dist_args.spawn_method, force=True)
    with mp.Manager():
        pass

    local_rank = get_local_rank()

    os.environ["RANK"] = str(get_global_rank())
    os.environ["WORLD_SIZE"] = str(get_world_size())
    os.environ["MASTER_ADDR"] = get_master_addr()
    os.environ["MASTER_PORT"] = str(
        get_master_port(job_id=int(os.environ.get("SLURM_JOB_ID", -1)))
    )

    if get_is_torch_run():
        logger.info(f"Run launched with torchrun, local rank: {local_rank}")
    elif get_is_slurm_job():
        logger.info(f"Run launched with slurm, local rank: {local_rank}")
    else:
        logger.info("Single GPU job")

    logger.info(f"ENV: {os.environ}")

    # set GPU device
    assert 0 <= local_rank < 8
    if dist_args.matmul_allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        logger.warning(
            f"WARNING: Setting torch.backends.matmul.allow_tf32 to True. This is faster but less accurate."
        )
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (
        dist_args.allow_bf16_reduced_precision_reduction
    )
    if torch.cuda.device_count() > 1:
        torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(init_method="env://", backend="nccl")
    torch.autograd.set_detect_anomaly(dist_args.detect_anomaly)


def get_module(module, access_string):
    names = access_string.split(sep=".")
    return reduce(getattr, names, module)


def set_module(module, access_string, value):
    names = access_string.split(sep=".")
    parent = reduce(getattr, names[:-1], module)
    setattr(parent, names[-1], value)


def default_fsdp_grouping_plan(n_layers: int) -> List[Tuple[str, bool]]:
    return [(f"layers.{i}", i < n_layers - 1) for i in range(n_layers)]


def get_default_policy(no_recompute_ops=None):
    no_recompute_ops = no_recompute_ops or default_no_recompute_ops

    def default_policy(ctx, func, *args, **kwargs):
        return (
            CheckpointPolicy.MUST_SAVE
            if func in no_recompute_ops
            else CheckpointPolicy.PREFER_RECOMPUTE
        )

    return default_policy


@torch.no_grad()
def check_model_value_range(
    model: torch.nn.Module, range: float = 1e3, std: float = 1e3
):
    for name, param in chain(model.named_parameters(), model.named_buffers()):
        if isinstance(param, DTensor):
            param = param.to_local()

        if param.numel() == 0:
            logger.warning(
                f"Model parameter {name} is empty, probably because of FSDP sharding"
            )
            continue

        if torch.isnan(param).any() or torch.isinf(param).any():
            logger.warning(f"Model parameter {name} contains NaN or Inf")

        param_range = param.max() - param.min()
        param_std = param.std()
        if param_range > range:
            logger.warning(
                f"Model parameter {name} has a suspiciously large range ({param_range}): please check initialization and init_weights is defined and called"
            )
        if param_std > std:
            logger.warning(
                f"Model parameter {name} has a suspiciously large standard deviation ({param_std}): please check initialization and init_weights is defined and called"
            )
        if (param == 0).all():
            logger.warning(
                f"Model parameter {name} is all zeros: it might be because of a missing initialization"
            )


def init_signal_handler(callable):
    """
    Handle signals sent by SLURM for time limit / pre-emption.
    """
    signal.signal(signal.SIGUSR2, callable)
    logger.warning("Signal handler installed.")


def requeue_slurm_job():
    prod_id = int(os.environ["SLURM_PROCID"])
    logger.warning("Host: %s - Global rank: %i" % (socket.gethostname(), prod_id))
    if prod_id == 0 and os.environ.get("LAUNCH_WITH", "") != "DORA":
        logger.warning("Requeuing job " + os.environ["SLURM_JOB_ID"])
        os.system("scontrol requeue " + os.environ["SLURM_JOB_ID"])
    else:
        logger.warning("Not the master process, no need to requeue.")
    sys.exit(0)


@contextlib.contextmanager
def clean_env():
    distrib_names = (
        "MASTER_ADDR",
        "MASTER_PORT",
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "TORCHELASTIC_RUN_ID",
        "DORA_FORCE_DISTRIB",
    )
    cluster_env = {
        x: os.environ.pop(x)
        for x in os.environ
        if x.startswith(
            ("SLURM_", "SLURMD_", "SRUN_", "SBATCH_", "SUBMITIT_", "WANDB_")
        )
        or x in distrib_names
    }
    try:
        yield
    finally:
        os.environ.update(cluster_env)


def parallelize_model(
    model: torch.nn.Module,
    device_mesh,
    model_args,
    distributed_args: DistributedArgs,
    fsdp_grouping_plan: Optional[List[Tuple[str, bool]]] = None,
    tp_parallelize=None,
    no_recompute_ops=None,
) -> torch.nn.Module:
    if distributed_args.tp_size > 1:
        assert (
            distributed_args.fsdp_type == "full_shard"
        ), "Only full shard is supported for TP parallelism"
        assert tp_parallelize is not None, "TP plan is required for TP parallelism"
        assert (
            distributed_args.compile == False
        ), "Compile is not supported for TP parallelism"

        tp_parallelize(model, device_mesh["tp"], model_args, distributed_args)

    if distributed_args.float8_recipe is not None:
        if distributed_args.tp_size > 1:
            raise RuntimeError("float8 is incompatible with tensor-parallelism for now")
        model = convert_linears_to_fp8(
            model, distributed_args.float8_recipe, distributed_args.float8_filter
        )

    param_dtype = dict(fp32=torch.float32, fp16=torch.float16, bf16=torch.bfloat16)[
        distributed_args.model_dtype
    ]
    if (
        distributed_args.fsdp_type == "full_shard"
        or distributed_args.fsdp_type == "no_shard"
    ):
        if distributed_args.fsdp_type == "no_shard":
            assert (
                distributed_args.dp_shard == 1
            ), "dp_shard must be 1 for no_shard fsdp_type"
            assert (
                device_mesh["dp_shard"].size() == 1
            ), "dp_shard must be 1 for no_shard fsdp_type"

        fsdp_config = dict(
            mp_policy=(
                MixedPrecisionPolicy(
                    param_dtype=param_dtype,
                    reduce_dtype=torch.float32,
                )
            ),
            mesh=(
                device_mesh["dp_replicate", "dp_shard"]
                if distributed_args.dp_shard > 1
                or distributed_args.fsdp_type == "no_shard"
                else device_mesh["dp_replicate"]
            ),
        )

        if fsdp_grouping_plan is None:
            # Assume that the model has list of layers and group around it
            fsdp_grouping_plan = default_fsdp_grouping_plan(len(model.layers))

        for path, reshard_after_forward in fsdp_grouping_plan:
            module = get_module(model, path)
            set_module(
                model,
                path,
                fully_shard(
                    module, **fsdp_config, reshard_after_forward=reshard_after_forward
                ),
            )

        model = fully_shard(model, **fsdp_config, reshard_after_forward=True)
    else:
        raise ValueError(f"Invalid fsdp_type: {distributed_args.fsdp_type}")

    if distributed_args.selective_activation_checkpointing:
        # only works for blt models
        # assuming that entropy models will not use checkpointing
        for module in [
            model.global_transformer,
            model.local_encoder,
            model.local_decoder,
        ]:
            for i in range(len(module.layers)):
                module.layers[i] = checkpoint_wrapper(
                    module.layers[i],
                )

    if distributed_args.compile:
        torch._dynamo.config.cache_size_limit = (
            distributed_args.compile_cache_size_limit
        )
        model = torch.compile(model)

    return model
