#!/bin/bash

# NCCL_SOCKET_IFNAME=bond0 \
# NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=bond0 \
# NVSHMEM_DISABLE_CUDA_VMM=0 \
# NVSHMEM_ENABLE_NIC_PE_MAPPING=1 \
# NVSHMEM_HCA_LIST=mlx5_0:1,mlx5_3:1,mlx5_4:1,mlx5_7:1 \
# bash ./scripts/launch.sh python/triton_dist/test/nvidia/test_gemm_rs.py -M 8192 -N 8192 -K 29568 --check \
#     2>&1 | tee ./logs/tests/test_gemm_rs.log

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

export EXP_NAME="test_gemm_rs"
mkdir -p logs/${EXP_NAME}

WORLD_SIZE=1
WORLD_SIZE=2
# WORLD_SIZE=3
WORLD_SIZE=8
# WORLD_SIZE=16
# WORLD_SIZE=32

export PLATFORM=H20
PARTITION=debug
PARTITION=long
# NODES="bjdb-h20-node-020"

# # If this shell was left from an expired salloc/sbatch session, srun will try to
# # attach to that stale allocation instead of creating a new one.
# if [ -n "${SLURM_JOB_ID:-}" ] && command -v scontrol >/dev/null 2>&1; then
#     if ! scontrol show job "$SLURM_JOB_ID" >/dev/null 2>&1; then
#         echo "Warning: stale SLURM_JOB_ID=${SLURM_JOB_ID}; unsetting Slurm allocation environment before srun."
#         unset SLURM_JOB_ID SLURM_JOBID SLURM_STEP_ID SLURM_STEPID
#     fi
# fi

if [ $WORLD_SIZE -le 8 ]; then
    NNODES=1
    # NPROC_PER_NODE=$WORLD_SIZE
    NGPUS_PER_NODE=$WORLD_SIZE
else
    NNODES=$((WORLD_SIZE/8))
    # NPROC_PER_NODE=8
    NGPUS_PER_NODE=8
fi
NPROC_PER_NODE=1
# NGPUS_PER_NODE=$NPROC_PER_NODE
# export MASTER_ADDR="localhost"
export MASTER_PORT=12581

# Envs
export CLUSTER_INFO="QC"
# export FLA_USE_TMA=1
# export TB_DIR=./logs/tb/${EXP_NAME}_w${WORLD_SIZE}_${TIMESTAMP}
# mkdir -p $TB_DIR
# export CUDA_DEVICE_MAX_CONNECTIONS=1    # Important for CC Overlap
# export CUDA_LAUNCH_BLOCKING=1   # [DEBUG]
# Specific settings on Fit
# export NVSHMEM_HCA_LIST=^mlx5_2
# export NVSHMEM_HCA_LIST=mlx5_0,mlx5_3,mlx5_4,mlx5_7
export NVSHMEM_HCA_LIST=mlx5_0:1,mlx5_3:1,mlx5_4:1,mlx5_7:1
export NCCL_IB_HCA=mlx5_0,mlx5_3,mlx5_4,mlx5_7
CPUS=${CPUS:-${SLURM_CPUS_ON_NODE:-200}}   # debug/long allocations currently cap this at 200
CPU_PER_TASK=$((CPUS / NPROC_PER_NODE ))   # [NOTE]
# # Network/NCCL Args
# export NCCL_ALGO=Ring   # for allreduce barrier
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG=WARN
# # export NCCL_DEBUG=ERROR
# # export NCCL_NET_GDR_LEVEL=5 # the same as 3
# # export NCCL_NET_GDR_LEVEL=0   # Disable GDR
# export NCCL_NET_GDR_LEVEL=1 # Best performance: NCCL_NET_GDR_LEVEL=1/2(default)
# export NCCL_IB_DISABLE=0
# export NCCL_DEBUG_SUBSYS=NET
# export NCCL_NET=IB
# Network/NCCL/NVSHMEM Args
export NCCL_SOCKET_IFNAME=bond0
export NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=bond0
export NVSHMEM_DISABLE_CUDA_VMM=0
# export NVSHMEM_ENABLE_NIC_PE_MAPPING=1    # Useless
# PROFILING Args
#   Nsight
NSIGHT_CMD=""
USE_NSIGHT=True
USE_NSIGHT=False
if [ $USE_NSIGHT == "True" ]; then
    NSYS_DIR=./logs/nsys
    mkdir -p $NSYS_DIR
    # NSIGHT_CMD="nsys profile --mpi-impl=openmpi -o ${NSYS_DIR}/${EXP_NAME}_w${WORLD_SIZE}_${TIMESTAMP}"
    NSIGHT_CMD="nsys profile --trace=cuda,nvtx,osrt,mpi,nvtx --mpi-impl=openmpi -o ${NSYS_DIR}/${EXP_NAME}_w${WORLD_SIZE}_${TIMESTAMP}"
    NSIGHT_CMD=""
fi
# End

SRUN_SCRIPT="srun \
    --partition=${PARTITION} \
    --nodes=${NNODES} \
    --ntasks-per-node=${NPROC_PER_NODE} \
    --gpus-per-node=${NGPUS_PER_NODE} \
    --exclusive \
"
if [ ! -z $NODES ]; then
    SRUN_SCRIPT+=" -w ${NODES} "
fi
if [ ! -z "$CPU_PER_TASK" ]; then
    SRUN_SCRIPT="$SRUN_SCRIPT \
        --cpus-per-task=$CPU_PER_TASK \
        --cpu-bind=none \
    "
fi
# $SRUN_SCRIPT nvidia-smi topo -m # NV18=450GB/s

# # Task Spec
# MODE='check'
# MODE='perf'
# # End
#     "LLaMA-3.1-70B": {"N": 28672, "K": 8192}, # gate_prog [8192, 28672], up_prog [8192, 28672], down_prog: [28672, 8192]
#     "LLaMA-3.1-70B": -M 8192 -N 8192 -K 28672
set -x
${NSIGHT_CMD} \
$SRUN_SCRIPT \
bash ./scripts/launch.sh python/triton_dist/test/nvidia/test_gemm_rs.py \
    -M 8192 -N 8192 -K 28672 \
    2>&1 | tee logs/${EXP_NAME}/output_${TIMESTAMP}.log
# --check
# hostname \
# ./scripts/executor_qc.sh \

set +x
