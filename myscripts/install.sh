# Build Successed!

# Reference
https://github.com/ByteDance-Seed/Triton-distributed/blob/main/docs/build.md
python -m pip index versions <package_name> # show all versions supported in current environment

# Conda
conda create -n triton_dist_py312 python=3.12  # Use the latest version python
conda activate triton_dist_py312
# Nvcc
spack load cuda@12.8.1
# # Cmake
# spack load cmake@3.31.9
# Pytorch 2.11.0 w/ cuda 12.8 (latest stable version)
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
# Triton-distributed
git clone git@github.com:ByteDance-Seed/Triton-distributed.git && cd Triton-distributed
git checkout -b yhy_dev
git submodule update --init --recursive
# Other dependencies
# pip3 install setuptools==69.0.0 wheel pybind11
python -m pip install setuptools wheel pybind11
#   Extra dependencies  # [NOTE] Added by yhy
# extras_require={
#     "build": ["cmake>=3.20,<4.0", "lit", "ninja", "pybind11"],
#     "tests": [
#         "autopep8",
#         "isort",
#         "numpy",
#         "pytest",
#         "pytest-forked",
#         "pytest-xdist",
#         "scipy>=1.7.1",
#         "llnl-hatchet",
#         "transformers",
#         "tqdm",
#     ] + DEPS_TEST,
#     "tutorials": [
#         "matplotlib",
#         "pandas",
#         "tabulate",
#         "chardet",
#     ],
# },
python -m pip install lit ninja "cmake>=3.20,<4.0" autopep8 isort numpy pytest pytest-forked \
    pytest-xdist scipy>=1.7.1 llnl-hatchet transformers tqdm
# Build Triton-distributed
#   Remove triton installed with torch
python -m pip uninstall triton
python -m pip uninstall triton_dist # remove previous triton-dist
#   Install dependencies
python -m pip install cuda.core==0.2.0 cuda-python==12.4 nvidia-nvshmem-cu12==3.3.9 Cython==0.29.24 nvshmem4py-cu12==0.1.2
# ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
# torch 2.11.0+cu128 requires nvidia-nvshmem-cu12==3.4.5; platform_system == "Linux", but you have nvidia-nvshmem-cu12 3.3.9 which is incompatible.

rm -rf /usr/local/lib/python3.12/dist-packages/triton   # No this folder
#   Install Triton-distributed
cd /workspace/Triton-distributed
export USE_TRITON_DISTRIBUTED_AOT=0
echo 'numpy<2' > ./tmp/pip_install_constraint.txt
salloc -p h01 -N 1 --gres=gpu:1 --cpus-per-task=40
salloc -p h01 -N 1 --cpus-per-task=40
MAX_JOBS=40 python -m pip install -c ./tmp/pip_install_constraint.txt -e python[build,tests,tutorials] \
    --verbose --no-build-isolation --use-pep517 \
    2>&1 | tee ./logs/triton_dist_install.log

# Test NVIDIA Installation
#   Quick Validation Tests
#       Basic distributed wait test. Passed✅
bash ./scripts/launch.sh python/triton_dist/test/nvidia/test_distributed_wait.py --case correctness_tma \
    2>&1 | tee ./logs/tests/test_distributed_wait.log
#       NVSHMEM API test. Passed✅
# NCCL_SOCKET_IFNAME=bond0 NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=bond0 NVSHMEM_DISABLE_CUDA_VMM=0 \
NCCL_SOCKET_IFNAME=bond0 \
NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=bond0 \
NVSHMEM_DISABLE_CUDA_VMM=0 \
NVSHMEM_ENABLE_NIC_PE_MAPPING=1 \
NVSHMEM_HCA_LIST=mlx5_0:1,mlx5_3:1,mlx5_4:1,mlx5_7:1 \
bash ./scripts/launch.sh python/triton_dist/test/nvidia/test_nvshmem_api.py \
    2>&1 | tee ./logs/tests/test_nvshmem_api.log

#   AllGather GEMM Tests. 
#       Passed✅
NCCL_SOCKET_IFNAME=bond0 \
NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=bond0 \
NVSHMEM_DISABLE_CUDA_VMM=0 \
NVSHMEM_ENABLE_NIC_PE_MAPPING=1 \
NVSHMEM_HCA_LIST=mlx5_0:1,mlx5_3:1,mlx5_4:1,mlx5_7:1 \
bash ./scripts/launch.sh python/triton_dist/test/nvidia/test_ag_gemm.py --case check \
    2>&1 | tee ./logs/tests/test_ag_gemm_0.log
NCCL_SOCKET_IFNAME=bond0 \
NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=bond0 \
NVSHMEM_DISABLE_CUDA_VMM=0 \
NVSHMEM_ENABLE_NIC_PE_MAPPING=1 \
NVSHMEM_HCA_LIST=mlx5_0:1,mlx5_3:1,mlx5_4:1,mlx5_7:1 \
bash ./scripts/launch.sh python/triton_dist/test/nvidia/test_ag_gemm.py --case perf \
    2>&1 | tee ./logs/tests/test_ag_gemm_perf_0.log
#       Passed✅
bash ./scripts/launch.sh --nproc_per_node 2 python/triton_dist/test/nvidia/test_ag_gemm.py --case check 2>&1 | tee ./logs/tests/test_ag_gemm_1.log

#   GEMM ReduceScatter Tests
#       Passed✅
NCCL_SOCKET_IFNAME=bond0 \
NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=bond0 \
NVSHMEM_DISABLE_CUDA_VMM=0 \
NVSHMEM_ENABLE_NIC_PE_MAPPING=1 \
NVSHMEM_HCA_LIST=mlx5_0:1,mlx5_3:1,mlx5_4:1,mlx5_7:1 \
bash ./scripts/launch.sh python/triton_dist/test/nvidia/test_gemm_rs.py -M 8192 -N 8192 -K 29568 --check \
    2>&1 | tee ./logs/tests/test_gemm_rs.log

#   AllReduce Tests
#       Passed✅
NCCL_SOCKET_IFNAME=bond0 \
NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=bond0 \
NVSHMEM_DISABLE_CUDA_VMM=1 \
NVSHMEM_ENABLE_NIC_PE_MAPPING=1 \
NVSHMEM_HCA_LIST=mlx5_0:1,mlx5_3:1,mlx5_4:1,mlx5_7:1 \
bash ./scripts/launch.sh python/triton_dist/test/nvidia/test_allreduce.py --method one_shot --stress --iters 2 \
    2>&1 | tee ./logs/tests/test_allreduce.log

#   Flash Decoding Tests
#       Passed✅
NCCL_SOCKET_IFNAME=bond0 \
NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=bond0 \
NVSHMEM_DISABLE_CUDA_VMM=0 \
NVSHMEM_ENABLE_NIC_PE_MAPPING=1 \
NVSHMEM_HCA_LIST=mlx5_0:1,mlx5_3:1,mlx5_4:1,mlx5_7:1 \
bash ./scripts/launch.sh python/triton_dist/test/nvidia/test_decode_attn.py --case perf_8k \
    2>&1 | tee ./logs/tests/test_decode_attn.log
#       Passed✅
NCCL_SOCKET_IFNAME=bond0 \
NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=bond0 \
NVSHMEM_DISABLE_CUDA_VMM=0 \
NVSHMEM_ENABLE_NIC_PE_MAPPING=1 \
NVSHMEM_HCA_LIST=mlx5_0:1,mlx5_3:1,mlx5_4:1,mlx5_7:1 \
bash ./scripts/launch.sh python/triton_dist/test/nvidia/test_sp_decode_attn.py --case correctness \
    2>&1 | tee ./logs/tests/test_sp_decode_attn.log

#   MoE Tests [TODO]
#       Dense model
bash ./scripts/launch.sh python/triton_dist/test/nvidia/test_tp_e2e.py --bsz 8 --seq_len 256 --model <model_path> --check --mode ag_rs \
    2>&1 | tee ./logs/tests/test_tp_e2e.log
#       E2E inference
bash ./scripts/launch.sh python/triton_dist/test/nvidia/test_e2e_inference.py --bsz 4096 --gen_len 128 --max_length 150 --model <model_path> --backend triton_dist \
    2>&1 | tee ./logs/tests/test_e2e_inference.log

# Run Tutorial Tests
NCCL_SOCKET_IFNAME=bond0 \
NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=bond0 \
NVSHMEM_DISABLE_CUDA_VMM=0 \
NVSHMEM_ENABLE_NIC_PE_MAPPING=1 \
NVSHMEM_HCA_LIST=mlx5_0:1,mlx5_3:1,mlx5_4:1,mlx5_7:1 \
bash .codebase/scripts/nvidia/run_tutorial_test.sh \
    2>&1 | tee ./logs/tests/run_tutorial_test.log

# Install tg4perfetto for intra-kernel profiling
# conda install -c conda-forge protobuf
conda install --override-channels -c conda-forge libprotobuf
which protoc
pip install tg4perfetto
