export FORCE_CUDA=1
export TCNN_CUDA_ARCHITECTURES=70,72,75,80,86
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
