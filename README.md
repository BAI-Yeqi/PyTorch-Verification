# PyTorch-Verification

[![license](https://img.shields.io/github/license/BAI-Yeqi/PyTorch-Verification.svg)](https://github.com/BAI-Yeqi/PyTorch-Verification/blob/master/LICENSE)
[![open issues](https://isitmaintained.com/badge/open/BAI-Yeqi/PyTorch-Verification.svg)](https://github.com/BAI-Yeqi/PyTorch-Verification/issues)


This repo implements a set of PyTorch environment checker and cuda-based operators, which helps you verify whether your GPU-based PyTorch is installed properly.

## Run on local env
```
python verify_torch.py
```

## Run on slurm env
```
srun -p ${your_partition} --gres gpu:1 python verify_torch.py

# or

srun -p ${your_partition} --gres gpu:1 --cpus-per-task=10 \
  python verify_torch.py
```
