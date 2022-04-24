# PyTorch-Verification

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
