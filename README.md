# GLAM: Global-Local Variation Awareness in Mamba-based World Model 



## 1. Prerequisites

- Linux (recommended)
- Conda (Miniconda or Anaconda)
- NVIDIA GPU + CUDA driver (recommended for training)

## 2. Create Conda Environment

```bash
conda create -n glam python=3.10 -y
conda activate glam
```

## 3. Install PyTorch

Choose one command based on your CUDA version.

CUDA 11.8:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```


## 4. Install GLAM Dependencies

Run from the project root:

```bash
pip install -r requirements.txt
```

If `mamba-ssm` installation fails, first confirm that your PyTorch/CUDA versions match, then retry.


## 5. Quick Start

Training example:

```bash
python train.py \
  -suite atari \
  -env_name RoadRunner \
  -seed 1 \
  -base_model Glam \
  -version 1_1 \
  -config_path config_files/Glam.yaml \
  -cuda_device 0
```

## 6. Open-Source Notes

- The current codebase uses both `gymnasium` and `gym`; for compatibility, keep both installed.
- If you need a strict lock file for reproducibility, export one separately:

```bash
pip freeze > requirements-lock.txt
```