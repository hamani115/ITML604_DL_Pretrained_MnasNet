# Hayrat GPU Setup for MNASNet on Imagenette (ITML604 Assignment)

This README shows how to set up and run the MNASNet python scripts on the Hayrat cluster for ITML604 Deep Learning course assignment work (presented [here](https://docs.google.com/presentation/d/12dyahcTCVioS2vrKSC-EHHiw-9xrFbiuzYIDqyQqXQE/edit?usp=sharing))
## Python scripts
There is two main version of MNASNet python script that was used to get the results presented in the slides ([here](https://docs.google.com/presentation/d/12dyahcTCVioS2vrKSC-EHHiw-9xrFbiuzYIDqyQqXQE/edit?usp=sharing)).
### Simple split version:
filename:
```bash
mnasnet_imagenette_simple_split.py
```

Executed with following configuration:
```bash
python3 mnasnet_imagenette_simple_split.py \
    --data_root $DATA_ROOT \
    --out_dir runs_mnasnet_T4_test1 \
    --img_size 224 \
    --epochs 8 \
    --batch_size 128 \
    --lr 3e-4
```

### K-fold version:
filename:
```bash
mnasnet_imagenette_kfold.py
```

Executed with following configuration:
```bash
python3 mnasnet_imagenette_kfold.py \
    --data_root $DATA_ROOT \
    --out_dir runs_mnasnet_kfold_T4_test1 \
    --img_size 224 \
    --epochs 8 \
    --batch_size 128 \
    --lr 3e-4 \
    --kfold 5
```

# Step by Step Guide
## 1) Go to your working directory

```bash
cd /data/datasets/$USER/
```

## 2) Clone your repository

```bash
git clone https://github.com/hamani115/ITML604_DL_Pretrained_MnasNet.git
cd ITML604_DL_Pretrained_MnasNet
```

## 3) Allocate a Slurm job

Start an allocation in the background using `screen`:

```bash
screen -d -m salloc --time=12:00:00 --partition=compute --nodes=1 --ntasks=1
# or
# screen -d -m salloc --time=48:00:00 --partition=gpu --nodes=1 --ntasks=1
```

Check your jobs:

```bash
squeue --me
```

## 4) Open an interactive shell on the allocated node

Replace `JOBID` with your Slurm job ID.

```bash
srun --jobid=$JOBID --interactive --pty --job-name=shell-from-$HOST-$$ $SHELL
```

## 5) Prepare the dataset folder

```bash
mkdir -p content
cd content
```

Download Imagenette (if not downloaded):

```bash
wget -q https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz
tar -xzf imagenette2-160.tgz
```

Set the dataset path:

```bash
export DATA_ROOT="/data/datasets/$USER/ITML604_DL_Pretrained_MnasNet/content/imagenette2-160"
ls -lah $DATA_ROOT
```

You should see:
* `train/`
* `val/`

**Note:** that the code expect the dataset as the following layout:
```bash
$DATA_ROOT/
    train/<class folders...>
    val/<class folders...>
```

## 6) Create a Python environment

Recommended: Python 3.11 venv.

```bash
cd /data/datasets/$USER/ITML604_DL_Pretrained_MnasNet
/usr/bin/python3.11 -m venv dl_env_py311
source dl_env_py311/bin/activate
python --version
```

Upgrade pip tools:

```bash
python -m pip install --upgrade pip setuptools wheel
```

## 7) Install Python requirements

Install the PyTorch CUDA manually:
```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Install rest packages from `requirements.txt` file:

```bash
python -m pip install -r requirements.txt
```

**If did not work**, install the main packages manually:

```bash
python -m pip install numpy matplotlib pillow scikit-learn
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

## 8) Redirect PyTorch cache to scratch/storage

This avoids filling your home directory when pretrained weights are downloaded.

```bash
export TORCH_HOME=/data/datasets/$USER/torch_cache
export XDG_CACHE_HOME=/data/datasets/$USER/.cache
mkdir -p "$TORCH_HOME/hub/checkpoints"
mkdir -p "$XDG_CACHE_HOME"
```

## 9) Run one of the Python scripts

Set the dataset path first:

```bash
export DATA_ROOT="/data/datasets/$USER/ITML604_DL_Pretrained_MnasNet/content/imagenette2-160"
```

### K-fold version

```bash
python3 mnasnet_imagenette_kfold.py \
    --data_root $DATA_ROOT \
    --out_dir runs_mnasnet_kfold_T4_test1 \
    --img_size 224 \
    --epochs 8 \
    --batch_size 128 \
    --lr 3e-4 \
    --kfold 5
```

### Simple split version

```bash
python3 mnasnet_imagenette_simple_split.py \
    --data_root $DATA_ROOT \
    --out_dir runs_mnasnet_T4_test1 \
    --img_size 224 \
    --epochs 8 \
    --batch_size 128 \
    --lr 3e-4
```
---
### Or run `run.sh`

The repository already contains a `run.sh` which has python scripts, make sure it activates the environment first:

```bash
source /data/datasets/$USER/ITML604_DL_Pretrained_MnasNet/dl_env_py311/bin/activate
```

Then run:

```bash
bash run.sh
```

## 10) Useful checks

Check Python and CUDA visibility:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
```

Check GPU status:

```bash
nvidia-smi
```

## 11) Output files

Your run directory will usually contain:

* `best.pt`
* `best_metrics.json`
* `summary.json`
* `curve_loss.png`
* `curve_top1.png`
* `curve_f1.png`
* `confusion_matrix_val.png`
* `roc_val.png`

## Notes

* `simple_split` uses the predefined Imagenette `train/` and `val/` folders.
* `kfold` uses stratified K-fold cross-validation on the training set.
* If you change the repository folder name, update the paths accordingly.
