# Hayrat GPU Setup for MNASNet on Imagenette (ITML604 Assignment)

This README shows how to set up and run the MNASNet scripts on the Hayrat cluster.

## 1) Go to your working directory

```bash
cd /data/dataset/$USER/
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
export DATA_ROOT='/data/datasets/$USER/ITML604_DL_Pretrained_MnasNet/content/imagenette2-160'
ls -lah $DATA_ROOT
```

You should see:
* `train/`
* `val/`

**Note:** that the code expect the dataset as the following layout:
```
$DATA_ROOT/
    train/<class folders...>
    val/<class folders...>
```

## 6) Create a Python environment

Recommended: Python 3.11 venv.

```bash
cd /data/dataset/$USER/ITML604_DL_Pretrained_MnasNet
/usr/bin/python3.11 -m venv dl_env_py311
source dl_env_py311/bin/activate
python --version
```

Upgrade pip tools:

```bash
python -m pip install --upgrade pip setuptools wheel
```

## 7) Install Python requirements

If you have a `requirements.txt` file:

```bash
python -m pip install -r requirements.txt
```

If not, install the main packages manually:

```bash
python -m pip install numpy matplotlib pillow scikit-learn
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

## 8) Redirect PyTorch cache to scratch/storage

This avoids filling your home directory when pretrained weights are downloaded.

```bash
export TORCH_HOME=/data/dataset/$USER/torch_cache
export XDG_CACHE_HOME=/data/dataset/$USER/.cache
mkdir -p "$TORCH_HOME/hub/checkpoints"
mkdir -p "$XDG_CACHE_HOME"
```

## 9) Run one of the Python scripts

Set the dataset path first:

```bash
export DATA_ROOT='/data/datasets/$USER/ITML604_DL_Pretrained_MnasNet/content/imagenette2-160'
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

## 10) Or run `run.sh`

If your repository already contains a `run.sh`, make sure it activates the environment first:

```bash
source /data/dataset/$USER/ITML604_DL_Pretrained_MnasNet/dl_env_py311/bin/activate
```

Then run:

```bash
bash run.sh
```

## 11) Useful checks

Check Python and CUDA visibility:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
```

Check GPU status:

```bash
nvidia-smi
```

## 12) Output files

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
