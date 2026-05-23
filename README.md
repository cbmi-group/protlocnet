# ProtLocNet: Morphology-Aware Self-Supervised Representation Learning of Protein Localization in Single Cells

This repository is the official implementation of ProtLocNet: Morphology-Aware Self-Supervised Representation Learning of Protein Localization in Single Cells.

## Installation

The training and evaluation code requires PyTorch 2.0 and [xFormers](https://github.com/facebookresearch/xformers) 0.0.18 as well as a number of other 3rd party packages. Note that the code has only been tested with the specified versions and also expects a Linux environment. To setup all the required dependencies for training and evaluation, please follow the instructions below:

*[conda](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html)* **(Recommended)** - Clone the repository and then create and activate a `dinov2` conda environment using the provided environment definition:

```shell
conda env create -f conda.yaml
conda activate dinov2
```

*[pip](https://pip.pypa.io/en/stable/getting-started/)* - Clone the repository and then use the provided `requirements.txt` to install the dependencies:

```shell
pip install -r requirements.txt
```

For dense tasks (depth estimation and semantic segmentation), there are additional dependencies (specific versions of `mmcv` and `mmsegmentation`) which are captured in the `extras` dependency specifications:

*[conda](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html)* **(Recommended)**:

```shell
conda env create -f conda-extras.yaml
conda activate dinov2-extras
```

*[pip](https://pip.pypa.io/en/stable/getting-started/)*:

```shell
pip install -r requirements.txt -r requirements-extras.txt
```

## Data preparation

The processed data are available at [Zenodo](https://zenodo.org/record/20352173). Please download the data and extract it to a location of your choice. The extracted data should have the following structure:

```
HPACustom /
├── train /
│   ├── ENSG00000005007 /
│   │   ├── 7bb40e07-eada-482e-b530-e3c803a36795.png
│   │   ├── ...
│   ├── ENSG00000171109 /
│   │   ├── a1a65aa4-7e92-4d49-abe0-19f3678db1a0.png
│   │   ├── ...
├── test /
│   ├── ENSG00000005007 /
│   │   ├── 7bb40e07-eada-482e-b530-e3c803a36795.png
│   │   ├── ...
│   ├── ENSG00000171109 /
│   │   ├── a1a65aa4-7e92-49-abe0-19f3678db1a0.png
│   │   ├── ...
```

## Training

Run ProtLocNet training on a 4 4090-24GB GPUs with torchrun for 100 epochs:

```shell
torchrun --nproc_per_node=4 dinov2/train/train.py \
    --config-file dinov2/configs/prot/protl.yaml \
    --output-dir <PATH/TO/OUTPUT/DIR> \
    opts train.dataset_path=prot:root=<PATH/TO/DATASET>:split=train
```

## Evaluation

Run ProtLocNet protein identification evaluation on a single node with 4 4090-24GB GPUs with torchrun:

```shell
torchrun --nproc_per_node=4 dinov2/eval/linear.py \
    --config-file dinov2/configs/prot/protl.yaml \
    --train-dataset prot:root=<PATH/TO/DATASET>:split=train \
    --val-dataset prot:root=<PATH/TO/DATASET>:split=test \
    --output-dir <PATH/TO/OUTPUT/DIR> \
    --val-metric-type confusion_matrix \
    --pretrained-weights <PATH/TO/CHECKPOINT>/teacher_checkpoint.pth
    --batch-size 64
```

Run ProtLocNet subcellular localization evaluation on a single node with 4 4090-24GB GPUs with torchrun:

```shell
torchrun --nproc_per_node=4 dinov2/eval/linear.py \
    --config-file dinov2/configs/prot/protl.yaml \
    --train-dataset prot:root=<PATH/TO/DATASET>:split=train:mode=PROTEIN_LOCALIZATION \
    --val-dataset prot:root=<PATH/TO/DATASET>:split=test:mode=PROTEIN_LOCALIZATION \
    --output-dir <PATH/TO/OUTPUT/DIR> \
    --val-metric-type multilabel_confusion_matrix \
    --pretrained-weights <PATH/TO/CHECKPOINT>/teacher_checkpoint.pth
    --batch-size 64 --multilabel
```