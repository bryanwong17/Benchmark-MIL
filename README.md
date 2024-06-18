<h1 align="center"> Benchmark MIL </h1>

PyTorch implementation for Benchmarking MIL for WSI Classification within a single framework

## Environments
- Linux Ubuntu 20.04.6
- 2 NVIDIA A100 GPUs (40GB each)
- CUDA version: 12.2
- Python version: 3.12.2

## Installation

Install [Anaconda](https://www.anaconda.com/download#)<br>
Create a new environment and activate it

```bash
conda create --name benchmark_mil python==3.12.2
conda activate benchmark_mil
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
python -m pip install opencv-contrib-python pytorch-lightning==2.0.2 torchmetrics==0.11.4 pytorch_optimizer pandas wandb tqdm timm==0.9.16 jpeg4py hydra-core omegaconf future nystrom-attention==0.0.9 git+https://github.com/oval-group/smooth-topk.git
```

## Dataset Preparation

Below is an example of using the TCGA NSCLC (Lung Cancer) dataset provided by [DSMIL](https://github.com/binli123/dsmil-wsi). The dataset can be downloaded from this [link](https://drive.google.com/file/d/17zCn-WRNzxxxh8kkdBTbDLDZy0XZ3RIu/view)

We stratified a 10% sample from the training to create validation while keeping the test untouched. The structure of the dataset after splitting it into data types (e.g., train, val, test) and classes (e.g., LUAD, LUSC) is as follows:

<details>
<summary>
Folder structure
</summary>

```bash
<datasets>/
├── tcga_nsclc/
    ├── patch/
        ├── train/
            ├── LUAD/
                ├── TCGA-4B-A93V-01Z-00-DX1/
                    ├── TCGA-4B-A93V-01Z-00-DX1_x1_y1.jpg
                    ├── TCGA-4B-A93V-01Z-00-DX1_x2_y2.jpg
                ├── ...
            
            ├── LUSC/
                ├── ...

        ├── val/
            ├── ...

        ├── test/
            ├── ...
                
```
</details>

**Note**: If one wants to use another dataset, it is necessary to define the name of the dataset and specify the class names inside `dataset/__init__.py`

## Merge Patches

To reduce the loading time of many patches, we merge X patches (default:200) vertically into a single image and save it in .jpg or .png format. The merging process is performed paralell using the `joblib` package

Example usage:

```bash
python merge_patch.py --config-name --dataset-root /vast/WSI_datasets --dataset-name tcga_nsclc --output-ext .jpg --patch-size 224 --batch-size 200 --parallel-n 10
```

If the dataset contains texts or captions (e.g., PatchGastricADC22), one can specify `--use-caption` to save the captions into the dataset CSV file

The merged patches will be saved under `/vast/WSI_datasets/tcga_nsclc/merge_patch` and a CSV file saved under `dataset_csv/tcga_nsclc.csv`, containing information such as `data_type`, `class_name`, `slide_id`, `len_merge_patches(200)`, and `caption`

## Feature Extraction

Create a config file under `config/`. A good starting point is to follow the example config file provided (e.g., `resnet50-supervised-imagenet1k.yaml`)

<details>
<summary>
Config File
</summary>

```bash
gpu_id: 0

# Dataset settings
dataset:
  root: '/vast/WSI_datasets'
  name: 'tcga_nsclc'
  mean: [0.485, 0.456, 0.406] # ImageNet mean normalization
  std: [0.229, 0.224, 0.225] # ImageNet std normalization
  
# Model settings
feature_extractor:
  name: 'resnet50-tr-supervised-imagenet1k'
  backbone: 'resnet50'
  pretrained_path: 

```
</details>
 
To kick off the feature extraction process using `resnet50-tr-supervised-imagenet1k` features
 
```bash
python extract_features.py --config-name resnet50-tr-supervised-imagenet1k
```

The extracted features will be saved under `/vast/WSI_datasets/tcga_nsclc/feature`

## MIL Aggregator Training

The implemented models so far are:

- Mean pooling
- Max pooling
- [ABMIL, GABMIL](https://github.com/AMLab-Amsterdam/AttentionDeepMIL)
- [DSMIL](https://github.com/binli123/dsmil-wsi)
- [CLAM-SB, CLAM-MB](https://github.com/mahmoodlab/CLAM)
- [TransMIL](https://github.com/szc19990412/TransMIL)
- [DTFD-MIL AFS, DTFD-MIL MaxMinS, DTFD-MIL MaxS](https://github.com/hrzhang1123/DTFD-MIL)

Below are the useful arguments to train MIL models:

```bash
seed                # Seed for reproducing experiments (can be multiple seeds)
dataset_root        # Path to WSI dataset root folders
dataset_name        # The name of dataset
feature_extractor   # Name of the feature extractor used
mil_model           # Which MIL aggregator model to use ['ABMIL', 'DSMIL', 'CLAM-SB', 'CLAM-MB', 'TransMIL', 'DTFD-MIL']
distill             # Used only for DTFD-MIL ['AFS', 'MaxMinS', 'MaxS']
epochs              # Number of epochs to train
lr                  # Initial learning rate
weight_decay        # Weight decay of the optimizer
gpu_id              # Which GPU for training
````

To train MIL models with seed 0 using the previously extracted features (e.g., resnet50-tr-supervised-pretrained-imagenet-1k) is as below (refer to `train.sh`):

```bash
python main.py --seed 2000,1,17 --dataset-root /vast/WSI_datasets --dataset-name tcga_nsclc --feature-extractor resnet50-tr-supervised-imagenet1k --mil-model ABMIL --epochs 50 --lr 1e-4 --weight-decay 1e-4 --gpu-id 0
python main.py --seed 2000,1,17 --dataset-root /vast/WSI_datasets --dataset-name tcga_nsclc --feature-extractor resnet50-tr-supervised-imagenet1k --mil-model DSMIL --epochs 50 --lr 1e-4 --weight-decay 1e-4 --gpu-id 0
python main.py --seed 2000,1,17 --dataset-root /vast/WSI_datasets --dataset-name tcga_nsclc --feature-extractor resnet50-tr-supervised-imagenet1k --mil-model CLAM-SB --epochs 50 --lr 2e-4 --weight-decay 1e-5 --gpu-id 0 
python main.py --seed 2000,1,17 --dataset-root /vast/WSI_datasets --dataset-name tcga_nsclc --feature-extractor resnet50-tr-supervised-imagenet1k --mil-model CLAM-MB --epochs 50 --lr 2e-4 --weight-decay 1e-5 --gpu-id 0
python main.py --seed 2000,1,17 --dataset-root /vast/WSI_datasets --dataset-name tcga_nsclc --feature-extractor resnet50-tr-supervised-imagenet1k --mil-model TransMIL --epochs 50 --lr 2e-4 --weight-decay 1e-5 --opt lookahead_radam --gpu-id 0
python main.py --seed 2000,1,17 --dataset-root /vast/WSI_datasets --dataset-name tcga_nsclc --feature-extractor resnet50-tr-supervised-imagenet1k --mil-model DTFD-MIL --distill AFS --epochs 50 --lr 1e-4 --weight-decay 1e-4 --gpu-id 0
python main.py --seed 2000,1,17 --dataset-root /vast/WSI_datasets --dataset-name tcga_nsclc --feature-extractor resnet50-tr-supervised-imagenet1k --mil-model DTFD-MIL --distill MaxMinS --epochs 50 --lr 1e-4 --weight-decay 1e-4 --gpu-id 0
python main.py --seed 2000,1,17 --dataset-root /vast/WSI_datasets --dataset-name tcga_nsclc --feature-extractor resnet50-tr-supervised-imagenet1k --mil-model DTFD-MIL --distill MaxS --epochs 50 --lr 1e-4 --weight-decay 1e-4 --gpu-id 0
```

## How to Contribute?

If one wants to add more MIL models to this framework, you are welcome to submit `pull requests`. Here are the steps to consider:

1. **Trainer Module**: Check if the Trainer Module follows common MIL models `pl_model/mil_trainer` or differs from the common approach (e.g., DTFD-MIL `pl_model/mil_trainer_dtfdmil`, which has 2 forward functions, 2 losses, and 2 optimizers)
2. **Define the Model Class**: Define the new model class inside the `model/` directory
3. **Initialize the Model and Loss**: Initialize the model and loss in `model/__init__.py`, especially in the `get_mil_model` function.
4. **Define the Model Forward Function**: Define the forward function for the model in `pl_model/forward_fn` and add it to the `get_forward_func` function

## Error or Bugs

If one finds any errors in the code or bugs in the implementation, feel free to submit `issues` or `pull requests` as the code is still under development and will be updated further

## Acknowledgement

Our framework uses [PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning) for its efficiency and simplicity. The code is built on top of these excellent works: [PromptMIL](https://github.com/cvlab-stonybrook/PromptMIL), [ABMIL](https://github.com/AMLab-Amsterdam/AttentionDeepMIL), [DSMIL](https://github.com/binli123/dsmil-wsi), [TransMIL](https://github.com/szc19990412/TransMIL), [DTFD-MIL](https://github.com/hrzhang1123/DTFD-MIL)