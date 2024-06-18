python main.py --seed 2000,1,17 --dataset-root /vast/kaist/WSI_datasets --dataset-name tcga_nsclc --feature-extractor resnet50-tr-supervised-imagenet1k --mil-model ABMIL --epochs 50 --lr 1e-4 --weight-decay 1e-4 --gpu-id 0
python main.py --seed 2000,1,17 --dataset-root /vast/kaist/WSI_datasets --dataset-name tcga_nsclc --feature-extractor resnet50-tr-supervised-imagenet1k --mil-model DSMIL --epochs 50 --lr 1e-4 --weight-decay 1e-4 --gpu-id 0
python main.py --seed 2000,1,17 --dataset-root /vast/kaist/WSI_datasets --dataset-name tcga_nsclc --feature-extractor resnet50-tr-supervised-imagenet1k --mil-model CLAM-SB --epochs 50 --lr 2e-4 --weight-decay 1e-5 --gpu-id 0 
python main.py --seed 2000,1,17 --dataset-root /vast/kaist/WSI_datasets --dataset-name tcga_nsclc --feature-extractor resnet50-tr-supervised-imagenet1k --mil-model CLAM-MB --epochs 50 --lr 2e-4 --weight-decay 1e-5 --gpu-id 0
python main.py --seed 2000,1,17 --dataset-root /vast/kaist/WSI_datasets --dataset-name tcga_nsclc --feature-extractor resnet50-tr-supervised-imagenet1k --mil-model TransMIL --epochs 50 --lr 2e-4 --weight-decay 1e-5 --opt lookahead_radam --gpu-id 0
python main.py --seed 2000,1,17 --dataset-root /vast/kaist/WSI_datasets --dataset-name tcga_nsclc --feature-extractor resnet50-tr-supervised-imagenet1k --mil-model DTFD-MIL --distill AFS --epochs 50 --lr 1e-4 --weight-decay 1e-4 --gpu-id 0
python main.py --seed 2000,1,17 --dataset-root /vast/kaist/WSI_datasets --dataset-name tcga_nsclc --feature-extractor resnet50-tr-supervised-imagenet1k --mil-model DTFD-MIL --distill MaxMinS --epochs 50 --lr 1e-4 --weight-decay 1e-4 --gpu-id 0
python main.py --seed 2000,1,17 --dataset-root /vast/kaist/WSI_datasets --dataset-name tcga_nsclc --feature-extractor resnet50-tr-supervised-imagenet1k --mil-model DTFD-MIL --distill MaxS --epochs 50 --lr 1e-4 --weight-decay 1e-4 --gpu-id 0