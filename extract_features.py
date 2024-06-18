import numpy as np
import pandas as pd
import sys
import tqdm
import timm
import hydra
import time
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
from omegaconf import DictConfig

from torchvision import models
from torch.utils.data import DataLoader

from dataset import get_class_names
from dataset.extract_patch_wsi_dataset import ExtractFeaturesWSIDataset

def collate_fn(batch):
    item = batch[0]
    return item

def get_feature_extractor(name, pretrained_path=None):
    if name.startswith("densenet201"):
        model = models.densenet201(weights='DEFAULT')
        model.classifier = nn.Linear(model.classifier.in_features, 3)
        if "lossdiff" in name:
            model.load_state_dict(torch.load(pretrained_path))
        elif "kalm" in name:
            model.load_state_dict(torch.load(pretrained_path))["model_state_dict"]
        feature_extractor = nn.Sequential(*list(model.children())[:-1])
        num_feats = 1920
    
    elif name.startswith("resnet50-tr"):

        if "supervised-imagenet1k" in name:
            feature_extractor = timm.create_model('resnet50', pretrained=True)

        feature_extractor.layer4 = nn.Identity()
        feature_extractor.fc = nn.Identity()
        num_feats = 1024
    
    else:
        raise NotImplementedError
    return feature_extractor, num_feats
    

def split_tensor(data, batch_size):
    num_chk = int(np.ceil(data.shape[0] / batch_size))
    return torch.chunk(data, num_chk, dim=0)

def feature_extractor_forward(data, feature_extractor, batch_size):
    feats = []
    with torch.no_grad():
        for data_i in split_tensor(data, batch_size):
            ft = feature_extractor(data_i)
            feats.append(ft.cpu()) # Move to CPU immediately to save GPU memory
            torch.cuda.empty_cache() # Clear GPU cache
    feats = torch.cat(feats, dim=0)
    return feats

@hydra.main(
    version_base="1.2.0", config_path="config", config_name="default"
)
def main(cfg: DictConfig):
    start_time = time.time()

    features_dir = Path(cfg.dataset.root, cfg.dataset.name, "feature", cfg.feature_extractor.name)
    features_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda:" + str(cfg.gpu_id) if torch.cuda.is_available() else "cpu")
    print(f"Use {device} for feature extraction")

    dataset_csv_path = Path(f"dataset_csv/{cfg.dataset.name}.csv")
    dataset_csv_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_df = pd.read_csv(dataset_csv_path)
    class_names_list = get_class_names(cfg.dataset.name)
    print(f"{len(dataset_df)} slides found consisting of {len(class_names_list)} classes")

    extract_patch_dataset = ExtractFeaturesWSIDataset(cfg.dataset.root, cfg.dataset.name, dataset_df, cfg.dataset.mean, cfg.dataset.std, class_names_list)
    loader = DataLoader(extract_patch_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    feature_extractor, num_feats = get_feature_extractor(cfg.feature_extractor.name, cfg.feature_extractor.pretrained_path)
    feature_extractor = feature_extractor.to(device)
    print("Model is loaded successfully")

    feature_extractor.eval()

    with tqdm.tqdm(
        loader,
        desc="Slide Encoding",
        unit="slide",
        ncols=80,
        position=0,
        leave=True,
        file=sys.stderr
    ) as t1:

        for i, batch in enumerate(t1):
            data_type, class_name, slide_id, patches, caption = batch
    
            patches = patches.squeeze(0).to(device) # remove batch size(1)

            feats = feature_extractor_forward(patches, feature_extractor, cfg.batch_size)
            feats = feats.view(-1, num_feats)

            saved_path = Path(features_dir, data_type, str(class_name))
            saved_path.mkdir(parents=True, exist_ok=True)

            data_to_save = {"features": feats, "caption": caption}

            torch.save(data_to_save, Path(f"{saved_path}/{slide_id}.pt"))

            del patches, feats, data_to_save
            torch.cuda.empty_cache()
            gc.collect() # Ensure memory is released
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Extracting features in {cfg.dataset.name} using {cfg.feature_extractor.name} completed in {total_time // 3600:.0f} hours, {(total_time % 3600) // 60:.0f} minutes, and {total_time % 60:.0f} seconds")

if __name__ == "__main__":

    main()