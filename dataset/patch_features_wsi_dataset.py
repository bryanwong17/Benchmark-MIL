import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import numpy as np
from pathlib import Path

class PatchFeaturesWSIDataset(Dataset):
    def __init__(self, dataset_name, magnification, patch_size, data_type, 
                 class_names_list=["HP", "IP", "LP", "SSL", "TA", "TSA", "TVA", "VA"], feature_extractor_name="resnet50-tr-supervised-imagenet1k"):
        super().__init__()
       
        self.class_names_list = class_names_list

        self.data_type_path = Path(f"/{dataset_name}/TissueFeatures/{feature_extractor_name}/{magnification}/{patch_size}/{data_type}")
        
        self.data = self.get_data_type_list()
        self.label_mapping = self.create_label_mapping()
    
    def create_label_mapping(self):
        # Create a mapping where "TVA" and "VA" both map to label 6
        mapping = {}
        for i, class_name in enumerate(self.class_names_list):
            if class_name in ["TVA", "VA"]:
                mapping[class_name] = 6
            else:
                mapping[class_name] = i
        return mapping

    def get_data_type_list(self):
        data = []
        for class_dir in self.data_type_path.iterdir():
            for slide_dir in class_dir.iterdir():
                data.append((class_dir.name, slide_dir.name))
        return data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        class_name, slide_name = self.data[idx]
        slide_info = torch.load(Path(self.data_type_path / class_name / slide_name / "tissue_0.pt"))

        slide_features = slide_info["features"]
        slide_caption = slide_info["caption"]
        slide_label = self.label_mapping[class_name]

        return slide_features, slide_caption, slide_label

    def get_weights_of_class(self):
        class_names = [class_name for class_name, _, _ in self.dataset_type_files]
        labels = [self.label_mapping[class_name] for class_name in class_names]  # convert to labels
        unique, counts = np.unique(labels, return_counts=True)
        label_cnt = list(zip(unique, counts))
        label_cnt.sort(key=lambda x: x[0])
        weight_arr = np.array([x[1] for x in label_cnt], dtype=float)
        weight_arr = np.max(weight_arr) / weight_arr
        return torch.from_numpy(weight_arr.astype(np.float32))

class PatchFeaturesWSIDataModule(pl.LightningDataModule):
    def __init__(self, dataset_name, magnification, patch_size, 
                 class_names_list=["HP", "IP", "LP", "SSL", "TA", "TSA", "TVA", "VA"], feature_extractor_name="resnet50-tr-supervised-imagenet1k", num_workers=4):
        super().__init__()
        self.dataset_name = dataset_name
        self.magnification = magnification
        self.patch_size = patch_size
        self.class_names_list = class_names_list
        self.feature_extractor_name = feature_extractor_name
        self.num_workers = num_workers

        self.dataset_train = None
        self.dataset_val = None
        self.dataset_test = None

    def setup(self, stage=None):
        if self.dataset_train is None:
            self.dataset_train = PatchFeaturesWSIDataset(self.dataset_name, self.magnification, self.patch_size,  "train", self.class_names_list, self.feature_extractor_name)
            self.dataset_val = PatchFeaturesWSIDataset(self.dataset_name, self.magnification, self.patch_size, "val", self.class_names_list, self.feature_extractor_name)
            self.dataset_test = PatchFeaturesWSIDataset(self.dataset_name, self.magnification, self.patch_size, "test", self.class_names_list, self.feature_extractor_name)

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=1, shuffle=True, num_workers=self.num_workers, drop_last=False, pin_memory=False)
    
    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=1, shuffle=False, num_workers=self.num_workers, drop_last=False, pin_memory=False)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=1, shuffle=False, num_workers=self.num_workers, drop_last=False, pin_memory=False)
