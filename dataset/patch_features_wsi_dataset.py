import numpy as np
import pandas as pd

import torch
import pytorch_lightning as pl

from pathlib import Path
from torch.utils.data import Dataset, DataLoader

class PatchFeaturesWSIDataset(Dataset):
    def __init__(self, dataset_root, dataset_name, dataset_df, data_type, 
                 class_names_list, feature_extractor_name="resnet50-tr-supervised-imagenet1k",
                 few_shot_samples_per_class=None, seed=None):
        super().__init__()
        self.dataset_root = dataset_root
        self.dataset_name = dataset_name
    
        self.dataset_df = dataset_df
        self.data_type = data_type

        self.class_names_list = class_names_list

        self.feature_extractor_name = feature_extractor_name

        self.few_shot_samples_per_class = few_shot_samples_per_class
        self.seed = seed

        self.dataset_type_df = self.filter_dataset_csv()
    
    def filter_dataset_csv(self):
        dataset_type_df = self.dataset_df[self.dataset_df["data_type"] == self.data_type]

        if self.data_type == "train" and self.few_shot_samples_per_class is not None:
            # Sort to ensure deterministic order
            dataset_type_df = dataset_type_df.sort_values(by=["class_name", "slide_id"]).reset_index(drop=True)
            np.random.seed(self.seed) # Ensure reproducible sampling
            sampled_dfs = []
            for class_name in self.class_names_list:
                class_df = dataset_type_df[dataset_type_df["class_name"] == class_name]
                if len(class_df) > self.few_shot_samples_per_class:
                    class_df = class_df.sample(n=self.few_shot_samples_per_class, random_state=self.seed)
                sampled_dfs.append(class_df)
            dataset_type_df = pd.concat(sampled_dfs).reset_index(drop=True)

            sampled_csv_path = Path(f"dataset_csv/{self.dataset_name}_few_shot_{self.few_shot_samples_per_class}_seed_{self.seed}.csv")
            dataset_type_df.to_csv(sampled_csv_path, index=False)

        return dataset_type_df

    def __len__(self):
        return len(self.dataset_type_df)
    
    def __getitem__(self, idx):
        row = self.dataset_type_df.iloc[idx]
        data_type, class_name, slide_id = row["data_type"], str(row["class_name"]), row["slide_id"]

        slide_info_path = f"{self.dataset_root}/{self.dataset_name}/feature/{self.feature_extractor_name}/{data_type}/{class_name}/{slide_id}.pt"
        slide_info = torch.load(slide_info_path)

        slide_features = slide_info["features"]
        slide_caption = slide_info["caption"]
        slide_label = self.class_names_list.index(class_name)

        return slide_features, slide_caption, slide_label

    def get_weights_of_class(self):
        class_names = self.dataset_type_df["class_name"]
        labels = [self.class_names_list.index(class_name) for class_name in class_names] # convert to labels
        unique, counts = np.unique(labels, return_counts=True)
        label_cnt = list(zip(unique, counts))
        label_cnt.sort(key=lambda x: x[0])
        weight_arr = np.array([x[1] for x in label_cnt], dtype=float)
        weight_arr = np.max(weight_arr) / weight_arr
        return torch.from_numpy(weight_arr.astype(np.float32))


class PatchFeaturesWSIDataModule(pl.LightningDataModule):
    def __init__(self, dataset_root, dataset_name, dataset_df,
                 class_names_list, feature_extractor_name="resnet50-tr-supervised-imagenet1k", num_workers=4,
                 few_shot_samples_per_class=None, seed=None):
        super().__init__()
        self.dataset_root = dataset_root
        self.dataset_name = dataset_name
        self.dataset_df = dataset_df

        self.class_names_list = class_names_list

        self.feature_extractor_name = feature_extractor_name

        self.num_workers = num_workers

        self.few_shot_samples_per_class = few_shot_samples_per_class
        self.seed = seed

        self.dataset_train = None
        self.dataset_val = None
        self.dataset_test = None

    def setup(self, stage=None):
        if self.dataset_train is None:
            self.dataset_train = PatchFeaturesWSIDataset(self.dataset_root, self.dataset_name, self.dataset_df, "train",
                                            self.class_names_list, self.feature_extractor_name,
                                            self.few_shot_samples_per_class, self.seed)

            self.dataset_val = PatchFeaturesWSIDataset(self.dataset_root, self.dataset_name, self.dataset_df, "val", 
                                                self.class_names_list, self.feature_extractor_name,
                                                self.few_shot_samples_per_class, self.seed)
            
        
            self.dataset_test = PatchFeaturesWSIDataset(self.dataset_root, self.dataset_name, self.dataset_df, "test",
                                                self.class_names_list, self.feature_extractor_name,
                                                self.few_shot_samples_per_class, self.seed)

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=1, shuffle=True, num_workers=self.num_workers,
                          drop_last=False, pin_memory=False)
    
    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=1, shuffle=False, num_workers=self.num_workers,
                          drop_last=False, pin_memory=False)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=1, shuffle=False, num_workers=self.num_workers,
                          drop_last=False, pin_memory=False)