import numpy as np
import jpeg4py as jpeg
import cv2
import gc

import torch
import torchvision.transforms as T

from pathlib import Path
from torch.utils.data import Dataset

def read_img(img_path):
    if str(img_path).endswith((".jpg", ".jpeg")):
        img = jpeg.JPEG(img_path).decode()
    else:
        img = cv2.cvtColor(cv2.imread(str(img_path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    
    return img

def patch_transforms(data_mean, data_std):

    transforms = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=data_mean, std=data_std) # conch doesnt use normalization
    ])

    return transforms

class ExtractFeaturesWSIDataset(Dataset):
    def __init__(self, dataset_root, dataset_name, dataset_df, data_mean, data_std,
                 class_names_list, batch_patch_size=100):
        super().__init__()
        self.dataset_root = dataset_root
        self.dataset_name = dataset_name
        self.dataset_df = dataset_df
        self.data_mean = data_mean
        self.data_std = data_std

        self.class_names_list = class_names_list

        self.batch_patch_size = batch_patch_size
    
    def __len__(self):
        return len(self.dataset_df)

    def __getitem__(self, idx):
        row = self.dataset_df.iloc[idx]
        data_type, class_name, slide_id, caption = row["data_type"], row["class_name"], row["slide_id"], row["caption"]

        slide_dir = f"{self.dataset_root}/{self.dataset_name}/merge_patch/{data_type}/{class_name}/{slide_id}"
        merge_patch_paths = list(Path(slide_dir).glob('*'))

        patches = []
        transforms = patch_transforms(self.data_mean, self.data_std)

        for merge_patch_path in merge_patch_paths:
            merge_patch = read_img(merge_patch_path)
            assert len(merge_patch.shape) == 3, f"Invalid shape: {merge_patch.shape}"  # Ensure valid shape
            h, w, c = merge_patch.shape

            # Split the merge patches to ensure original patches (224 x 224) are preserved
            batch_patch = np.split(merge_patch, h // w, axis=0)  # 200, 224, 224, 3

            for i in range(0, len(batch_patch), self.batch_patch_size):
                current_batch = batch_patch[i:i+self.batch_patch_size]
                
                # Apply transformations and collect patches
                transformed_batch_patch = torch.stack([transforms(patch) for patch in current_batch], dim=0)
                patches.append(transformed_batch_patch)

                # Release memory
                del current_batch
                del transformed_batch_patch
                torch.cuda.empty_cache()
                gc.collect() # Ensure memory is released

        patches = torch.cat(patches, dim=0)  # Concatenate patches

        # Release memory
        del merge_patch_paths, batch_patch, merge_patch
        torch.cuda.empty_cache()
        gc.collect() # Ensure memory is released

        return data_type, class_name, slide_id, patches, caption