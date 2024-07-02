import pandas as pd
import numpy as np
import argparse
import jpeg4py as jpeg
import cv2
import time
import pickle

from tqdm import tqdm
from joblib import Parallel, delayed
from pathlib import Path

from dataset import get_class_names

def read_img_resize(img_path, img_size):
    if str(img_path).endswith((".jpg", "jpeg")):
        img = jpeg.JPEG(img_path).decode()
    else:
        img = cv2.cvtColor(cv2.imread(str(img_path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    
    if img.shape[0] != img_size:
        img = resize_img(img, img_size)
    
    return img

def resize_img(img, patch_size):
    return cv2.resize(img, (patch_size, patch_size))

def list_split(list_, n):
    for x in range(0, len(list_), n):
        yield list_[x:x+n]

def process_tissue(parallel, slide_path, tissue_coords, patch_size, batch_size, output_dir, tissue_name):
    patch_paths = [
        slide_path / f"{slide_path.stem}_{coord[0]}-{coord[1]}.png"
        for coord in tissue_coords
    ]
    merge_patches = []
    for batch in list_split(patch_paths, batch_size):
        patches = parallel(delayed(read_img_resize)(img_path=patch_path, img_size=patch_size) for patch_path in batch)
        merge_patch = np.concatenate(patches, axis=0)
        merge_patches.append(merge_patch)
    
    tissue_output_dir = output_dir / tissue_name
    tissue_output_dir.mkdir(parents=True, exist_ok=True)
    
    if len(merge_patches) == 1:
        cv2.imwrite(str(tissue_output_dir / f"{tissue_name}.jpg"),
                    cv2.cvtColor(merge_patches[0], cv2.COLOR_RGB2BGR))
    else:
        for i, merge_patch in enumerate(merge_patches):
            cv2.imwrite(str(tissue_output_dir / f"{tissue_name}_{i}.jpg"),
                        cv2.cvtColor(merge_patch, cv2.COLOR_RGB2BGR))
    
    return len(merge_patches)

def main(args):
    start_time = time.time()

    parallel = Parallel(n_jobs=args.parallel_n, backend="loky")

    output_dir = Path(f"{args.dataset_root}/MergeTissue")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    dataset_csv_path = Path(f"dataset_csv/{args.dataset_name}.csv")
    class_names = get_class_names(args.dataset_name)
    
    slide_info = []

    if not dataset_csv_path.exists():
        for class_name in class_names:
            class_path = Path(f"{args.dataset_root}/patches/{args.magnification}/{args.size}/{class_name}")
            slide_paths = [p for p in class_path.glob('*') if p.is_dir()]

            for slide_path in tqdm(slide_paths, desc=f"{class_name}"):

                patch_file = Path(f"{args.dataset_root}/wsi_pt_files/{class_name}/{slide_path.stem}.pt")
                
                with open(patch_file, 'rb') as f:
                    data = pickle.load(f)
                
                tissues = data[args.magnification][args.size]['tissues']

                for tissue_name, tissue_data in tissues.items():
                    tissue_coords = tissue_data['coordinates']
                    num_merged_patches = process_tissue(parallel, slide_path, tissue_coords, args.patch_size, args.batch_size, output_dir / args.magnification / str(args.size) / class_name / slide_path.stem, tissue_name)

                    if args.use_caption:
                        caption_df = pd.read_csv(f"dataset_csv/{args.dataset_name}_captions.csv")
                        caption = caption_df[caption_df["id"] == slide_path.stem]["text"].iloc[0]
                    else:
                        caption = None

                    slide_info.append([class_name, f"{slide_path.stem}/{tissue_name}", num_merged_patches, caption])

        slide_data = pd.DataFrame(slide_info, columns=["class_name", "slide_id/tissue_name", "num_tissues", "caption"])
        slide_data.to_csv(dataset_csv_path, index=False)

    dataset_df = pd.read_csv(dataset_csv_path)
    print(f"{len(dataset_df)} slides found consisting of {len(class_names)} classes")

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Merging patches completed in {total_time // 3600:.0f} hours, {(total_time % 3600) // 60:.0f} minutes, and {total_time % 60:.0f} seconds")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default="/data1", help="Path to WSI dataset root folders")
    parser.add_argument("--dataset_name", type=str, default="seegene_new_new", help="The name of dataset")

    parser.add_argument("--use_caption", action="store_true", help="Use caption for the dataset")

    parser.add_argument("--output_ext", type=str, default=".jpg", help="Output extension of the patches")

    parser.add_argument("--patch_size", type=int, default=224, help="The size of each patch after resize")

    parser.add_argument("--magnification", type=str, default='x10', choices=["x5", "x10", "x20"])

    parser.add_argument("--batch_size", type=int, default=200, help="Batch size used to merge patches")
    parser.add_argument("--parallel_n", type=int, default=10, help="Number of parallel processes to merge patches")

    parser.add_argument("--size", type=int, default=256, help="size of image")
    
    args = parser.parse_args()
    main(args)
    print("Done")