import pandas as pd
import numpy as np
import argparse
import jpeg4py as jpeg
import cv2
import time

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
        every_chunk = list_[x:x+n]
        yield every_chunk

def process_batch(parallel, patch_paths, batch_size, patch_size):
    merge_patches = []
    for batch_paths in list_split(patch_paths, batch_size):
        patches = parallel(delayed(read_img_resize)(img_path=patch_path, img_size=patch_size) for patch_path in batch_paths)
        merge_patch = np.concatenate(patches, axis=0)
        merge_patches.append(merge_patch)
    
    return merge_patches

def main(args):
    start_time = time.time()

    parallel = Parallel(n_jobs=args.parallel_n, backend="loky")

    output_dir = Path(f"{args.dataset_root}/{args.dataset_name}/merge_patch")
    Path(output_dir).mkdir(exist_ok=True)

    dataset_csv_path = Path(f"{"dataset_csv"}/{args.dataset_name}.csv")
    class_names = get_class_names(args.dataset_name)
    
    slide_info = []

    if not dataset_csv_path.exists():
        for data_type in ['train', 'val', 'test']:
            for class_name in class_names:
                class_path = f"{args.dataset_root}/{args.dataset_name}/patch/{data_type}/{class_name}"
                slide_paths = [p for p in Path(class_path).glob('*') if p.is_dir()]

                for slide_path in tqdm(slide_paths, desc=f"{data_type}/{class_name}"):
                    if args.dataset_name == "camelyon16":
                        patch_paths = list(Path(slide_path, "imgs").glob('*'))
                    elif args.dataset_name == "tcga_nsclc":
                        slide_path = Path(slide_path)
                        patch_paths = list(slide_path.glob('*.jpg')) + list(slide_path.glob('*/*.jpg'))
                    else:
                        patch_paths = list(Path(slide_path).glob('*'))

                    merge_patches = process_batch(parallel, patch_paths, args.batch_size, args.patch_size)
                    len_merge_patches = len(merge_patches)

                    for i, merge_patch in enumerate(merge_patches):
                        merge_patch_path = Path(f"{output_dir}/{data_type}/{class_name}/{slide_path.stem}")
                        merge_patch_path.mkdir(parents=True, exist_ok=True)
                                            
                        cv2.imwrite(str(Path(f"{merge_patch_path}/{slide_path.stem}_{i}{args.output_ext}")),
                                     cv2.cvtColor(merge_patch, cv2.COLOR_RGB2BGR))

                    if args.use_caption:
                        caption_df = pd.read_csv(f"dataset_csv/{args.dataset_name}_captions.csv")
                        caption = caption_df[caption_df["id"] == slide_path.stem]["text"].iloc[0]
                    else:
                        caption = None
                    
                    slide_info.append([data_type, class_name, slide_path.stem, len_merge_patches, caption])
        
        slide_data = pd.DataFrame(slide_info, columns=["data_type", "class_name", "slide_id", "len_merge_patches(200)", "caption"])

        slide_data = slide_data.to_csv(dataset_csv_path, index=False)

    dataset_df = pd.read_csv(dataset_csv_path)
    print(f"{len(dataset_df)} slides found consisting of {len(class_names)} classes")

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Merging patches completed in {total_time // 3600:.0f} hours, {(total_time % 3600) // 60:.0f} minutes, and {total_time % 60:.0f} seconds")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=str, default="/vast/kaist/WSI_datasets", help="Path to WSI dataset root folders")
    parser.add_argument("--dataset-name", type=str, default=None, help="The name of dataset")

    parser.add_argument("--use-caption", action="store_true", help="Use caption for the dataset")

    parser.add_argument("--output-ext", type=str, default=".jpg", help="Output extension of the patches")

    parser.add_argument("--patch-size", type=int, default=224, help="The size of each patch after resize")

    parser.add_argument("--batch-size", type=int, default=200, help="Batch size used to merge patches")
    parser.add_argument("--parallel-n", type=int, default=10, help="Number of parallel processes to merge patches")
    
    args = parser.parse_args()
    main(args)
    print("Done")
