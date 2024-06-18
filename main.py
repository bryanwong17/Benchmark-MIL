import argparse
import time
from pathlib import Path

import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping

from option import add_common_arguments, add_dtfd_mil_arguments
from utils import seed_everything, save_parameters, get_loss_weight
from dataset import get_class_names
from dataset.patch_features_wsi_dataset import PatchFeaturesWSIDataModule
from model import get_model_module

def main(args):
    """Main function to run the training and evaluation process."""

    try:
        torch.set_float32_matmul_precision('medium')
    except Exception as e:
        print("Unable to activate TensorCore")
        print(e)

    class_names_list = get_class_names(args.dataset_name)
    num_classes = len(class_names_list)
    print(f"{args.dataset_name} has {num_classes} classes")

    dataset_csv_path = Path(f"dataset_csv/{args.dataset_name}.csv")
    dataset_csv_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_df = pd.read_csv(dataset_csv_path)

    metrics = {
        "Seed": [],
        "Test ACC": [],
        "Test AUC": [],
        "Time (min)": [],
        "Time (sec)": []
    }

    for seed in args.seed:

        # set seed for reproducibility
        seed_everything(seed)

        start_seed_time = time.time()
 
        save_dir = Path(
            args.output_dir, 
            args.dataset_name, 
            f"{args.mil_model}-{args.distill}" if args.mil_model == "DTFD-MIL" else args.mil_model, 
            args.feature_extractor
        )
        save_dir.mkdir(parents=True, exist_ok=True)

        save_parameters(args, save_dir)

        data_module = PatchFeaturesWSIDataModule(
            args.dataset_root, args.dataset_name, dataset_df, 
            class_names_list, args.feature_extractor, args.num_workers
        )

        loss_weight = get_loss_weight(args, data_module)
    
        trainer_model = get_model_module(args, args.mil_model, args.num_feats, num_classes, loss_weight=loss_weight)

        save_seed_dir = Path(save_dir, f"seed_{seed}")
        save_seed_dir.mkdir(parents=True, exist_ok=True)

        # Logger and Callbacks
        # logger = WandbLogger(project=args.project, name=f"{args.mil_model}_{args.feature_extractor}")

        checkpoint_callback = ModelCheckpoint(
            monitor="Loss/val",
            dirpath=save_seed_dir,
            filename="best-{epoch:02d}",
            save_top_k=1,
            verbose=True,
            mode="min"
        )

        early_stop_callback = EarlyStopping(
            monitor="Loss/val",
            min_delta=0.00,
            patience=args.patience,
            verbose=True,
            mode="min"
        )

        # Initialize trainer
        trainer = pl.Trainer(
            default_root_dir=save_seed_dir,
            max_epochs=args.epochs, log_every_n_steps=50, num_sanity_val_steps=0,
            precision=args.precision,
            accelerator="gpu", devices=args.gpu_id,
            # logger=logger,
            callbacks=[checkpoint_callback, early_stop_callback],
            strategy="ddp" if len(args.gpu_id) > 1 else "auto"
        )
        
        # Train model
        trainer.fit(trainer_model, data_module)

        # Test model
        if len(args.gpu_id) > 1:
            torch.distributed.destroy_process_group()  # Test only with single GPU
            if trainer.is_global_zero:
                trainer = pl.Trainer(
                    default_root_dir=save_seed_dir,
                    num_sanity_val_steps=0,
                    # logger=logger,
                    accelerator="gpu", devices=[args.gpu_id[0]]
                )
                test_results = trainer.test(trainer.model, data_module)
        else:
            test_results = trainer.test(trainer_model, data_module)
        
        # Calculate elapsed time
        end_seed_time = time.time()
        elapsed_time = end_seed_time - start_seed_time
        minutes, seconds = divmod(elapsed_time, 60)

        # Record metrics
        metrics["Seed"].append(int(seed))
        metrics["Test ACC"].append(round(test_results[0]["final_test/ACC"] * 100, 3))
        metrics["Test AUC"].append(round(test_results[0]["final_test/AUROC"] * 100, 3))
        metrics["Time (min)"].append(int(minutes))
        metrics["Time (sec)"].append(int(seconds))

   
    metrics_df = pd.DataFrame(metrics)

    total_times_sec = metrics_df["Time (min)"] * 60 + metrics_df["Time (sec)"]
    avg_time_sec = total_times_sec.mean()
    std_time_sec = total_times_sec.std()
    avg_minutes, avg_seconds = divmod(avg_time_sec, 60)
    std_minutes, std_seconds = divmod(std_time_sec, 60)

    final_metrics = {
        "Seed": "Average ± Std",
        "Test ACC": f"{metrics_df['Test ACC'].mean():.3f} ± {metrics_df['Test ACC'].std():.3f}",
        "Test AUC": f"{metrics_df['Test AUC'].mean():.3f} ± {metrics_df['Test AUC'].std():.3f}",
        "Time (min)": f"{int(avg_minutes)} ± {int(std_minutes)}",
        "Time (sec)": f"{int(avg_seconds)} ± {int(std_seconds)}"
    }

    final_metrics_df = pd.DataFrame([final_metrics])
    metrics_df = pd.concat([metrics_df, final_metrics_df], ignore_index=True)

    for key, value in metrics_df.items():
        print(f"{key}: {value}")
    
    metrics_df.to_csv(save_dir / "final_results.csv", index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    args, unknown = parser.parse_known_args()

    if "DTFD-MIL" in args.mil_model:
        add_dtfd_mil_arguments(parser)
    
    args = parser.parse_args()

    main(args)
