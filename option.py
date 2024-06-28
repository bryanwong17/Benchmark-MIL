
def add_common_arguments(parser):
    # seed
    parser.add_argument('--seed', type=lambda s: [int(item) for item in s.split(',')], default=None)

    parser.add_argument("--output-dir", type=str, default="results", help="An output directory")

    # dataset related parameters
    parser.add_argument("--dataset-root", type=str, help="Path to WSI dataset root folders")
    parser.add_argument("--dataset-name", type=str, default=None, help="The name of dataset")
    
    parser.add_argument("--num-workers", type=int, default=112, help="Set number of workers equal to the number of CPU cores")

    # feature extractor
    parser.add_argument("--feature-extractor", type=str, default="resnet50-tr-supervised-imagenet1k", help="The name of feature extractor")
    parser.add_argument("--num-feats", type=int, default=1024, help="The number of features extracted from the feature extractor")
    
    # MIL model
    parser.add_argument("--mil-model", type=str, default=None, 
                        choices=["mean_pooling", "max_pooling", "ABMIL", "GABMIL", "DSMIL", "CLAM-SB", "CLAM-MB",
                                  "TransMIL", "DTFD-MIL"],
                        help="The name of MIL model or aggregator")

    # training related parameters
    parser.add_argument("--few-shot-samples-per-class", type=int, default=None, help="Number of samples per class for few-shot learning")

    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs to train")
    parser.add_argument("--precision", type=int, default=32, help="32 or 16 bit precision during training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate (default: 1e-4)")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="Weight decay of the optimizer (default: 1e-2)")
    parser.add_argument("--lr-factor", type=float, default=1., help="Learning rate multiplication for feature extractor (default: 1.0)")
    parser.add_argument("--opt", type=str, default="adam", help="Optimizer used for training (Adam | AdamW)")
    
    parser.add_argument("--loss-weight", type=lambda s: [float(item) for item in s.split(',')], default=None,
                        help="Weight of each class")
    parser.add_argument("--auto-loss-weight", action="store_true", help="Automatically calculate the weight of each class")

    parser.add_argument("--batch-size", type=int, default=1, help="batch size used during training (default: 1)")
    parser.add_argument("--accumulate-grad-batches", type=int, default=1, help="simulate larger batch size by "
                                                                               "accumulating gradients")

    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    
    # gpu
    parser.add_argument('--gpu-id', type=lambda s: [int(item) for item in s.split(',')], default=None)

    # logging
    # parser.add_argument("--project", type=str, default=None, help="Project name for wandb logging")
    # parser.add_argument("--name", type=str, default=None, help="Unique name for the experiment in wandb logging")

def add_dtfd_mil_arguments(parser):
    # DTFD-MIL related parameters
    parser.add_argument("--distill", type=str, default="MaxMinS", choices=["MaxMinS", "MaxS", "AFS"], help="Distillation method")
    parser.add_argument("--total-instance", type=int, default=4, help="Total number of instances")
    parser.add_argument("--numGroup", type=int, default=4, help="Number of groups")
    parser.add_argument("--grad-clipping",type=int, default=5, help="Gradient clipping value")
    parser.add_argument("--lr-decay-ratio", type=float, default=0.2, help="Learning rate decay ratio")
