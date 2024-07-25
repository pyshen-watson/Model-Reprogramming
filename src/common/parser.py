from argparse import ArgumentParser
from .path import ROOT_DIR

basic_parser = ArgumentParser()
basic_parser.add_argument( "-r", "--random_seed", type=int, default=42, help="The random seed for all dependencies (default: 42)")

# For data
basic_parser.add_argument("--data_dir", type=str, default=ROOT_DIR, help="The path to the dataset (default: ./)", )
basic_parser.add_argument("--dataset", type=str, default="imagenet10", choices=["cifar10", "stl10", "svhn", "imagenet10"], help="The name of target dataset, it can be cifar10, stl10 or svhn (default: cifar10)", )
basic_parser.add_argument("--src_size", type=int, default=112, help="The size of source model (default: 112)", )
basic_parser.add_argument("--num_workers", type=int, default=12, help="Number of workers for data loading (default: 12)", ) 
basic_parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training (default: 128)", ) 

# For source model
basic_parser.add_argument( "-m", "--model", type=str, choices=["CNN", "VGG", "ResNet"], help="The type of model (required)", ) 
basic_parser.add_argument( "-l", "--level", type=int, default=1, help="The number of conv layers per group (default: 1)", ) 
basic_parser.add_argument( "-g", "--group", type=int, default=1, help="The number of groups. (default: 1)", ) 
basic_parser.add_argument( "-w", "--conv_width", type=int, default=32, help="The number of channel of convolutional layer (default: 32)", ) 

# For optimizer
basic_parser.add_argument( "--learning_rate", type=float, default=1e-3, help="Learning rate (default: 1e-3)" ) 
basic_parser.add_argument( "--weight_decay", type=float, default=1e-3, help="Weight decay (default: 1e-3)", ) 
basic_parser.add_argument( "--loss_fn", type=str, default="CE", choices=["CE", "MSE"], help="The loss function is either cross entropy(CE) or mean square error (MSE) (default: CE)", ) 

# For trainer
basic_parser.add_argument( "-G", "--gpu_id", type=int, default=0, help="The id of the (default: 0)", )
basic_parser.add_argument( "-M", "--max_steps", type=int, default=10000, help="Maximum number of steps for training (default: 5000)", )
basic_parser.add_argument( "-D", "--dry_run", action="store_true", help="Perform a dry run without training (default: False)", ) 
basic_parser.add_argument( "-P", "--patience", type=int, default=5, help="The patience epochs for early stop. (default: 5)", )
