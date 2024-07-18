from argparse import ArgumentParser
from .path import ROOT_DIR

basic_parser = ArgumentParser()
basic_parser.add_argument( "-r", "--random_seed", type=int, default=42, help="The random seed for all dependencies (default: 42)")

# For data
basic_parser.add_argument("--data_dir", type=str, default=ROOT_DIR, help="The path to the dataset (default: ./)", )
basic_parser.add_argument("--src_size", type=int, default=112, help="The size of source model (default: 112)", )
basic_parser.add_argument("--num_workers", type=int, default=12, help="Number of workers for data loading (default: 12)", ) 
basic_parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training (default: 128)", ) 

# For source model
basic_parser.add_argument( "-M", "--model", type=str, choices=["VGG"], help="The type of model (required)", ) 
basic_parser.add_argument( "-L", "--level", type=int, default=1, help="The number of conv layers per block (default: 1)", ) 
basic_parser.add_argument( "-P", "--pooling", type=int, default=3, help="The number of pooling, i.e. the number of blocks (default: 3)", ) 
basic_parser.add_argument( "-w", "--conv_width", type=int, default=32, help="The number of channel of convolutional layer (default: 32)", ) 

# For optimizer
basic_parser.add_argument( "--learning_rate", type=float, default=1e-3, help="Learning rate (default: 1e-3)" ) 
basic_parser.add_argument( "--weight_decay", type=float, default=1e-3, help="Weight decay (default: 1e-3)", ) 

# For trainer
basic_parser.add_argument( "-m", "--max_steps", type=int, default=5000, help="Maximum number of steps for training (default: 5000)", )
basic_parser.add_argument( "-d", "--dry_run", action="store_true", help="Perform a dry run without training (default: False)", ) 
basic_parser.add_argument( "-p", "--patience", type=int, default=5, help="The patience epochs for early stop. (default: 5)", )
