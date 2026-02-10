"""
Unified LineML Training Script.

This script serves as the main entry point for training the LineML model.
It supports both:
1. Simple Mode: Training on a single GPU.
2. Parallel Mode: Distributed Data Parallel (DDP) training across multiple GPUs.

Usage:
    python run_unified_training.py --mode simple --gpu 0
    python run_unified_training.py --mode parallel
"""

import os
import sys
import argparse
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch_geometric.loader import NeighborLoader
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Creating a tensor from a list.*")

# Add local paths
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
dp_path = os.path.join(current_dir, "DataPropocessing")
if dp_path not in sys.path: sys.path.append(dp_path)
models_path = os.path.join(current_dir, "Models")
if models_path not in sys.path: sys.path.append(models_path)

from DataPropocessing.linegraph_loader import LinkClassificationDataManager
from LineMLmodel import LineMLlinkPredictor as MLGNN
from utils.training import UnifiedTrainer

def run_training(rank, world_size, args):
    """
    Main training function to be executed by each process.

    Args:
        rank (int): Process rank (0 for single GPU, 0-N for DDP).
        world_size (int): Total number of processes.
        args (argparse.Namespace): Command line arguments.
    """
    is_ddp = args.mode == 'parallel'
    
    if is_ddp:
        UnifiedTrainer.setup_ddp(rank, world_size)
    
    # Device setup
    if is_ddp:
        device = torch.device(f"cuda:{rank}")
    else:
        # Simple Mode
        device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
        rank = 0 # Default rank for logs
    
    dataset_name = "BUP"
    local_data_dir = os.path.abspath(os.path.join(current_dir, "..", "data"))
    
    # --- Configuration ---
    config = {
        'features': 'random',
        'pruning': {'method': 'knn', 'k': 0},
        'pos_neg_ratio': 2.0, 
        'sampling_strategy': 'degree'
    }

    # --- Data Loading ---
    # Setup DataManager
    dm = LinkClassificationDataManager(local_data_dir, verbose=args.verbose)
    
    # In DDP, ensure base data (raw graph) is downloaded/processed by Rank 0 first
    if is_ddp:
        if rank == 0:
            if args.verbose: print(f"[DDP] Rank 0 checking base data...")
            dm.loader.load_or_process_graph(dataset_name, config)
        dist.barrier()
    
    # Generate/Load Line Graph (In-Memory)
    lg_data = dm.process_dataset(dataset_name, config)
    
    # Feature validation
    if torch.isnan(lg_data.x).any() or torch.isinf(lg_data.x).any():
        lg_data.x = torch.nan_to_num(lg_data.x, nan=0.0)
    
    # --- DataLoaders ---
    if is_ddp:
        # Manual splitting for DDP
        train_idx = lg_data.train_mask.nonzero(as_tuple=False).view(-1)
        # Partition indices
        num_train = len(train_idx)
        indices = torch.arange(num_train)
        rank_indices = indices[rank::world_size]
        local_train_idx = train_idx[rank_indices]
        
        lg_data = lg_data.to(device)
        train_loader = NeighborLoader(lg_data, num_neighbors=[10, 5], input_nodes=local_train_idx, 
                                      batch_size=args.batch_size, shuffle=True, num_workers=2)
    else:
        # Simple Mode
        train_idx = lg_data.train_mask.nonzero(as_tuple=False).view(-1)
        lg_data = lg_data.to(device)
        train_loader = NeighborLoader(lg_data, num_neighbors=[10, 5], input_nodes=train_idx, 
                                      batch_size=args.batch_size, shuffle=True, num_workers=4)

    # Val/Test Loaders (Rank 0 only for eval)
    val_loader, test_loader = None, None
    if rank == 0:
        val_idx = lg_data.val_mask.nonzero(as_tuple=False).view(-1)
        test_idx = lg_data.test_mask.nonzero(as_tuple=False).view(-1)
        val_loader = NeighborLoader(lg_data, num_neighbors=[10, 5], input_nodes=val_idx, batch_size=args.batch_size)
        test_loader = NeighborLoader(lg_data, num_neighbors=[10, 5], input_nodes=test_idx, batch_size=args.batch_size)
    
    # --- Model Setup ---
    input_dim = lg_data.x.shape[1]
    model_config = {
        'hidden_dim': 256,
        'num_sage_layers': 3,
        'dropout': 0.3,
        'num_classifier_heads': 2,
        'output_dim': 2,
        'use_metric_learning': True, 
        'metric_projection_dim': 128,
        'triplet_margin':1.0
    }
    
    model = MLGNN(input_dim, model_config).to(device)
    
    # --- Training ---
    if rank == 0:
        if args.verbose:
            print(f"Starting Training | Mode: {args.mode} | Device: {device}")
        print(f"\n=== [V] Training Model (Early Stopping Active) ===")

    trainer = UnifiedTrainer(model, config, device, rank, is_ddp, verbose=args.verbose)
    
    start_time = time.time()
    auc, acc, final_epoch = trainer.run(train_loader, val_loader, test_loader, epochs=args.epochs, patience=20)
    total_time = time.time() - start_time
    
    if rank == 0:
        print(f"Final Results | AUC: {auc:.4f} | Accuracy: {acc:.4f} | Time: {total_time:.2f}s | Epochs: {final_epoch}")

    if is_ddp:
        UnifiedTrainer.cleanup_ddp()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified LineML Training Script")
    parser.add_argument('--mode', type=str, default='simple', choices=['simple', 'parallel'], help='Training mode')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID for simple mode')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output and progress bars')
    
    args = parser.parse_args()
    
    if args.mode == 'parallel':
        world_size = torch.cuda.device_count()
        if world_size < 2:
            print("Less than 2 GPUs detected. Falling back to simple mode on GPU 0.")
            args.mode = 'simple'
            run_training(0, 1, args)
        else:
            mp.spawn(run_training, args=(world_size, args), nprocs=world_size, join=True)
    else:
        run_training(0, 1, args)
