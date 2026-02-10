
import os
import sys
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader
import pandas as pd
import time
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score
import gc

# Add local paths
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
dp_path = os.path.join(current_dir, "DataPropocessing")
if dp_path not in sys.path: sys.path.append(dp_path)
models_path = os.path.join(current_dir, "Models")
if models_path not in sys.path: sys.path.append(models_path)

from DataPropocessing.linegraph_loader import LinkClassificationDataManager
from MLGNNmodel import SophisticatedLinkPredictor as MLGNN

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def run_ddp(rank, world_size):
    setup(rank, world_size)
    
    # Configuration
    dataset_name = "Cora"
    batch_size = 512
    epochs = 50
    lr = 0.0001 # Robust LR
    
    # 1. Data Loading (Rank 0 prepares, others wait/load cache)
    # We use the sequential loading trick from before to ensure cache exists
    local_data_dir = os.path.abspath(os.path.join(current_dir, "..", "data"))
    
    # Sync: Rank 0 prepares data
    if rank == 0:
        print(f"🔄 [Rank {rank}] Checking/Preparing data...")
        dm = LinkClassificationDataManager(local_data_dir)
        config = {
            'node2vec': 'original',
            'pruning': {'method': 'none'},
            'pos_neg_ratio': 1.0, 
            'sampling_strategy': 'hard'
        }
        # This triggers generation and caching
        dm.process_dataset(dataset_name, config)
        print(f"✅ [Rank {rank}] Data ready.")
        
    dist.barrier() # Wait for Rank 0
    
    # Now all ranks load the cached data
    dm = LinkClassificationDataManager(local_data_dir)
    config = {
        'node2vec': 'original',
        'pruning': {'method': 'none'},
        'pos_neg_ratio': 1.0,
        'sampling_strategy': 'hard'
    }
    lg_data = dm.process_dataset(dataset_name, config)
    
    # Feature validation
    if torch.isnan(lg_data.x).any() or torch.isinf(lg_data.x).any():
        lg_data.x = torch.nan_to_num(lg_data.x, nan=0.0)
    
    # Create DataLoaders with DistributedSampler
    # We need to manually split indices for the masks since PyG doesn't support DistributedSampler natively on Data objects easily like this
    # Actually, we can just use the masks to filter, then wrap in a dataset
    
    class TensorDataset(torch.utils.data.Dataset):
        def __init__(self, data, mask):
            self.x = data.x
            self.edge_index = data.edge_index # Full graph structure needed for convolution? 
            # Wait, MLGNN is GNN? If it's pure link prediction on line graph features, 
            # line graph nodes are edges in original graph.
            # If MLGNN uses edge_index of the line graph, we need subgraph sampling or full graph.
            # DDP usually implies mini-batching. 
            # The previous script used `create_mini_batch_loaders` which returns `NeighborLoader` or similar.
            # Let's see what `dataloader.py` does.
            pass

    # Re-using dataloader.py logic but adapting for DDP
    # `create_mini_batch_loaders` logic:
    # It returns NeighborLoader. NeighborLoader supports DDP via `input_nodes` splitting?
    # PyG NeighborLoader doesn't directly support DistributedSampler in the standard way.
    # Standard practice: Split input_nodes manually based on rank.
    
    # Let's do manual splitting of indices for training
    train_idx = lg_data.train_mask.nonzero(as_tuple=False).view(-1)
    val_idx = lg_data.val_mask.nonzero(as_tuple=False).view(-1)
    test_idx = lg_data.test_mask.nonzero(as_tuple=False).view(-1)
    
    # Split train_idx for this rank
    num_train = len(train_idx)
    indices = torch.arange(num_train)
    # Simple partitioning
    rank_indices = indices[rank::world_size]
    local_train_idx = train_idx[rank_indices]
    
    # Loaders
    # We use NeighborLoader from PyG
    from torch_geometric.loader import NeighborLoader
    
    # We need the full data on each GPU for sampling (usually)
    lg_data = lg_data.to(rank) # Send full graph structure to GPU? 
    # Warning: If graph is large, this might OOM. But Cora Line Graph is ~10k nodes, fits easily.
    
    train_loader = NeighborLoader(
        lg_data,
        num_neighbors=[10, 5],
        input_nodes=local_train_idx,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    
    # Val/Test usually only on Rank 0 or all? Let's do Rank 0 for eval to save time
    if rank == 0:
        val_loader = NeighborLoader(lg_data, num_neighbors=[10, 5], input_nodes=val_idx, batch_size=batch_size)
        test_loader = NeighborLoader(lg_data, num_neighbors=[10, 5], input_nodes=test_idx, batch_size=batch_size)
    
    # Model Setup
    input_dim = lg_data.x.shape[1]
    model_config = {
        'hidden_dim': 128,
        'num_sage_layers': 2,
        'dropout': 0.5,
        'num_classifier_heads': 2,
        'output_dim': 2,
        'use_metric_learning': True,
        'metric_projection_dim': 64,
        'triplet_margin': 0.5
    }
    
    model = MLGNN(input_dim, model_config).to(rank)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss().to(rank)
    
    # Training Loop
    if rank == 0:
        print(f"🚀 [Rank {rank}] Starting DDP Training (Input Dim: {input_dim})")
        
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        steps = 0
        
        # train_loader has its own shuffling (via NeighborLoader), but DistributedSampler isn't used explicitly
        # because we manually split indices. So we don't need sampler.set_epoch(epoch)
        
        for batch in train_loader:
            batch = batch.to(rank)
            optimizer.zero_grad()
            
            # Forward
            # MLGNN expects x and edge_index
            logits = model(batch.x, batch.edge_index)
            
            # Slicing for target size (batch size)
            batch_size_curr = batch.batch_size
            pred = logits[:batch_size_curr]
            target = batch.y[:batch_size_curr].long()
            
            loss = criterion(pred, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            steps += 1
            
        avg_loss = total_loss / steps if steps > 0 else 0
        # Reduce loss for logging? Or just log Rank 0?
        # Let's just log Rank 0's view
        if rank == 0 and epoch % 10 == 0:
             print(f"   [Epoch {epoch+1}/{epochs}] Loss: {avg_loss:.4f}")
             
    train_time = time.time() - start_time
    
    # Evaluation (Rank 0 only)
    if rank == 0:
        print("🔍 [Rank 0] Evaluating...")
        model.eval()
        preds = []
        targets = []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(rank)
                logits = model(batch.x, batch.edge_index) # DDP model wrapper handles this? Yes.
                probs = torch.softmax(logits[:batch.batch_size], dim=1)[:, 1]
                preds.append(probs.cpu())
                targets.append(batch.y[:batch.batch_size].cpu())
        
        y_scores = torch.cat(preds).numpy()
        y_true = torch.cat(targets).numpy()
        y_pred = (y_scores > 0.5).astype(int)
        
        auc = roc_auc_score(y_true, y_scores)
        acc = accuracy_score(y_true, y_pred)
        
        print(f"✅ [DDP Final] AUC: {auc:.4f} | Acc: {acc:.4f} | Time: {train_time:.2f}s")
        
        # Log to CSV
        res = {
            'Dataset': dataset_name,
            'Experiment': 'DDP_Hadamard',
            'Method': 'none',
            'Nodes': lg_data.num_nodes,
            'Edges': lg_data.edge_index.shape[1],
            'Features': input_dim,
            'Total Training Time (s)': train_time,
            'AUC': auc, 'Accuracy': acc,
            'Status': 'Success'
        }
        out_csv = os.path.join(current_dir, "..", "results_custom", dataset_name, f"lineml_ddp_{dataset_name.lower()}.csv")
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        pd.DataFrame([res]).to_csv(out_csv, mode='a', header=not os.path.exists(out_csv), index=False)
        print(f"📄 Results saved to {out_csv}")

    cleanup()

if __name__ == "__main__":
    world_size = 2 # 2 GPUs
    mp.spawn(run_ddp, args=(world_size,), nprocs=world_size, join=True)
