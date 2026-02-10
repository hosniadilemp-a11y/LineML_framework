
import sys
import os
import time
import tracemalloc
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score
from tqdm import tqdm
import gc
import queue

# --- Paths ---
# Use absolute path of the script as anchor
current_dir = os.path.dirname(os.path.abspath(__file__))
# LineMLcode is where this script is.
sys.path.append(current_dir)

# DataPropocessing is inside LineMLcode (current_dir)
dp_path = os.path.join(current_dir, "DataPropocessing")
if dp_path not in sys.path: sys.path.append(dp_path)

# Models is inside LineMLcode (current_dir/Models) or root/Models?
# Based on ls, Models is a subdir of LineMLcode.
models_path = os.path.join(current_dir, "Models")
if models_path not in sys.path: sys.path.append(models_path)

import DataPropocessing.linegraph_loader as lg_loader
from DataPropocessing.dataloader import create_mini_batch_loaders
from Models.MLGNNmodel import SophisticatedLinkPredictor as MLGNN


# --- Memory Helpers ---
def track_memory(device):
    if device.type == 'cuda':
        return torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    else:
        current, peak = tracemalloc.get_traced_memory()
        return peak / (1024 * 1024)

def reset_memory(device):
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    else:
        tracemalloc.start()

# --- Logging ---
HEADER = [
    'Dataset', 'Experiment', 'Method', 'Nodes', 'Edges', 
    'Total Training Time (s)', 'Training Time/Epoch (s)',
    'Inference Time (s)', 'Inference Latency (ms/node)',
    'Peak Memory (MB)', 'Memory Type',
    'AUC', 'AP', 'Accuracy', 'F1',
    'Status'
]

def get_paths(dataset_name):
    # flexible results path
    base_out = os.path.join(current_dir, "..", "results_custom", dataset_name)
    if not os.path.exists(base_out): os.makedirs(base_out, exist_ok=True)
    return os.path.join(base_out, f"lineml_original_{dataset_name.lower()}.csv")

def log_result(dataset_name, res, lock=None):
    if lock:
        lock.acquire()
    try:
        out_csv = get_paths(dataset_name)
        df = pd.DataFrame([res])
        for col in HEADER:
            if col not in df.columns: df[col] = None
        df = df[HEADER]
        df.to_csv(out_csv, mode='a', header=not os.path.exists(out_csv), index=False)
        print(f"Logged results to {out_csv}")
    finally:
        if lock:
            lock.release()

# --- Experiment Function ---
def run_line_graph_experiment(dataset_name, exp_name, pruning_config, use_metric_learning, device_id, lock):
    # Ensure patch is applied in worker process
    # lg_loader.ParallelLineGraphConverter.convert_to_linegraph = sequential_wrapper
    
    device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
    print(f"\n⚡ [Worker-{device_id}][LineGraph][{dataset_name}] Running {exp_name} on {device}...")
    
    try:
        # Config
        batch_size = 512
        epochs = 50 
        
        # USE ORIGINAL FEATURES
        node2vec_config = 'original'
        
        # Correct path to data directory (sibling to LineMLcode)
        # current_dir is LineMLcode
        data_dir_path = os.path.abspath(os.path.join(current_dir, "..", "data"))
        
        # Create Loaders
        # Note: we are passing pruning_config={'method': 'none'}
        train_loader, val_loader, test_loader, criterion, full_data = create_mini_batch_loaders(
            dataset_name=dataset_name,
            data_dir=data_dir_path, 
            batch_size=batch_size,
            pruning_config=pruning_config,
            pos_neg_ratio=1.0,
            val_ratio=0.1,
            test_ratio=0.1,
            sampling_strategy='hard',
            isolated_splits=True,
            node2vec_config=node2vec_config 
        )
        
        # Ensure criterion is on the correct device
        if hasattr(criterion, 'to'):
            criterion = criterion.to(device)
        
        num_lg_nodes = full_data.num_nodes
        num_lg_edges = full_data.edge_index.shape[1]
        
        print(f"Graph loaded. Nodes: {num_lg_nodes}, Edges: {num_lg_edges}, Feature Dim: {full_data.x.shape[1]}")

        if num_lg_edges == 0:
            log_result(dataset_name, {
                'Dataset': dataset_name, 'Experiment': exp_name, 'Method': pruning_config['method'],
                'Status': 'Empty Graph'
            }, lock)
            return

        # --- Data Validation & Normalization ---
        if torch.isnan(full_data.x).any() or torch.isinf(full_data.x).any():
            print(f"   ⚠️ [Worker-{device_id}] Found NaNs/Infs in features! Replacing with zeros.")
            full_data.x = torch.nan_to_num(full_data.x, nan=0.0, posinf=0.0, neginf=0.0)
            
        # Optional: Normalize features if they have large variance
        # Simple row-wise normalization or standard scaling could help
        # For now, let's just ensure they are float32
        full_data.x = full_data.x.float()

        # Model Init
        input_dim = full_data.x.shape[1]
        model_config = {
            'hidden_dim': 128,
            'num_sage_layers': 3,
            'dropout': 0.5,
            'num_classifier_heads': 2,
            'output_dim': 2,
            'use_metric_learning': use_metric_learning,
            'metric_projection_dim': 64,
            'triplet_margin': 1.0
        }
        
        model = MLGNN(input_dim, model_config).to(device)
        # Reduced LR and added weight_decay for stability
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)
        
        # Training
        reset_memory(device)
        tracemalloc.start()
        start_train = time.time()
        
        model.train()
        for epoch in range(epochs):
            loss_epoch = 0
            steps = 0
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                logits = model(batch.x, batch.edge_index)
                batch_size_curr = batch.batch_size
                pred = logits[:batch_size_curr]
                target = batch.y[:batch_size_curr].long()
                loss = criterion(pred, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                loss_epoch += loss.item()
                steps += 1
            
            if epoch % 10 == 0:
                print(f"   [Worker-{device_id}][{exp_name}] Epoch {epoch+1}/{epochs} Loss: {loss_epoch/steps:.4f}")

        end_train = time.time()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        if device.type == 'cuda':
            peak_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
            mem_type = 'GPU'
        else:
            peak_mb = peak / 10**6
            mem_type = 'CPU'
            
        train_time = end_train - start_train
        time_per_epoch = train_time / epochs
        
        # Inference
        start_inf = time.time()
        model.eval()
        
        preds = []
        targets = []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                logits = model(batch.x, batch.edge_index)
                probs = torch.softmax(logits[:batch.batch_size], dim=1)[:, 1]
                preds.append(probs.cpu())
                targets.append(batch.y[:batch.batch_size].cpu())
        
        y_scores = torch.cat(preds).numpy()
        y_true = torch.cat(targets).numpy()
        y_pred = (y_scores > 0.5).astype(int)
        
        inf_time = time.time() - start_inf
        latency = (inf_time * 1000) / len(y_true) if len(y_true) > 0 else 0
        
        # Metrics
        try: auc = roc_auc_score(y_true, y_scores)
        except: auc = 0.5
        ap = average_precision_score(y_true, y_scores)
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        print(f"   ✅ [Worker-{device_id}][{dataset_name}] {exp_name} Finished | AUC: {auc:.4f} | Acc: {acc:.4f}")
        
        # Log
        res = {
            'Dataset': dataset_name,
            'Experiment': exp_name,
            'Method': pruning_config['method'],
            'Nodes': num_lg_nodes,
            'Edges': num_lg_edges,
            'Total Training Time (s)': train_time,
            'Training Time/Epoch (s)': time_per_epoch,
            'Inference Time (s)': inf_time,
            'Inference Latency (ms/node)': latency,
            'Peak Memory (MB)': peak_mb,
            'Memory Type': mem_type,
            'AUC': auc, 'AP': ap, 'Accuracy': acc, 'F1': f1,
            'Status': 'Success'
        }
        log_result(dataset_name, res, lock)
        
        # Cleanup
        del model, train_loader, val_loader, test_loader, full_data
        gc.collect()
        if device.type == 'cuda': torch.cuda.empty_cache()

    except Exception as e:
        print(f"   ❌ [Worker-{device_id}][{dataset_name}] {exp_name} Failed: {e}")
        import traceback
        traceback.print_exc()
        log_result(dataset_name, {'Dataset': dataset_name, 'Experiment': exp_name, 'Method': pruning_config['method'], 'Status': f'Failed: {e}'}, lock)

# --- Worker Process ---
def worker(gpu_id, task_queue, lock):
    # Determine device
    
    while True:
        try:
            task = task_queue.get(timeout=3)
        except queue.Empty:
            break
            
        if task is None:
            break
            
        dataset_name, exp_name, config, use_metric_learning = task
        run_line_graph_experiment(dataset_name, exp_name, config, use_metric_learning, gpu_id, lock)

def main():
    mp.set_start_method('spawn', force=True)
    
    dataset = "Cora"
    
    # Task Definitions
    # User Request: LineML method, Original Features, No Pruning.
    # Creating 2 tasks to utilize both GPUs
    pruning_exps = [
        ("LineML_Cora_Original_Run1", {'method': 'none'}, True), 
        ("LineML_Cora_Original_Run2", {'method': 'none'}, True), 
    ]
    
    task_queue = mp.Queue()
    
    # Populate Queue
    for exp_name, config, use_ml in pruning_exps:
        task_queue.put((dataset, exp_name, config, use_ml))
            
    # Add poison pills
    num_workers = 2 
    for _ in range(num_workers):
        task_queue.put(None)
        
    lock = mp.Lock()
    processes = []
    
    gpu_assignments = [0, 1] # Use CUDA:0 and CUDA:1
    
    print(f"=== Starting LineML  Experiment ===")

  

    local_data_dir = os.path.abspath(os.path.join(current_dir, "..", "data"))
    print(f"🔄 Pre-loading data sequentially to ensure clean cache...")
    try:
        # Load once to generate caches (Cora.pkl and Cora_lg.pkl)
        # We don't need the loaders, just trigger data processing
        from DataPropocessing.linegraph_loader import LinkClassificationDataManager
        dm = LinkClassificationDataManager(local_data_dir)
        # Config matching the experiment
        config = {
            'node2vec': 'original',
            'pruning': {'method': 'none'},
            'pos_neg_ratio': 1.0,
            'val_ratio': 0.1,
            'test_ratio': 0.1,
            'sampling_strategy': 'hard'
        }
        dm.process_dataset(dataset, config)
        print("✅ Data pre-loaded successfully.")
    except Exception as e:
        print(f"❌ Data pre-loading failed: {e}")
        # Proceed anyway, maybe workers can handle it or it will crash there
    
    for i in range(num_workers):
        gpu_id = gpu_assignments[i]
        p = mp.Process(target=worker, args=(gpu_id, task_queue, lock))
        p.start()
        processes.append(p)
        print(f"Started Worker-{i} on CUDA:{gpu_id}")
        
    for p in processes:
        p.join()
        
    print("=== Completed ===")

if __name__ == "__main__":
    main()
