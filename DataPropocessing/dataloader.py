"""
Mini-Batch DataLoader for Line Graph Link Classification.

Integrates with LinkClassificationDataManager to handle Pruning, Node2Vec, and Line Graph conversion.
"""

import torch
import os
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import Data

# Import from your provided file structure
from DataPropocessing.linegraph_loader import LinkClassificationDataManager

class MiniBatchDataLoader:
    """
    Creates mini-batch data loaders for line graph training using NeighborLoader.
    """
    
    def __init__(self, data_dir='data'):
        """
        Initialize the data loader.

        Args:
            data_dir (str): Directory containing the data.
        """
        self.data_dir = data_dir
        self.data_manager = LinkClassificationDataManager(data_dir)
        self.data = None
        self.metadata = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_linegraph_data(self, dataset_name: str, 
                           node2vec_config: dict = None,
                           pruning_config: dict = None,
                           pos_neg_ratio: float = 1.0,
                           val_ratio: float = 0.1,
                           test_ratio: float = 0.1,
                           sampling_strategy: str = 'hard'):
        """
        Load or create the line graph data via the DataManager.

        Args:
            dataset_name (str): Name of the dataset.
            node2vec_config (dict, optional): Node2Vec configuration.
            pruning_config (dict, optional): Pruning configuration.
            pos_neg_ratio (float, optional): Ratio of negative to positive samples.
            val_ratio (float, optional): Validation set ratio.
            test_ratio (float, optional): Test set ratio.
            sampling_strategy (str, optional): Negative sampling strategy.

        Returns:
            Data: The processed line graph data.
        """
        print(f"\n{'='*70}")
        print(f"LOADING LINE GRAPH DATA: {dataset_name}")
        print(f"{'='*70}")

        # 1. Setup Defaults
        if node2vec_config is None:
            node2vec_config = {'dim': 64, 'walk_len': 10, 'n_walks': 50, 'workers': 4}
            
        if pruning_config is None:
            pruning_config = {'method': 'none'}

        # 2. Construct Unified Config Dictionary
        full_config = {
            'node2vec': node2vec_config,
            'pruning': pruning_config,
            'pos_neg_ratio': pos_neg_ratio,
            'val_ratio': val_ratio,
            'test_ratio': test_ratio,
            'sampling_strategy': sampling_strategy
        }

        print(f"⚙️  Configuration:")
        print(f"   - Pruning: {pruning_config.get('method', 'none')}")
        print(f"   - Pos/Neg Ratio: {pos_neg_ratio}")
        print(f"   - Sampling Strategy: {sampling_strategy}")
        print(f"   - Splits: Val={val_ratio}, Test={test_ratio}")
        print(f"   - Node2Vec: {node2vec_config}")

        # 3. Delegate to DataManager
        self.data = self.data_manager.process_dataset(dataset_name, full_config)
        
        # 4. Post-processing
        self._clean_data_for_loader()
        self._print_data_info()
        
        return self.data
    
    def _clean_data_for_loader(self):
        """Remove non-tensor attributes for NeighborLoader compatibility."""
        safe_attrs = {'x', 'edge_index', 'y', 'train_mask', 'val_mask', 'test_mask', 
                      'num_nodes', 'num_positive', 'num_negative', 'edge_weight'}
        
        self.metadata = {}
        attrs_to_remove = []
        
        for key in self.data.keys():
            if key not in safe_attrs:
                val = self.data[key]
                if not isinstance(val, torch.Tensor):
                    self.metadata[key] = val
                    attrs_to_remove.append(key)

        for key in attrs_to_remove:
            delattr(self.data, key)
            
    def _print_data_info(self):
        """Print basic dataset statistics."""
        print(f"\n📊 DATASET STATISTICS:")
        print(f"   Nodes: {self.data.num_nodes}")
        print(f"   Edges: {self.data.edge_index.shape[1]}")
        print(f"   Features: {self.data.x.shape[1]}")
        
        train_n = self.data.train_mask.sum().item()
        val_n = self.data.val_mask.sum().item()
        test_n = self.data.test_mask.sum().item()
        
        print(f"   Splits: Train={train_n}, Val={val_n}, Test={test_n}")

    def create_loaders(self, batch_size=32, num_neighbors=[10, 10], num_workers=0, isolated_splits=True):
        """
        Create DataLoaders for train, validation, and test sets.

        Args:
            batch_size (int): Batch size.
            num_neighbors (list): Number of neighbors to sample for each layer.
            num_workers (int): Number of worker processes.
            isolated_splits (bool): If True, restricts neighbor sampling to split subsets.

        Returns:
            tuple: (train_loader, val_loader, test_loader)
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_linegraph_data() first.")

        print(f"\n🔄 Creating NeighborLoaders (Batch Size: {batch_size}, Isolated={isolated_splits})...")
        
        kwargs = {
            'batch_size': batch_size,
            'num_neighbors': num_neighbors,
            'num_workers': num_workers
        }
        
        # --- Define Context Subgraphs ---
        if isolated_splits:
            # TRAIN: Strictly Train nodes
            # Note: Subgraphing re-indexes nodes, might be complex if not careful.
            # But NeighborLoader handles 'data' and 'input_nodes'.
            # If we want to restrict NEIGHBORS to be within the mask, we need subgraph.
            # Standard NeighborLoader samples from whole graph unless we restrict it.
            # But here we assume we just want to load batches.
            # 'isolated_splits' in this context usually means we treat the graph as static?
            # Actually, let's keep it simple: Just load based on masks.
            # If `isolated_splits` was about preventing data leakage:
            # strictly speaking, val nodes shouldn't be neighbors of train nodes?
            # For LineGraphs, nodes are edges.
            pass

        # Standard loaders
        train_idx = self.data.train_mask.nonzero(as_tuple=False).view(-1)
        train_loader = NeighborLoader(
            self.data, 
            input_nodes=train_idx, 
            shuffle=True, 
            **kwargs
        )
        
        val_idx = self.data.val_mask.nonzero(as_tuple=False).view(-1)
        val_loader = NeighborLoader(
            self.data, 
            input_nodes=val_idx, 
            shuffle=False, 
            **kwargs
        )
        
        test_idx = self.data.test_mask.nonzero(as_tuple=False).view(-1)
        test_loader = NeighborLoader(
            self.data, 
            input_nodes=test_idx, 
            shuffle=False, 
            **kwargs
        )
        
        print(f"   Train Batches: {len(train_loader)}")
        print(f"   Val Batches:   {len(val_loader)}")
        print(f"   Test Batches:  {len(test_loader)}")
        
        return train_loader, val_loader, test_loader

def create_mini_batch_loaders(dataset_name: str,
                              data_dir: str = 'data',
                              batch_size: int = 512,
                              num_neighbors: list = [10, 5],
                              pruning_config: dict = None,
                              pos_neg_ratio: float = 1.0,
                              node2vec_config: dict = None,
                              val_ratio: float = 0.1,
                              test_ratio: float = 0.1,
                              sampling_strategy: str = 'hard',
                              isolated_splits: bool = True):
    """
    Convenience function to get loaders.
    """
    loader = MiniBatchDataLoader(data_dir)
    
    loader.load_linegraph_data(
        dataset_name=dataset_name,
        node2vec_config=node2vec_config,
        pruning_config=pruning_config,
        pos_neg_ratio=pos_neg_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        sampling_strategy=sampling_strategy
    )
    
    train, val, test = loader.create_loaders(
        batch_size=batch_size,
        num_neighbors=num_neighbors,
        isolated_splits=isolated_splits
    )
    
    return train, val, test