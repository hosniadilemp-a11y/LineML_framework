import torch
import numpy as np
import networkx as nx
import random
from sklearn.cluster import KMeans
from collections import defaultdict
from torch_geometric.data import Data
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from torch_geometric.utils import to_scipy_sparse_matrix

class UnifiedLineGraphPruner:
    """
    Consolidated pruning techniques for line graphs.
    Includes Random, Degree, Feature-based, Spectral, and Community-based methods.
    """

    @staticmethod
    def apply_pruning(linegraph_data: Data, config: dict) -> Data:
        """
        Apply pruning to the line graph based on the configuration.

        Args:
            linegraph_data (Data): The input line graph data.
            config (dict): Pruning configuration dictionary.
                           Format: {'method': 'method_name', 'param1': value, ...}

        Returns:
            Data: Pruned line graph data.
        """
        method = config.get('method', 'none')
        print(f"✂️ Applying Pruning Method: {method}")

        if method == 'none':
            return linegraph_data
        
        elif method == 'random':
            return UnifiedLineGraphPruner.prune_random(
                linegraph_data, 
                prune_ratio=config.get('prune_ratio', 0.3),
                seed=config.get('seed', 42)
            )
        
        elif method == 'degree_based':
            return UnifiedLineGraphPruner.prune_degree_based(
                linegraph_data,
                degree_threshold=config.get('degree_threshold', 0.3),
                mode=config.get('mode', 'high')
            )
        
        elif method == 'spectral':
            return UnifiedLineGraphPruner.prune_spectral_sparsification(
                linegraph_data,
                preserve_ratio=config.get('preserve_ratio', 0.7)
            )
        
        elif method == 'community':
            return UnifiedLineGraphPruner.prune_community_preserving(
                linegraph_data,
                n_clusters=config.get('n_clusters', 10),
                preserve_intra=config.get('preserve_intra', 0.8),
                preserve_inter=config.get('preserve_inter', 0.3)
            )
            
        elif method == 'feature_similarity':
            return UnifiedLineGraphPruner.prune_feature_similarity(
                linegraph_data,
                similarity_threshold=config.get('similarity_threshold', 0.5)
            )
            
        elif method == 'knn':
            return UnifiedLineGraphPruner.prune_sparsify_knn(
                linegraph_data,
                k=config.get('k', 5)
            )

        elif method == 'adaptive':
             return UnifiedLineGraphPruner.prune_adaptive(
                 linegraph_data,
                 target_density=config.get('target_density', 0.1)
             )
        
        else:
            print(f"⚠ Warning: Unknown pruning method '{method}'. Returning original data.")
            return linegraph_data

    # --- Basic Methods ---

    @staticmethod
    def prune_random(linegraph_data, prune_ratio=0.3, seed=42):
        """
        Randomly remove a fraction of edges.

        Args:
            linegraph_data (Data): Input graph.
            prune_ratio (float): Ratio of edges to remove.
            seed (int): Random seed.

        Returns:
            Data: Pruned graph.
        """
        random.seed(seed)
        edge_index = linegraph_data.edge_index.clone()
        num_edges = edge_index.shape[1]
        num_keep = int(num_edges * (1 - prune_ratio))
        
        keep_indices = random.sample(range(num_edges), num_keep)
        keep_indices = sorted(keep_indices) # Sort for determinism
        
        pruned_data = linegraph_data.clone()
        pruned_data.edge_index = edge_index[:, keep_indices]
        pruned_data.pruning_method = 'random'
        return pruned_data

    @staticmethod
    def prune_degree_based(linegraph_data, degree_threshold=0.3, mode='high'):
        """
        Prune edges based on the degree of connected nodes.

        Args:
            linegraph_data (Data): Input graph.
            degree_threshold (float): Quantile threshold for pruning.
            mode (str): 'high' to prune interactions between high-degree nodes, 
                        'low' for low-degree.

        Returns:
            Data: Pruned graph.
        """
        edge_index = linegraph_data.edge_index.clone()
        n_nodes = linegraph_data.num_nodes
        
        # Calculate node degrees
        deg = torch.bincount(edge_index[0], minlength=n_nodes) + \
              torch.bincount(edge_index[1], minlength=n_nodes)
        
        # Calculate threshold
        threshold = torch.quantile(deg.float(), degree_threshold).item()
        
        if mode == 'high':
            # Remove edges where at least one node has degree > threshold
            mask = (deg[edge_index[0]] < threshold) & (deg[edge_index[1]] < threshold)
        else: # mode == 'low'
            # Remove edges where at least one node has degree < threshold
            mask = (deg[edge_index[0]] > threshold) | (deg[edge_index[1]] > threshold)
            
        pruned_data = linegraph_data.clone()
        pruned_data.edge_index = edge_index[:, mask]
        pruned_data.pruning_method = 'degree_based'
        return pruned_data

    @staticmethod
    def prune_feature_similarity(linegraph_data, similarity_threshold=0.5):
        """
        Prune edges between nodes with low feature cosine similarity.

        Args:
            linegraph_data (Data): Input graph.
            similarity_threshold (float): Minimum cosine similarity to keep an edge.

        Returns:
            Data: Pruned graph.
        """
        x = linegraph_data.x
        edge_index = linegraph_data.edge_index
        
        # Compute Cosine Sim for edges only
        u = x[edge_index[0]]
        v = x[edge_index[1]]
        
        # Cosine Sim = (u . v) / (|u| |v|)
        sim = torch.nn.functional.cosine_similarity(u, v, dim=1)
        
        mask = sim >= similarity_threshold
        
        pruned_data = linegraph_data.clone()
        pruned_data.edge_index = edge_index[:, mask]
        pruned_data.pruning_method = 'feature_similarity'
        return pruned_data
        
    @staticmethod
    def prune_sparsify_knn(linegraph_data, k=5):
        """
        Keep only the k-nearest neighbors for each node based on feature similarity.

        Args:
            linegraph_data (Data): Input graph.
            k (int): Number of neighbors to keep.

        Returns:
            Data: Pruned graph.
        """
        x = linegraph_data.x
        edge_index = linegraph_data.edge_index
        num_nodes = linegraph_data.num_nodes
        
        # If graph is too small, skip
        if num_nodes < k: return linegraph_data
        
        # Optimization: Only evaluate existing edges? 
        # KNN usually implies creating NEW edges too, but here we are PRUNING.
        # So we select top-k EXISTING neighbors for each node.
        
        # 1. Calculate scores for all edges
        u = x[edge_index[0]]
        v = x[edge_index[1]]
        scores = torch.nn.functional.cosine_similarity(u, v, dim=1)
        
        # 2. Top-k per node
        # Convert to dense or use scatter?
        # A simple way: Sort edges by (source, score)
        
        # Create a tensor of [u, v, score, edge_idx]
        # We need to process both directions u->v and v->u logic?
        # Usually KNN is directed. We can make it symmetric later.
        
        # Let's filter: For each u, keep top-k v's (indices in edge_index)
        
        # Using a dense mask approach is slow. 
        # Lexsort by (u, -score)
        
        src = edge_index[0]
        
        # Optim: use pandas groupby tail
        df = pd.DataFrame({
            'u': edge_index[0].numpy(),
            'score': scores.numpy(),
            'idx': np.arange(len(scores))
        })
        
        # For each u, keep top k
        keep_indices_u = df.sort_values('score', ascending=False).groupby('u').head(k)['idx'].values
        
        # For each v, keep top k (to ensure symmetry if desired, or just consider u as source)
        # For pruning, we usually want to keep an edge if it's a top-k neighbor for *either* node.
        df_rev = pd.DataFrame({
            'u': edge_index[1].numpy(), # treat destination as source
            'score': scores.numpy(),
            'idx': np.arange(len(scores))
        })
        keep_indices_v = df_rev.sort_values('score', ascending=False).groupby('u').head(k)['idx'].values
        
        combined_keep_indices = np.unique(np.concatenate([keep_indices_u, keep_indices_v]))
        
    @staticmethod
    def prune_spectral_sparsification(linegraph_data, preserve_ratio=0.7):
        """
        Prune graph using spectral sparsification (Effective Resistance).

        Args:
            linegraph_data (Data): Input graph.
            preserve_ratio (float): Ratio of edges to preserve.

        Returns:
            Data: Pruned graph.
        """
        edge_index = linegraph_data.edge_index
        num_nodes = linegraph_data.num_nodes
        num_edges = edge_index.shape[1]
        
        # Target number of edges
        num_keep = int(num_edges * preserve_ratio)
        if num_keep >= num_edges: return linegraph_data

        try:
            # 1. Construct Laplacian
            adj = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes)
            adj = (adj + adj.T) / 2
            
            # Degree matrix
            degrees = np.array(adj.sum(axis=1)).flatten()
            laplacian = sp.diags(degrees) - adj
            
            # 2. Spectral Embedding
            k = min(num_nodes - 1, max(10, int(np.log(num_nodes) * 4)))
            
            try:
                evals, evecs = spl.eigsh(laplacian, k=k, which='SM', sigma=1e-6)
            except:
                 evals, evecs = spl.eigsh(laplacian, k=k, sigma=1e-6)

            # 3. Approximated Effective Resistances
            evals = np.abs(evals)
            evals[evals < 1e-9] = 1e-9
            
            Z = evecs @ np.diag(1.0 / np.sqrt(evals))
            
            src, dst = edge_index[0].numpy(), edge_index[1].numpy()
            diff = Z[src] - Z[dst]
            Re = np.sum(diff**2, axis=1)
            
            # 4. Sampling Probabilities
            probs = Re / np.sum(Re)
            
            # 5. Sample
            sampled_indices = np.random.choice(num_edges, size=num_keep, replace=False, p=probs)
            sampled_indices.sort()
            
            # 6. Reweight (Optional but recommended for spectral correctness)
            # We add edge_weight attribute to the Data object
            selected_probs = probs[sampled_indices]
            selected_probs = np.clip(selected_probs, 1e-9, 1.0)
            raw_weights = 1.0 / selected_probs
            scale = num_edges / raw_weights.sum()
            final_weights = torch.tensor(raw_weights * scale, dtype=torch.float)
            
            pruned_data = linegraph_data.clone()
            pruned_data.edge_index = edge_index[:, sampled_indices]
            pruned_data.edge_weight = final_weights
            pruned_data.pruning_method = 'spectral'
            
            return pruned_data

        except Exception as e:
            print(f"Warning: Spectral Pruning failed ({e}). Fallback to Random.")
            return UnifiedLineGraphPruner.prune_random(linegraph_data, 1.0 - preserve_ratio)

    @staticmethod
    def prune_community_preserving(linegraph_data, n_clusters=10, preserve_intra=0.8, preserve_inter=0.3):
        """
        Prune graph while preserving community structure.

        Args:
            linegraph_data (Data): Input graph.
            n_clusters (int): Number of communities to detect.
            preserve_intra (float): Ratio of intra-community edges to keep.
            preserve_inter (float): Ratio of inter-community edges to keep.

        Returns:
            Data: Pruned graph.
        """
        x = linegraph_data.x.numpy()
        edge_index = linegraph_data.edge_index
        n_edges = edge_index.shape[1]
        
        # 1. Quick Clustering
        if x.shape[0] > n_clusters:
            kmeans = KMeans(n_clusters=n_clusters, n_init=5, random_state=42).fit(x)
            labels = torch.tensor(kmeans.labels_)
        else:
            labels = torch.zeros(x.shape[0], dtype=torch.long)
        
        u_labels = labels[edge_index[0]]
        v_labels = labels[edge_index[1]]
        
        is_intra = (u_labels == v_labels)
        
        intra_mask = is_intra
        inter_mask = ~is_intra
        
        intra_indices = torch.where(intra_mask)[0]
        inter_indices = torch.where(inter_mask)[0]
        
        n_intra = len(intra_indices)
        n_inter = len(inter_indices)
        
        keep_intra = intra_indices[torch.randperm(n_intra)[:int(n_intra * preserve_intra)]]
        keep_inter = inter_indices[torch.randperm(n_inter)[:int(n_inter * preserve_inter)]]
        
        keep_indices = torch.cat([keep_intra, keep_inter])
        keep_indices, _ = torch.sort(keep_indices)
        
        pruned_data = linegraph_data.clone()
        pruned_data.edge_index = edge_index[:, keep_indices]
        pruned_data.pruning_method = 'community'
        return pruned_data

    @staticmethod
    def prune_adaptive(linegraph_data, target_density=0.1):
        """
        Prune to a target density by randomly removing edges if needed.

        Args:
            linegraph_data (Data): Input graph.
            target_density (float): Desired density.

        Returns:
            Data: Pruned graph.
        """
        n = linegraph_data.num_nodes
        current_edges = linegraph_data.edge_index.shape[1]
        
        if n < 2: return linegraph_data
        
        max_edges = n * (n - 1) // 2
        target_edges = int(max_edges * target_density)
        
        if current_edges > target_edges:
            # We need to remove edges. 
            # prune_ratio is fraction TO REMOVE.
            # We want to KEEP target_edges.
            # keep_ratio = target / current
            # prune_ratio = 1 - keep_ratio
            ratio = 1.0 - (target_edges / current_edges)
            pruned_data = UnifiedLineGraphPruner.prune_random(linegraph_data, prune_ratio=ratio)
        else:
            pruned_data = linegraph_data.clone()
        
        pruned_data.pruning_method = 'adaptive'
        return pruned_data