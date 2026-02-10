"""
Line Graph Data Loader and Processor.

This module handles:
1. Loading raw graph data (.net files).
2. Converting Primal Graphs to Line Graphs.
3. Generating features (Random, Node2Vec, or Original).
4. Generating negative samples.
5. Managing data via `LinkClassificationDataManager`.
"""

import os
import torch
import numpy as np
import scipy.sparse as sp
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
import pickle
import random
from collections import defaultdict
from tqdm import tqdm
from joblib import Parallel, delayed

from Graphpruner import UnifiedLineGraphPruner

# Attempt to import node2vec, handle failure gracefully
try:
    from node2vec import Node2Vec
except ImportError:
    Node2Vec = None

def compute_incidence_matrix(G):
    """
    Compute the incidence matrix of a graph.

    Args:
        G (nx.Graph): Input networkx graph.

    Returns:
        scipy.sparse.csr_matrix: Incidence matrix.
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()
    
    # Map edges to indices
    # We need a stable ordering of edges
    edges = list(G.edges())
    # Create incidence matrix B (n x m)
    # entries are 1 if node i is incident to edge j
    
    # Indices for sparse matrix
    row_ind = []
    col_ind = []
    data = []
    
    for j, (u, v) in enumerate(edges):
        row_ind.extend([u, v])
        col_ind.extend([j, j])
        data.extend([1, 1])
        
    return sp.csr_matrix((data, (row_ind, col_ind)), shape=(n, m))

def incidence_to_linegraph_adj(B):
    """
    Convert incidence matrix to line graph adjacency matrix.

    Args:
        B (scipy.sparse.csr_matrix): Incidence matrix (n x m).

    Returns:
        scipy.sparse.csr_matrix: Line graph adjacency matrix (m x m).
    """
    # A_L = B.T @ B - 2 * I
    # The diagonal of B.T @ B is the degree of the edge in the line graph + 2
    # because each edge has 2 endpoints.
    # We want adjacency, so we remove the self-loops (diagonal).
    Bi = B.astype(bool).astype(int) # Ensure binary
    # Intersection matrix: (m x m), entry (i, j) is number of shared nodes between edge i and edge j
    # For simple graphs, this is 0, 1, or 2 (if parallel edges, but we assume simple)
    # Actually, for simple graphs, edges share at most 1 node.
    # So B^T @ B gives 2 on diagonal and 1 on off-diagonal if edges share a node.
    return Bi.T @ Bi - 2 * sp.eye(Bi.shape[1])

def process_node_chunk(chunk):
    """Helper to process a chunk of (node, incident_indices) items - Optimized"""
    local_edges = set()
    for _, incident in chunk: # chunk is list of (node, [indices]) items
        n_inc = len(incident)
        if n_inc < 2: continue
        
        # incident is a list of edge indices. We want pairs of these indices.
        # Since these are indices into the edge list, they are integers.
        # We store them as sorted tuples (min, max) to deduplicate undirected edges.
        
        # Optimization: Manual loop is often faster than itertools for simple combinations 
        # when we need to filter/sort.
        for i in range(n_inc):
            e1 = incident[i]
            for j in range(i + 1, n_inc):
                e2 = incident[j]
                if e1 < e2:
                    local_edges.add((e1, e2))
                else:
                    local_edges.add((e2, e1))
    return list(local_edges)

def compute_block(Bi, Bj):
    """Helper to compute dot product of two sparse blocks"""
    return Bi.T @ Bj

class GraphDataLoader:
    """
    Efficient data loader for graph datasets (.net format).
    
    Handles loading raw graphs, generating features (Node2Vec, Random, Original),
    and managing file paths.
    """
    
    def __init__(self, data_dir='data', verbose=False):
        """
        Initialize the loader.

        Args:
            data_dir (str, optional): Root directory for data. Defaults to 'data'.
            verbose (bool, optional): Enable verbose logging. Defaults to False.
        """
        self.data_dir = data_dir
        self.verbose = verbose
        self.raw_dir = os.path.join(data_dir, 'raw')
        self.processed_dir = os.path.join(data_dir, 'processed')
        self.linegraph_dir = os.path.join(data_dir, 'linegraph')
        
        for d in [self.raw_dir, self.processed_dir, self.linegraph_dir]:
            os.makedirs(d, exist_ok=True)
            
    def load_net_file(self, filename):
        """
        Load a graph from a .net file (Pajek format variant).

        Args:
            filename (str): Path to the .net file.

        Returns:
            nx.Graph: The loaded NetworkX graph.
        """
        if self.verbose: print(f"[>] Loading raw graph: {filename}")
        G = nx.Graph()
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        mode = ''
        for line in lines:
            line = line.strip()
            if not line or line.startswith('%'): continue
            
            if line.lower().startswith('*vertices'):
                mode = 'vertices'
                continue
            elif line.lower().startswith('*edges') or line.lower().startswith('*arcs'):
                mode = 'edges'
                continue
                
            parts = line.split()
            if mode == 'vertices':
                # Pajek indices are 1-based, converting to 0-based
                node_id = int(parts[0]) - 1 
                G.add_node(node_id)
            elif mode == 'edges':
                u, v = int(parts[0]) - 1, int(parts[1]) - 1
                w = float(parts[2]) if len(parts) > 2 else 1.0
                G.add_edge(u, v, weight=w)
                
        if self.verbose: print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G

    def generate_node2vec(self, G, dim=32, walk_len=10, n_walks=50, workers=4):
        """
        Generate Node2Vec embeddings for the graph.

        Args:
            G (nx.Graph): Input graph.
            dim (int): Embedding dimension.
            walk_len (int): Length of random walks.
            n_walks (int): Number of walks per node.
            workers (int): Number of parallel workers.

        Returns:
            np.ndarray: Node embeddings (normalized).
        """
        num_nodes = G.number_of_nodes()
        
        if G.number_of_edges() == 0:
            return np.zeros((num_nodes, dim), dtype=np.float32)
        
        if Node2Vec is None:
            print("Warning: Node2Vec not installed. Using Random features.")
            return np.random.randn(num_nodes, dim).astype(np.float32)

        # Robust Node2Vec generation
        n2v = Node2Vec(G, 
                       dimensions=dim, 
                       walk_length=walk_len, 
                       num_walks=n_walks, 
                       workers=workers, 
                       quiet=not self.verbose)
        model = n2v.fit(window=2, min_count=0)
        
        emb = []
        for i in range(num_nodes):
            str_i = str(i)
            if str_i in model.wv:
                emb.append(model.wv[str_i])
            else:
                # Fallback for disconnected nodes
                emb.append(np.zeros(dim))
        
        # Normalize features (Crucial for GNN stability)
        from sklearn.preprocessing import StandardScaler
        return StandardScaler().fit_transform(np.array(emb, dtype=np.float32))

    def load_or_process_graph(self, dataset_name, config):
        """
        Load the graph and generate features based on configuration.

        Args:
            dataset_name (str): Name of the dataset (e.g., 'Cora').
            config (dict): Feature configuration.

        Returns:
            Data: PyG Data object with 'x', 'edge_index', 'num_nodes'.
        """
        pkl_path = os.path.join(self.processed_dir, f"{dataset_name}.pkl")
        raw_path = os.path.join(self.raw_dir, f"{dataset_name}.net")
        if not os.path.exists(raw_path):
            candidates = [f for f in os.listdir(self.raw_dir) if dataset_name in f and f.endswith('.net')]
            if not candidates: raise FileNotFoundError(f"No .net file found for {dataset_name}")
            raw_path = os.path.join(self.raw_dir, candidates[0])

        G = self.load_net_file(raw_path)
        
        # Determine features
        if self.verbose: print('Handling features...')
        
        # New config structure: 'features': 'original' OR {'method': 'random', 'dim': 64}
        feature_config = config.get('features', config.get('node2vec', 'original'))
        
        # Parse config
        if isinstance(feature_config, dict):
            method = feature_config.get('method', 'random')
            params = feature_config
        else:
            method = str(feature_config).lower()
            params = {}
            
        if method in ['none', 'random']:
            if self.verbose: print("   Generating Random Features ...")
            dim = params.get('dim', 64)
            x = np.random.randn(G.number_of_nodes(), dim).astype(np.float32)
            
        elif method == 'original':
             if self.verbose: print("   Using Original Features...")
             try:
                 from torch_geometric.datasets import Planetoid
                 # Fallback to PyG for Cora features if not in G
                 dataset = Planetoid(root=os.path.join(self.data_dir, 'pyg_data'), name=dataset_name)
                 x = dataset[0].x.numpy()
                 if self.verbose: print(f"      Loaded features from PyG: shape {x.shape}")
             except Exception as e:
                 if self.verbose: print(f"      Failed to load original features: {e}. Fallback to Identity/Random.")
                 x = np.eye(G.number_of_nodes(), dtype=np.float32) if G.number_of_nodes() < 2000 else np.random.randn(G.number_of_nodes(), 64).astype(np.float32)
        
        elif method == 'node2vec':
            if self.verbose: print(f"   Running Node2Vec Random Walks...")
            dim = params.get('dim', 32)
            walk_len = params.get('walk_len', 10)
            n_walks = params.get('n_walks', 50)
            workers = params.get('workers', 4)
            x = self.generate_node2vec(G, dim=dim, walk_len=walk_len, n_walks=n_walks, workers=workers)
        else:
            if self.verbose: print(f"   Unknown feature method '{method}'. Defaulting to Random.")
            x = np.random.randn(G.number_of_nodes(), 64).astype(np.float32)
        
        # Create Edge Index (making undirected)
        edge_index = torch.tensor(list(G.edges()), dtype=torch.long).t().contiguous()
        if not nx.is_directed(G):
             edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

        data = Data(x=torch.tensor(x), edge_index=edge_index, num_nodes=G.number_of_nodes())
        return data

class ParallelLineGraphConverter:
    """
    Parallelized Line Graph Construction.
    """
    @staticmethod
    def convert_to_linegraph(original_x, pos_edges, neg_edges, split_labels, n_jobs=4, verbose=False):
        """
        Convert primal graph interactions to line graph.
        
        Args:
            original_x (Tensor): Node features of primal graph.
            pos_edges (list): List of positive edge tuples (u, v).
            neg_edges (list): List of negative edge tuples (u, v).
            split_labels (list): 'train', 'val', or 'test' label for each edge.
            n_jobs (int): Number of parallel workers.
            verbose (bool): Verbose logging.
            
        Returns:
            Data: Line graph Data Object.
        """
        if verbose: print(f"Converting to Line Graph (Parallel Optimized, n_jobs={n_jobs})...")
        all_edges = pos_edges + neg_edges
        
        # 1. Features
        x_np = original_x.numpy() if isinstance(original_x, torch.Tensor) else original_x
        
        # Hadamard product (element-wise multiplication)
        lg_features = [x_np[u] * x_np[v] for u, v in all_edges]
        lg_features = torch.tensor(np.array(lg_features), dtype=torch.float)
        
        # 2. Build node->edges mapping
        node_to_edge_idx = defaultdict(list)
        for idx, (u, v) in enumerate(all_edges):
            node_to_edge_idx[u].append(idx)
            node_to_edge_idx[v].append(idx)
        
        if verbose: print(f"  Constructing connections for {len(all_edges)} nodes...")
        
        # 3. Sort by degree for better load balancing
        items = list(node_to_edge_idx.items())
        items_sorted = sorted(items, key=lambda x: len(x[1]), reverse=True)
        
        # 4. Create chunks with similar total work
        chunk_size = max(1, len(items_sorted) // (n_jobs * 4))
        chunks = [items_sorted[i:i + chunk_size] for i in range(0, len(items_sorted), chunk_size)]
        
        if verbose: print(f"  Processing {len(chunks)} chunks with {n_jobs} workers...")
        
        # 5. Parallel Execution
        results = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(process_node_chunk)(chunk) 
            for chunk in tqdm(chunks, desc="Parallel Construction", disable=not verbose)
        )
        
        # 6. Efficient union of results
        unique_edges = set()
        for res in results:
            unique_edges.update(res)
            
        # 7. Convert to edge index
        if unique_edges:
            edges_list = list(unique_edges)
            edges_array = np.array(edges_list)
            # Undirected: add (u, v) and (v, u)
            src = np.concatenate([edges_array[:, 0], edges_array[:, 1]])
            dst = np.concatenate([edges_array[:, 1], edges_array[:, 0]])
            lg_edge_index = torch.tensor([src, dst], dtype=torch.long)
        else:
            lg_edge_index = torch.empty((2, 0), dtype=torch.long)
        
        y = torch.zeros(len(all_edges), dtype=torch.long)
        y[:len(pos_edges)] = 1 
        
        return Data(
            x=lg_features, 
            edge_index=lg_edge_index, 
            y=y,
            train_mask=torch.tensor(['train' in s for s in split_labels]),
            val_mask=torch.tensor(['val' in s for s in split_labels]),
            test_mask=torch.tensor(['test' in s for s in split_labels]),
            num_nodes=len(all_edges)
        )

class SparseLineGraphConverter:
    @staticmethod
    def convert_to_linegraph(original_x, pos_edges, neg_edges, split_labels):
        print("[*] Converting to Line Graph (Sparse Matrix)...")
        all_edges = pos_edges + neg_edges
        n_original_nodes = original_x.shape[0]
        n_lg_nodes = len(all_edges)
        
        # 1. Feature Construction (Vectorized)
        # Flattening u,v indices for all edges to index into original_x
        u_indices = [e[0] for e in all_edges]
        v_indices = [e[1] for e in all_edges]
        
        # Using numpy advanced indexing (faster than list comprehension)
        x_u = original_x[u_indices].numpy()
        x_v = original_x[v_indices].numpy()
        # Hadamard product
        lg_features = torch.tensor(x_u * x_v, dtype=torch.float)
        
        # 2. Incidence Matrix Construction
        print(f"  Constructing incidence matrix ({n_original_nodes} x {n_lg_nodes})...")
        # B[n, e] = 1 if node n is endpoint of edge e
        # Rows: Nodes, Cols: Edges
        # Each column (edge) has exactly two 1s
        
        row_indices = []
        col_indices = []
        
        # We can construct these lists efficiently
        # edge 0: u0, v0 -> (u0, 0), (v0, 0)
        # edge i: ui, vi -> (ui, i), (vi, i)
        
        row_indices = u_indices + v_indices # Concatenate lists
        col_indices = list(range(n_lg_nodes)) + list(range(n_lg_nodes))
        data = np.ones(len(row_indices), dtype=np.uint8) # Binary matrix
        
        B = sp.csr_matrix((data, (row_indices, col_indices)), shape=(n_original_nodes, n_lg_nodes))
        
        # 3. Line Graph Adjacency = B.T @ B
        # Entry (i, j) is number of shared nodes between edge i and edge j
        # 0: distinct, 1: share one node (adjacent), 2: parallel edges (share 2 nodes)
        print("  Computing B.T @ B...")
        L = B.T @ B
        
        # 4. Extract Edges
        # Remove diagonal (self-loops)
        L.setdiag(0)
        L.eliminate_zeros()
        
        # Get coordinates
        print("  Extracting edges...")
        # L is symmetric, so we get (i, j) and (j, i).
        # We want to keep all non-zero entries.
        # Note: L[i, j] could be 2 if edges share 2 nodes. We treat >0 as an edge.
        # But for simple graphs, it's 1. 
        # Multi-graph logic: if they share 2 nodes, they are connected.
        # So we just need the non-zero structure.
        
        src, dst = L.nonzero()
        
        lg_edges = np.stack([src, dst], axis=0) # Shape (2, E_lg)
        lg_edge_index = torch.tensor(lg_edges, dtype=torch.long).contiguous() if lg_edges.size > 0 else torch.empty((2, 0), dtype=torch.long)
        
        y = torch.zeros(len(all_edges), dtype=torch.long)
        y[:len(pos_edges)] = 1 
        
        return Data(x=lg_features, edge_index=lg_edge_index, y=y, 
                    train_mask=torch.tensor(['train' in s for s in split_labels]),
                    val_mask=torch.tensor(['val' in s for s in split_labels]),
                    test_mask=torch.tensor(['test' in s for s in split_labels]),
                    num_nodes=len(all_edges))

class ParallelSparseLineGraphConverter:


    @staticmethod
    def convert_to_linegraph(original_x, pos_edges, neg_edges, split_labels, n_jobs=4):
        print(f"[*] Converting to Line Graph (Parallel Sparse, n_jobs={n_jobs})...")
        all_edges = pos_edges + neg_edges
        n_original_nodes = original_x.shape[0]
        n_lg_nodes = len(all_edges)
        
        # 1. Feature Construction (Vectorized)
        u_indices = [e[0] for e in all_edges]
        v_indices = [e[1] for e in all_edges]
        x_u = original_x[u_indices].numpy()
        x_v = original_x[v_indices].numpy()
        lg_features = torch.tensor(np.concatenate([x_u, x_v], axis=1), dtype=torch.float)
        
        # 2. Block-wise Incidence Matrix Construction
        # Split edges into chunks for blocks
        chunk_size = max(1, n_lg_nodes // n_jobs)
        edge_chunks = [] 
        # Create slices
        slices = [slice(i, min(i + chunk_size, n_lg_nodes)) for i in range(0, n_lg_nodes, chunk_size)]
        
        print(f"  Constructing {len(slices)} incidence blocks...")
        blocks = []
        for s in slices:
            chunk_u = u_indices[s]
            chunk_v = v_indices[s]
            chunk_len = len(chunk_u)
            
            row_indices = chunk_u + chunk_v
            # Map cols to 0..chunk_len-1 range for local block
            col_indices = list(range(chunk_len)) + list(range(chunk_len))
            data = np.ones(len(row_indices), dtype=np.uint8)
            
            # Block B_k of shape (V, chunk_len)
            B_k = sp.csr_matrix((data, (row_indices, col_indices)), shape=(n_original_nodes, chunk_len))
            blocks.append(B_k)
            
        # 3. Parallel Block Compuation B.T @ B
        # Result is block matrix where R_ij = B_i.T @ B_j
        print("  Computing blocks in parallel...")
        
        # Generate tasks: (i, j)
        tasks = []
        for i in range(len(blocks)):
            for j in range(len(blocks)):
                tasks.append((blocks[i], blocks[j]))
                
        # Run parallel dot products
        # Note: limiting n_jobs helps not oversubscribe if BLAS is threaded
        # Sparse matmul is often single threaded in scipy, so n_jobs helps.
        block_results = Parallel(n_jobs=n_jobs)(delayed(compute_block)(b1, b2) for b1, b2 in tasks)
        
        # Reshape list into grid for bmat
        grid = []
        n_blocks = len(blocks)
        for i in range(n_blocks):
            row = block_results[i*n_blocks : (i+1)*n_blocks]
            grid.append(row)
            
        print("  Assembling result matrix...")
        L = sp.bmat(grid)
        
        # 4. Extract Edges
        L.setdiag(0)
        L.eliminate_zeros()
        src, dst = L.nonzero()
        
        lg_edges = np.stack([src, dst], axis=0)
        lg_edge_index = torch.tensor(lg_edges, dtype=torch.long).contiguous() if lg_edges.size > 0 else torch.empty((2, 0), dtype=torch.long)
        
        y = torch.zeros(len(all_edges), dtype=torch.long)
        y[:len(pos_edges)] = 1 
        
        return Data(x=lg_features, edge_index=lg_edge_index, y=y, 
                    train_mask=torch.tensor(['train' in s for s in split_labels]),
                    val_mask=torch.tensor(['val' in s for s in split_labels]),
                    test_mask=torch.tensor(['test' in s for s in split_labels]),
                    num_nodes=len(all_edges))

class LineGraphConverter:
    """
    Sequential Line Graph Construction and Negative Sampling.
    """
    def __init__(self, verbose=False):
        """
        Initialize the converter.

        Args:
            verbose (bool, optional): Enable verbose logging. Defaults to False.
        """
        self.verbose = verbose
        
    def generate_negative_edges(self, G, n_negatives, strategy='random', seed=42):
        """
        Generate negative edges (non-existing links) for training/evaluation.

        Strategies:
        - 'random': Pure random non-edges.
        - 'hard': Distance-2 pairs (u-v-w), avoiding direct edges.
        - 'degree': Proportional to degree sum/product.
        - 'common_neighbor': Sample pairs with common neighbors but no edge.

        Args:
            G (nx.Graph): Input graph.
            n_negatives (int): Number of negative edges to generate.
            strategy (str, optional): Sampling strategy. Defaults to 'random'.
            seed (int, optional): Random seed. Defaults to 42.

        Returns:
            list: List of negative edge tuples (u, v).
        """
        if self.verbose: print(f"Generating {n_negatives} negatives ({strategy})...")
        random.seed(seed); np.random.seed(seed)
        
        nodes = list(G.nodes())
        existing = set(tuple(sorted((u, v))) for u, v in G.edges())
        negatives = set()
        adj = defaultdict(set)
        for u, v in G.edges(): 
            adj[u].add(v)
            adj[v].add(u)
        
        # Precompute for degree-based sampling
        if strategy == 'degree':
            degrees = dict(G.degree())
            total_degree = sum(degrees.values())
            node_probs = np.array([degrees[n] / total_degree for n in nodes])
            
        attempts = 0
        max_attempts = n_negatives * 100  # Increased for harder strategies
        
        while len(negatives) < n_negatives and attempts < max_attempts:
            attempts += 1
            
            if strategy == 'hard':
                # Distance-2 sampling: u-v-w where u-w not connected
                u = random.choice(nodes)
                if not adj[u]: continue
                v = random.choice(list(adj[u]))
                if not adj[v]: continue
                w = random.choice(list(adj[v]))
                if w == u or w in adj[u]: continue 
                pair = tuple(sorted((u, w)))
                
            elif strategy == 'random':
                # Pure random sampling
                u, v = random.sample(nodes, 2)
                pair = tuple(sorted((u, v)))
                
            elif strategy == 'degree':
                # Degree-based: preferentially sample high-degree nodes
                u, v = np.random.choice(nodes, size=2, replace=False, p=node_probs)
                pair = tuple(sorted((int(u), int(v))))
                
            elif strategy == 'common_neighbor':
                # Sample pairs with common neighbors but no direct edge
                u = random.choice(nodes)
                if not adj[u]: continue
                
                # Find nodes at distance 2 from u (common neighbor exists)
                distance_2 = set()
                for neighbor in adj[u]:
                    for neighbor2 in adj[neighbor]:
                        if neighbor2 != u and neighbor2 not in adj[u]:
                            distance_2.add(neighbor2)
                
                if not distance_2: continue
                v = random.choice(list(distance_2))
                pair = tuple(sorted((u, v)))
                
            else:
                raise ValueError(f"Unknown sampling strategy: {strategy}")
            
            if pair not in existing and pair not in negatives: 
                negatives.add(pair)
        
        if len(negatives) < n_negatives:
            print(f"   ⚠️  Warning: Only generated {len(negatives)}/{n_negatives} negatives")
            
        return list(negatives)

    @staticmethod
    def convert_to_linegraph(original_x, pos_edges, neg_edges, split_labels):
        """
        Sequential conversion to line graph (fallback).

        Args:
            original_x (Tensor): Node features.
            pos_edges (list): Positive edges.
            neg_edges (list): Negative edges.
            split_labels (list): Split labels.

        Returns:
            Data: Line graph Data object.
        """
        print("🔄 Converting to Line Graph...")
        all_edges = pos_edges + neg_edges
        # Hadamard product
        lg_features = [original_x[u].numpy() * original_x[v].numpy() for u, v in all_edges]
        lg_features = torch.tensor(np.array(lg_features), dtype=torch.float)
        
        node_to_edge_idx = defaultdict(list)
        for idx, (u, v) in enumerate(all_edges):
            node_to_edge_idx[u].append(idx); node_to_edge_idx[v].append(idx)
            
        lg_edges = []
        processed_pairs = set()
        print(f"  Constructing connections for {len(all_edges)} nodes...")
        for node, incident in tqdm(node_to_edge_idx.items()):
            n_inc = len(incident)
            if n_inc < 2: continue
            for i in range(n_inc):
                for j in range(i + 1, n_inc):
                    e1, e2 = incident[i], incident[j]
                    pair = tuple(sorted((e1, e2)))
                    if pair not in processed_pairs:
                        lg_edges.append([e1, e2]); lg_edges.append([e2, e1])
                        processed_pairs.add(pair)
                        
        lg_edge_index = torch.tensor(lg_edges, dtype=torch.long).t().contiguous() if lg_edges else torch.empty((2, 0), dtype=torch.long)
        y = torch.zeros(len(all_edges), dtype=torch.long)
        y[:len(pos_edges)] = 1 
        
        return Data(x=lg_features, edge_index=lg_edge_index, y=y, 
                    train_mask=torch.tensor(['train' in s for s in split_labels]),
                    val_mask=torch.tensor(['val' in s for s in split_labels]),
                    test_mask=torch.tensor(['test' in s for s in split_labels]),
                    num_nodes=len(all_edges))

class LinkClassificationDataManager:
    """
    High-level data manager for Link Classification tasks.
    
    Orchestrates:
    1. Loading/Processing primal graph.
    2. Sampling links (pos/neg) and splitting (train/val/test).
    3. converting to Line Graph.
    4. Pruning the line graph.
    """
    def __init__(self, data_dir='data', verbose=False):
        """
        Initialize the Data Manager.

        Args:
            data_dir (str, optional): Data directory. Defaults to 'data'.
            verbose (bool, optional): Verbose logging. Defaults to False.
        """
        self.verbose = verbose
        self.loader = GraphDataLoader(data_dir, verbose=verbose)
        self.converter = LineGraphConverter(verbose=verbose)
        self.pruner = UnifiedLineGraphPruner()
        
    def process_dataset(self, dataset_name, config):
        """
        Process the dataset end-to-end.

        Args:
            dataset_name (str): Dataset name.
            config (dict): Configuration dictionary (pruning, ratios, etc.).

        Returns:
            Data: Processed Line Graph Data object ready for GNN.
        """
        pruning_method = config.get('pruning', {}).get('method', 'none')
        pos_neg_ratio = config.get('pos_neg_ratio', 1.0)
        
        # EXTRACT NEW RATIOS
        val_ratio = config.get('val_ratio', 0.1)
        test_ratio = config.get('test_ratio', 0.1)
        
        # Standard config check (omitted for brevity, we always proceed for simplicity here)
        is_standard_config = False 
        
        # Logging config
        if not is_standard_config:
            if self.verbose:
                print(f"Non-standard config (Ratio: {pos_neg_ratio}, Val: {val_ratio}, Test: {test_ratio}).")
                print("   Skipping cache load. Will create fresh Line Graph.")
        
        print("\n=== [I] Loading Data ===")
        print(f"Dataset: {dataset_name}")
        base_data = self.loader.load_or_process_graph(dataset_name, config)
        G = nx.Graph()
        G.add_nodes_from(range(base_data.num_nodes))
        G.add_edges_from(base_data.edge_index.t().tolist())
        print(f"Graph Structure: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        pos_edges = list(G.edges())
        random.seed(42); random.shuffle(pos_edges)
        
        # Split Data
        n = len(pos_edges)
        n_test = int(n * test_ratio)
        n_val = int(n * val_ratio)
        
        test_pos = pos_edges[:n_test]
        val_pos = pos_edges[n_test:n_test+n_val]
        train_pos = pos_edges[n_test+n_val:]
        
        n_train_neg = int(len(train_pos) * pos_neg_ratio)
        n_val_neg = int(len(val_pos) * pos_neg_ratio)
        n_test_neg = int(len(test_pos) * pos_neg_ratio)
        
        # Get sampling strategy from config (default to 'hard' for backward compatibility)
        sampling_strategy = config.get('sampling_strategy', 'hard')
        
        print(f"\n=== [II] Link Sampling ===")
        print(f"Strategy: {sampling_strategy}")
        print(f"Positive Samples: Train={len(train_pos)}, Val={len(val_pos)}, Test={len(test_pos)}")
        print(f"Negative Samples: Train={n_train_neg}, Val={n_val_neg}, Test={n_test_neg}")
        
        if self.verbose:
            print(f"Generating Negatives...")
        train_neg = self.converter.generate_negative_edges(G, n_train_neg, strategy=sampling_strategy, seed=1)
        val_neg = self.converter.generate_negative_edges(G, n_val_neg, strategy=sampling_strategy, seed=2)
        test_neg = self.converter.generate_negative_edges(G, n_test_neg, strategy=sampling_strategy, seed=3)
        
        pos_all = train_pos + val_pos + test_pos
        neg_all = train_neg + val_neg + test_neg
        
        labels = ['train']*len(train_pos) + ['val']*len(val_pos) + ['test']*len(test_pos) + \
                 ['train']*len(train_neg) + ['val']*len(val_neg) + ['test']*len(test_neg)
        
        print(f"\n=== [III] Line Graph Generation ===")
        print(f"Total Nodes to Process: {len(pos_all) + len(neg_all)}")
        
        # Try to use Parallel Converter if available and appropriate
        try:
             # Use optimized parallel converter
             lg_data = ParallelLineGraphConverter.convert_to_linegraph(base_data.x, pos_all, neg_all, labels, verbose=self.verbose)
        except Exception as e:
             if self.verbose: print(f"Fallback to sequential due to: {e}")
             lg_data = self.converter.convert_to_linegraph(base_data.x, pos_all, neg_all, labels)
             
        print(f"Generated Line Graph: {lg_data.num_nodes} nodes, {lg_data.edge_index.shape[1]} edges")
        
        if pruning_method != 'none':
            print(f"\n=== [IV] Pruning Line Graph ===")
            print(f"Method: {pruning_method}")
            if pruning_method == 'knn':
                print(f"Parameters: k={config['pruning'].get('k')}")
                
            print(f"Edges Before Pruning: {lg_data.edge_index.shape[1]}")
            lg_data = self.pruner.apply_pruning(lg_data, config['pruning'])
            print(f"Edges After Pruning:  {lg_data.edge_index.shape[1]}")
        else:
            print(f"\n=== [IV] Pruning Line Graph ===")
            print("Method: None (Skipping)")
            
        return lg_data