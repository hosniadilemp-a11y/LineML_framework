"""
LineML GNN Model Definition.

This module defines the SophisticatedLinkPredictor model and its components.
It implements a Residual GraphSAGE encoder with Jumping Knowledge and optional 
Metric Learning for robust link prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, LayerNorm

class EdgeAwareFeatureExtractor(nn.Module):
    """
    Extracts rich features from a pair of node embeddings (u, v).
    
    Generates a combined feature vector using:
    - Concatenation: [h_u || h_v]
    - Hadamard Product: h_u * h_v
    - Absolute Difference: |h_u - h_v|
    
    The components are weighted by learnable scalars.
    """
    def __init__(self, input_dim):
        """
        Initialize the extractor.

        Args:
            input_dim (int): Dimension of input node features. 
                             Note: The output dimension will be 4 * (input_dim / 2).
        """
        super().__init__()
        self.input_dim = input_dim
        self.node_dim = input_dim // 2
        
        # Output dimension: 4 parts (u, v, hadamard, diff) if we consider input is already concat?
        # Based on logic: input x is [h_u, h_v].
        self.output_dim = self.node_dim * 4
        
        # Learnable weights initialized to 1.0
        self.w_concat = nn.Parameter(torch.tensor(1.0))
        self.w_hadamard = nn.Parameter(torch.tensor(1.0))
        self.w_diff = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (Tensor): Input tensor of shape (batch, 2 * node_dim).
        
        Returns:
            Tensor: Enhanced feature components concatenated.
        """
        h_u = x[:, :self.node_dim]
        h_v = x[:, self.node_dim:]
        
        concat = torch.cat([h_u, h_v], dim=-1) * self.w_concat
        hadamard = (h_u * h_v) * self.w_hadamard
        diff = torch.abs(h_u - h_v) * self.w_diff
        
        return torch.cat([concat, hadamard, diff], dim=-1)

class ResidualSAGEEncoder(nn.Module):
    """
    Deep GraphSAGE Encoder with Residual Connections and Jumping Knowledge.
    """
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        """
        Initialize the encoder.

        Args:
            input_dim (int): Input feature dimension.
            hidden_dim (int): Hidden layer dimension.
            num_layers (int): Number of GraphSAGE layers.
            dropout (float): Dropout probability.
        """
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = dropout
        self.num_layers = num_layers
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_norm = LayerNorm(hidden_dim)
        
        for _ in range(num_layers):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.norms.append(LayerNorm(hidden_dim))

    def forward(self, x, edge_index):
        """
        Forward pass.

        Args:
            x (Tensor): Node features.
            edge_index (Tensor): Edge indices.

        Returns:
            Tensor: Concatenated representations from all layers (Jumping Knowledge).
        """
        x = self.input_proj(x)
        x = self.input_norm(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        layer_outputs = [] 
        
        for i in range(self.num_layers):
            x_in = x 
            x = self.convs[i](x, edge_index)
            x = self.norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + x_in # Residual
            layer_outputs.append(x)
            
        # Jumping Knowledge (Concat all layers)
        return torch.cat(layer_outputs, dim=-1)


class SophisticatedLinkPredictor(nn.Module):
    """
    Main Link Prediction Model.
    
    Integrates the encoder, optional metric learning, and an ensemble of classifier heads.
    """
    def __init__(self, input_dim, config):
        """
        Initialize the model.

        Args:
            input_dim (int): Input feature dimension.
            config (dict): Configuration dictionary.
        """
        super().__init__()
        
        hidden_dim = config.get('hidden_dim', 128)
        num_layers = config.get('num_sage_layers', 3)
        dropout = config.get('dropout', 0.3)
        num_heads = config.get('num_classifier_heads', 2)
        output_dim = config.get('output_dim', 2)
        
        # 1. Feature Extractor
        # Using Identity since inputs are already processed (Hadamard features).
        # We assume input_dim matches the expected input for the encoder.
        self.feature_extractor = nn.Identity()
        sage_input_dim = input_dim
        
        # 2. Encoder
        self.encoder = ResidualSAGEEncoder(sage_input_dim, hidden_dim, num_layers, dropout)
        
        # 3. JK Dimension (hidden * layers)
        jk_dim = hidden_dim * num_layers
        
        # 4. Metric Learning 
        self.use_metric_learning = config.get('use_metric_learning', False)
        metric_dim = config.get('metric_projection_dim', 64)
        
        if self.use_metric_learning:
            self.metric_proj = nn.Sequential(
                nn.Linear(jk_dim, metric_dim),
                nn.LayerNorm(metric_dim),
                nn.ReLU(),
                nn.Linear(metric_dim, metric_dim)
            )
            self.margin = config.get('triplet_margin', 0.5)
        
        # 5. Classifier Heads (Ensemble)
        self.classifier_heads = nn.ModuleList()
        for _ in range(num_heads):
            head = nn.Sequential(
                nn.Linear(jk_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, output_dim)
            )
            self.classifier_heads.append(head)
            
    def forward(self, x, edge_index, return_embeddings=False):
        """
        Forward pass.

        Args:
            x (Tensor): Node features.
            edge_index (Tensor): Edge indices.
            return_embeddings (bool, optional): Whether to return embeddings.

        Returns:
            Tensor or Tuple: Logits, or (Logits, Embeddings).
        """
        # 1. Features
        x_rich = self.feature_extractor(x)
        
        # 2. Encode
        x_encoded = self.encoder(x_rich, edge_index)
        
        # 3. Classify (Ensemble)
        logits_list = [head(x_encoded) for head in self.classifier_heads]
        logits = torch.stack(logits_list).mean(dim=0)
        
        # 4. Handle Return Logic 
        if return_embeddings:
            if self.use_metric_learning:
                # Normal path: Project and Normalize
                raw_embeddings = self.metric_proj(x_encoded)
                embeddings = F.normalize(raw_embeddings, p=2, dim=1)
                return logits, embeddings
            else:
                # Fallback path: Return raw encoded features so unpacking works
                return logits, x_encoded
            
        return logits