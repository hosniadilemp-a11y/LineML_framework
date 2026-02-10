# LineML: Scalable Link Prediction via Line Graphs

LineML is a novel framework for scalable and accurate link prediction in large-scale networks. It transforms the link prediction task on a primal graph into a node classification task on its corresponding **Line Graph**. By leveraging graph neural networks (GNNs) on the dual graph representation, LineML effectively captures complex edge-to-edge interactions and higher-order structural patterns.

![LineML Architecture](extra/fig_archi_1.pdf)
*(Note: To view the architecture diagram directly on GitHub, please convert the PDF in `extra/` to a PNG or SVG format.)*

## Key Features

- **Line Graph Transformation**: Converts edge prediction to node classification, enabling more expressive feature learning for links.
- **Scalable Training**: Supports Distributed Data Parallel (DDP) training for large datasets.
- **Advanced Graph Pruning**: helper module (`DataPropocessing/Graphpruner.py`) implements spectral, community-based, and degree-based pruning to manage line graph size.
- **Robust Model**: Uses a Residual GraphSAGE encoder with Jumping Knowledge and optional Metric Learning (`Models/MLGNNmodel.py`).

## Project Structure

```bash
LineMLcode/
├── DataPropocessing/       # Data handling and processing modules
│   ├── linegraph_loader.py # Core logic for Line Graph construction
│   ├── dataloader.py       # PyG DataLoaders for mini-batch training
│   └── Graphpruner.py      # Pruning techniques (Spectral, KNN, etc.)
├── Models/                 # Model definitions
│   └── MLGNNmodel.py       # SophisticatedLinkPredictor architecture
├── utils/                  # Utility scripts
│   └── training.py         # UnifiedTrainer class for training loops
├── extra/                  # Supplementary materials (figures, etc.)
├── run_unified_training.py # Main entry point for training
└── requirements.txt        # Python dependencies
```

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/yourusername/LineML.git
    cd LineML/LineMLcode
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The main training script `run_unified_training.py` supports both single-GPU (simple) and multi-GPU (parallel) execution.

### 1. Simple Mode (Single GPU)
Run the training on a specific GPU (default is GPU 0):

```bash
python run_unified_training.py --mode simple --gpu 0
```

### 2. Parallel Mode (Distributed Data Parallel)
Run distributed training across all available GPUs:

```bash
python run_unified_training.py --mode parallel
```

### Arguments

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--mode` | `simple` | Training mode: `simple` or `parallel` |
| `--gpu` | `0` | GPU ID to use in simple mode |
| `--epochs` | `200` | Number of training epochs |
| `--batch_size` | `512` | Batch size for training |
| `--lr` | `0.0001` | Learning rate |
| `--verbose` | `False` | Enable verbose logging |

## Configuration

You can customize the training configuration in `run_unified_training.py` by modifying the `config` dictionary:

```python
config = {
    'features': 'random',         # Feature initialization ('random', 'original', 'node2vec')
    'pruning': {'method': 'knn', 'k': 5}, # Pruning strategy
    'pos_neg_ratio': 2.0,         # Ratio of negative samples
    'sampling_strategy': 'degree' # Negative sampling strategy
}
```

## License

[MIT License](LICENSE)
