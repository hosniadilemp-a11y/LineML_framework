import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
import os

class UnifiedTrainer:
    """
    Unified trainer class for LineML models.
    
    Handles training, evaluation, and logging for both Single-GPU and 
    Distributed Data Parallel (DDP) modes. Includes Early Stopping.
    """
    def __init__(self, model, config, device, rank=0, is_ddp=False, verbose=False):
        """
        Initialize the trainer.

        Args:
            model (torch.nn.Module): The GNN model to train.
            config (dict): Configuration dictionary containing hyperparameters (lr, weight_decay).
            device (torch.device): Device to run training on.
            rank (int, optional): Process rank for DDP. Defaults to 0.
            is_ddp (bool, optional): Whether training is in DDP mode. Defaults to False.
            verbose (bool, optional): Whether to print progress logs. Defaults to False.
        """
        self.model = model
        self.config = config
        self.device = device
        self.rank = rank
        self.is_ddp = is_ddp
        self.verbose = verbose
        
        self.criterion = nn.CrossEntropyLoss().to(device)
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=config.get('lr', 0.0001), 
            weight_decay=config.get('weight_decay', 5e-4)
        )
        
        if self.is_ddp:
            self.model = DDP(self.model, device_ids=[self.rank], find_unused_parameters=True)
            
    def train_one_epoch(self, loader):
        """
        Run one epoch of training.

        Args:
            loader (DataLoader): Training data loader.

        Returns:
            float: Average loss for the epoch.
        """
        self.model.train()
        total_loss = 0
        steps = 0
        
        for batch in loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            
            logits = self.model(batch.x, batch.edge_index)
            
            batch_size_curr = batch.batch_size
            pred = logits[:batch_size_curr]
            target = batch.y[:batch_size_curr].long()
            
            loss = self.criterion(pred, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            steps += 1
            
        return total_loss / steps if steps > 0 else 0

    def evaluate(self, loader):
        """
        Evaluate the model on a dataset.

        Args:
            loader (DataLoader): Validation or Test data loader.

        Returns:
            tuple: (AUC, Accuracy)
        """
        self.model.eval()
        preds = []
        targets = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                logits = self.model(batch.x, batch.edge_index)
                probs = torch.softmax(logits[:batch.batch_size], dim=1)[:, 1]
                preds.append(probs.cpu())
                targets.append(batch.y[:batch.batch_size].cpu())
        
        y_scores = torch.cat(preds).numpy()
        y_true = torch.cat(targets).numpy()
        y_pred = (y_scores > 0.5).astype(int)
        
        auc = roc_auc_score(y_true, y_scores)
        acc = accuracy_score(y_true, y_pred)
        return auc, acc

    def run(self, train_loader, val_loader=None, test_loader=None, epochs=50, patience=10):
        """
        Run the full training loop with Early Stopping.

        Args:
            train_loader (DataLoader): Training data loader.
            val_loader (DataLoader, optional): Validation loader for early stopping.
            test_loader (DataLoader, optional): Test loader for final evaluation.
            epochs (int, optional): Maximum number of epochs. Defaults to 50.
            patience (int, optional): Early stopping patience. Defaults to 10.

        Returns:
            tuple: (Final AUC, Final Accuracy, Total Epochs)
        """
        best_auc = 0.0
        patience_counter = 0
        
        # tqdm for rank 0
        if self.rank == 0:
            iterator = tqdm(range(epochs), desc="Training", unit="epoch", disable=not self.verbose)
        else:
            iterator = range(epochs)
            
        for epoch in iterator:
            if self.is_ddp and hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(epoch)
                
            loss = self.train_one_epoch(train_loader)
            
            val_acc = 0.0
            val_auc = 0.0
            
            if self.rank == 0:
                if val_loader:
                    val_auc, val_acc = self.evaluate(val_loader)
                    
                    # Early Stopping Check
                    if val_auc > best_auc:
                        best_auc = val_auc
                        patience_counter = 0
                        # Save best model logic could involve a callback or path
                    else:
                        patience_counter += 1
                
                # Update progress bar
                if self.verbose:
                    iterator.set_postfix(loss=f"{loss:.4f}", auc=f"{val_auc:.4f}", acc=f"{val_acc:.4f}", pat=patience_counter)
                else:
                    if (epoch + 1) % 10 == 0:
                         print(f"Epoch {epoch+1}/{epochs} | Loss: {loss:.4f} | Val AUC: {val_auc:.4f}")

            # Broadcast early stopping decision
            if self.is_ddp:
                stop_signal = torch.tensor([1.0 if patience_counter >= patience else 0.0], device=self.device)
                dist.all_reduce(stop_signal, op=dist.ReduceOp.MAX)
                if stop_signal.item() > 0.5:
                    if self.rank == 0 and self.verbose:
                        print(f"\nEarly stopping triggered after {epoch+1} epochs.")
                    break
            elif patience_counter >= patience:
                if self.verbose: print(f"\nEarly stopping triggered after {epoch+1} epochs.")
                break
                
        # Final Evaluation
        if self.rank == 0 and test_loader:
            if self.verbose: print("Evaluating on Test Set...")
            auc, acc = self.evaluate(test_loader)
            return auc, acc, epoch + 1
            
        return 0, 0, epoch + 1

    @staticmethod
    def setup_ddp(rank, world_size):
        """Initializes the distributed process group."""
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    @staticmethod
    def cleanup_ddp():
        """Destroys the distributed process group."""
        dist.destroy_process_group()
