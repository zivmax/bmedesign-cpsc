import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
import numpy as np
from pandas import DataFrame
from tqdm import tqdm
import time
from .dataset import SignalDataset

SEED = 42
np.random.seed(SEED)
class NetCore(nn.Module):
    def __init__(self, input_length, embedding_dim, kernel_sizes, num_filters, drop_out):
        super(NetCore, self).__init__()

        self.conv_layers = nn.ModuleList([
            nn.Conv1d(in_channels=1, out_channels=num_filters, kernel_size=k)
            for k in kernel_sizes
        ])

        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)

        reduced_size = input_length // 16

        self.fc1 = nn.Linear(len(kernel_sizes) * num_filters * reduced_size, 256)
        self.fc_embedding = nn.Linear(256, embedding_dim)

        self.dropout = nn.Dropout(drop_out)
        self.batch_norm1 = nn.BatchNorm1d(len(kernel_sizes) * num_filters)
        self.batch_norm2 = nn.BatchNorm1d(256)

    def forward(self, x):
        conv_results = []
        for conv in self.conv_layers:
            conv_results.append(F.relu(conv(x)))

        x = torch.cat(conv_results, dim=1)
        x = self.batch_norm1(x)
        x = self.pool1(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.batch_norm2(x)
        x = self.dropout(x)

        embedding = self.fc_embedding(x)
        return embedding
    

class Hybrid:
    def __init__(self, 
                 cnn_params,
                 classifier_params,
                 embedding_dim=64, 
                 device='cuda'if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.embedding_dim = embedding_dim
        self.cnn_params = cnn_params
        self.cnn_model = NetCore(**cnn_params).to(device)
        self.classifier_params = classifier_params
        self.xgb_gpu_options = {
            'tree_method': 'gpu_hist',  
            'gpu_id': 0,                
            'predictor': 'gpu_predictor'  
        } if device == 'cuda' else {}
        self.classifier = XGBClassifier(**classifier_params, **self.xgb_gpu_options)


    def train(self, df:DataFrame, 
              target:DataFrame,
              batch_size=64,
              num_epochs=50,
              learning_rate=0.001,
              weight_decay=1e-5,
              early_stopping=10,
              splits=5,
              verbose=True):
        skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=SEED)
        total_fold_results = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        fold_iter = tqdm(enumerate(skf.split(df, target)) , 
                         total=splits, 
                         desc="Folds", 
                         position=0) if verbose else enumerate(skf.split(df, target))

        for fold, (train_idx, val_idx) in fold_iter:

            fold_start = time.time()

            cnn_model = NetCore(**self.cnn_params).to(self.device)
            classifier = XGBClassifier(**self.xgb_gpu_options, **self.classifier_params)
            optimizer = torch.optim.Adam(cnn_model.parameters(),
                                         learning_rate,
                                         weight_decay=weight_decay)

            x_train, y_train = df.iloc[train_idx], target.iloc[train_idx]
            x_val, y_val = df.iloc[val_idx], target.iloc[val_idx]

            train_dataset = SignalDataset(x_train, y_train)
            val_dataset = SignalDataset(x_val, y_val)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            fold_train_losses = []
            fold_train_acc = []
            fold_val_losses = []
            fold_val_acc = []
            best_val_loss = np.inf
            patience_counter = 0

            epoch_iter = tqdm(range(num_epochs), 
                              desc=f"Fold {fold+1} Epochs", 
                              position=1, 
                              leave=False) if verbose else range(num_epochs)
            for epoch in epoch_iter:
                cnn_model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0

                batch_iter = tqdm(train_loader, 
                                  desc='Training', 
                                  position=2, 
                                  leave=False) if verbose else train_loader
                
                for data, label in batch_iter:
                    data, label = data.to(self.device), label.to(self.device)

                    optimizer.zero_grad()
                    embeddings = cnn_model(data.unsqueeze(1))
                    classifier.fit(embeddings.detach().cpu().numpy(), label.cpu().numpy())
                    output = classifier.predict(embeddings.detach().cpu().numpy())
                    loss = F.cross_entropy(embeddings, label)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()
                    train_correct += (output == label.cpu().numpy()).sum()
                    train_total += len(label)

                    batch_iter.set_postfix({
                        'loss': f"{loss.item():.4f}",
                    }) if verbose else None

            train_loss = train_loss / len(train_loader)
            train_accuracy = train_correct / train_total
            fold_train_losses.append(train_loss)
            fold_train_acc.append(train_accuracy)

            cnn_model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            val_batch_iter = tqdm(val_loader, 
                                  desc='Validation', 
                                  position=2, 
                                  leave=False) if verbose else val_loader
            with torch.no_grad():
                for data, label in val_batch_iter:
                    data, label = data.to(self.device), label.to(self.device)
                    embeddings = cnn_model(data.unsqueeze(1))
                    output = classifier.predict(embeddings.detach().cpu().numpy())
                    loss = F.cross_entropy(embeddings, label)

                    val_loss += loss.item()
                    val_correct += (output == label.cpu().numpy()).sum()
                    val_total += len(label)

                val_batch_iter.set_postfix({
                        'loss': f"{loss.item():.4f}",
                    }) if verbose else None
                

            val_loss = val_loss / len(val_loader)
            val_accuracy = val_correct / val_total
            fold_val_losses.append(val_loss)
            fold_val_acc.append(val_accuracy)

            if verbose:
                epoch_iter.set_postfix({
                    'train_loss': f"{train_loss:.4f}",
                    'train_acc': f"{train_accuracy:.4f}",
                    'val_loss': f"{val_loss:.4f}",
                    'val_acc': f"{val_accuracy:.4f}"
                })
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_fold_model = {
                    'model_state': self.model.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_loss,
                    'val_accuracy': val_accuracy
                }
            else:
                patience_counter += 1
                if patience_counter >= early_stopping:
                    if verbose:
                        tqdm.write(f"Early stopping at epoch {epoch+1}")
                    break
        total_fold_results['train_loss'].append(fold_train_losses)
        total_fold_results['train_accuracy'].append(fold_train_acc)
        total_fold_results['val_loss'].append(fold_val_losses)
        total_fold_results['val_accuracy'].append(fold_val_acc)
        total_fold_results['best_models'].append(best_fold_model)
        
        fold_time = time.time() - fold_start
        if verbose:
            tqdm.write(f"Fold {fold+1}/{splits} completed in {fold_time:.2f}s | "
                       f"Best val loss: {best_val_loss:.4f} | "
                       f"Best val accuracy: {best_fold_model['val_accuracy']:.4f}")
            

        best_model_idx = np.argmin([model['val_loss'] for model in total_fold_results['best_models']])
        best_model = total_fold_results['best_models'][best_model_idx]


        self.model.load_state_dict(best_model['model_state'])

        if verbose:
            print(f"\nTraining completed. Best model from fold {best_model_idx+1} with "
                  f"validation loss: {best_model['val_loss']:.4f} and "
                  f"validation accuracy: {best_model['val_accuracy']:.4f}")

        return total_fold_results       

    