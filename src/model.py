import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import time
import datetime
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from pandas import DataFrame
from tqdm import tqdm
from sklearn.utils import shuffle
import warnings          
import matplotlib.pyplot as plt
import seaborn as sns
import scikitplot as skplot
import pandas as pd
import os
from dataset import SignalDataset
from pipeline import InteractionPipeline

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)

class NetCore(nn.Module):
    def __init__(self, input_length, embedding_dim, kernel_sizes, num_filters, drop_out):
        super(NetCore, self).__init__()

        self.conv_layers = nn.ModuleList([
            nn.Conv1d(in_channels=1, 
                      out_channels=num_filters, 
                      kernel_size=k,
                      padding=(k-1)//2)
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

class HybridModel:
    def __init__(self, 
                 cnn_params,
                 classifier_params,
                 device='cuda'if torch.cuda.is_available() else 'cpu'):
        self.device = device 
        self.cnn_params = cnn_params
        self.cnn_model = NetCore(**cnn_params).to(device)
        self.classifier_params = classifier_params
        self.xgb_gpu_options = {
            'tree_method': 'hist',
            'device': 'cuda' 
        } if device == 'cuda' else {'tree_method': 'approx'}

    def train(self, signal_df:DataFrame,
              target:DataFrame,
              batch_size=32,
              num_epochs=200,
              learning_rate=0.0015,
              weight_decay=1e-5,
              early_stopping=50,
              splits=5,
              verbose=True,
              model_save_path=r'src\model\best_cnn_model_{}.pth'.format(
                  datetime.datetime.now().strftime("%Y.%m.%d_%H.%M"))):
        skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=SEED)
        total_fold_results = {
            'train_loss': [],
            'train_accuracy': [],
            'train_f1': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_f1': [],
            'best_models': []
        }
        fold_iter = tqdm(enumerate(skf.split(signal_df, target)) , 
                         total=splits, 
                         desc="Folds", 
                         position=0) if verbose else enumerate(skf.split(signal_df, target))

        for fold, (train_idx, val_idx) in fold_iter:
            fold_start = time.time()
            cnn_model = NetCore(**self.cnn_params).to(self.device)
            n_classes = len(np.unique(target.iloc[train_idx]))
            classifier_layer = nn.Linear(self.cnn_params['embedding_dim'], n_classes).to(self.device)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(list(cnn_model.parameters()) + list(classifier_layer.parameters()),
                                         learning_rate,
                                         weight_decay=weight_decay)

# =============================== preprocessing ==================================
            X_train, y_train = signal_df.iloc[train_idx], target.iloc[train_idx]
            X_val, y_val = signal_df.iloc[val_idx], target.iloc[val_idx]

            # Fix: same scaling for train and val
            ip = InteractionPipeline()
            ip.fit(X_train)

            train_interaction = ip.transform(X_train)
            val_interaction = ip.transform(X_val)

            train_dataset = SignalDataset(X_train, y_train)
            val_dataset = SignalDataset(X_val, y_val)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

            fold_train_losses = []
            fold_train_f1 = []
            fold_train_accuracy = []
            fold_val_accuracy = []
            fold_val_losses = []
            fold_val_f1 = []
            best_val_loss = np.inf
            patience_counter = 0
            best_fold_model = None

            epoch_iter = tqdm(range(num_epochs), 
                              desc=f"Fold {fold+1} Epochs", 
                              position=1, 
                              leave=False) if verbose else range(num_epochs)
# ============================= epoch training ====================================

            for epoch in epoch_iter:
                cnn_model.train()
                classifier_layer.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0
                train_preds = []
                train_labels_list = []
                
                batch_iter = tqdm(enumerate(train_loader), 
                                  total=len(train_loader),
                                  desc='Training', 
                                  position=2, 
                                  leave=False) if verbose else enumerate(train_loader)
                
                for batch_idx, (data, label) in batch_iter:
                    data, label = data.to(self.device), label.to(self.device)
                    optimizer.zero_grad()

                    embeddings = cnn_model(data.squeeze(2))
                    logits = classifier_layer(embeddings)
                    if label.dim() > 1:
                        label = label.squeeze(1)  # Ensure label is 1D for CrossEntropyLoss
                    loss = criterion(logits, label)
                    loss.backward()
                    optimizer.step()


                    train_loss += loss.item()
                    _, predicted = torch.max(logits.data, 1)
                    train_total += label.size(0)
                    train_correct += (predicted == label).sum().item()
                    train_preds.extend(predicted.cpu().numpy())
                    train_labels_list.extend(label.cpu().numpy())
                    
                    if verbose:
                        batch_iter.set_postfix({'loss': f"{loss.item():.4f}"})

                train_loss /= len(train_loader)
                train_accuracy = train_correct / train_total
                f1_train = f1_score(train_labels_list, train_preds, average='weighted')
                fold_train_losses.append(train_loss)
                fold_train_f1.append(f1_train)
                fold_train_accuracy.append(train_accuracy)


                cnn_model.eval()
                classifier_layer.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                val_preds = []
                val_labels_list = []
                
                val_batch_iter = tqdm(enumerate(val_loader), 
                                      total=len(val_loader),
                                      desc='Validation', 
                                      position=2, 
                                      leave=False) if verbose else enumerate(val_loader)
# ============================= validation ========================================
                with torch.no_grad():
                    for batch_idx, (data, label) in val_batch_iter:
                        data, label = data.to(self.device), label.to(self.device)
                        embeddings = cnn_model(data.squeeze(2))

                        if label.dim() > 1:
                            label = label.squeeze(1)

                        logits = classifier_layer(embeddings)
                        loss = criterion(logits, label)
                        val_loss += loss.item()
                        _, predicted = torch.max(logits.data, 1)
                        val_total += label.size(0)
                        val_correct += (predicted == label).sum().item()
                        val_preds.extend(predicted.cpu().numpy())
                        val_labels_list.extend(label.cpu().numpy())
                        if verbose:
                            val_batch_iter.set_postfix({'loss': f"{loss.item():.4f}"})

                val_loss /= len(val_loader)
                val_accuracy = val_correct / val_total
                f1_val = f1_score(val_labels_list, val_preds, average='weighted')
                fold_val_losses.append(val_loss)
                fold_val_f1.append(f1_val)
                fold_val_accuracy.append(val_accuracy)


                if verbose:
                    epoch_iter.set_postfix({
                        'train_loss': f"{train_loss:.4f}",
                        'train_f1': f"{f1_train:.4f}",
                        'val_loss': f"{val_loss:.4f}",
                        'val_f1': f"{f1_val:.4f}"
                    })
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_fold_model = {
                        'cnn_state': cnn_model.state_dict(),
                        'classifier_layer': classifier_layer.state_dict(),
                        'epoch': epoch,
                        'val_loss': val_loss,
                        'val_f1': f1_val
                    }
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping:
                        if verbose:
                            tqdm.write(f"Early stopping at epoch {epoch+1}")
                        break

# ============================ XGB embedding ==================================
            cnn_model.load_state_dict(best_fold_model['cnn_state'])
            classifier_layer.load_state_dict(best_fold_model['classifier_layer'])
            cnn_model.eval()
            classifier_layer.eval()

            # train_dataset_full = SignalDataset(X_train, y_train)
            # val_dataset_full = SignalDataset(X_val, y_val)
            # train_loader_full = DataLoader(train_dataset_full, batch_size=batch_size, shuffle=False)
            # val_loader_full = DataLoader(val_dataset_full, batch_size=batch_size, shuffle=False)

            train_embeddings = []
            train_labels_all = []
            for data, label in train_loader:
                data = data.to(self.device)
                with torch.no_grad():
                    e = cnn_model(data.squeeze(2)).cpu().numpy()
                train_embeddings.append(e)
                train_labels_all.append(label.numpy())

            train_embeddings = np.vstack(train_embeddings)
            train_labels_all = np.concatenate(train_labels_all)
            train_feats = train_interaction.values
            train_combined = np.hstack((train_embeddings, train_feats))
            classifier = XGBClassifier(**self.classifier_params,
                                       **self.xgb_gpu_options,
                                       random_state=SEED)
            classifier.fit(train_combined, train_labels_all)

            val_embeddings = []
            val_labels_all = []
            for data, label in val_loader:
                data = data.to(self.device)
                with torch.no_grad():
                    e = cnn_model(data.squeeze(2)).cpu().numpy()
                val_embeddings.append(e)
                val_labels_all.append(label.numpy())
            val_embeddings = np.vstack(val_embeddings)
            val_labels_all = np.concatenate(val_labels_all)
            val_feats = val_interaction.values
            val_combined = np.hstack((val_embeddings, val_feats))
            val_preds = classifier.predict(val_combined)
            val_xgb_f1 = f1_score(val_labels_all, val_preds, average='weighted')

            total_fold_results['train_loss'].append(fold_train_losses)
            total_fold_results['train_accuracy'].append(fold_train_accuracy)

            
            total_fold_results['train_f1'].append(fold_train_f1)
            total_fold_results['val_accuracy'].append(fold_val_accuracy)
            
            total_fold_results['val_loss'].append(fold_val_losses)
            total_fold_results['val_f1'].append(fold_val_f1)
            best_fold_model['classifier'] = classifier
            best_fold_model['val_xgb_f1'] = val_xgb_f1
            best_fold_model['pipeline'] = ip
            total_fold_results['best_models'].append(best_fold_model)
            
            fold_time = time.time() - fold_start
            if verbose:
                tqdm.write(f"Fold {fold+1}/{splits} completed in {fold_time:.2f}s | "
                           f"Mean train accuracy: {np.mean(fold_train_accuracy):.3f} | "
                           f"Mean train F1: {np.mean(fold_train_f1):.3f} | "
                            f"Mean val accuracy: {np.mean(fold_val_accuracy):.3f} | "
                           f"Mean val F1: {np.mean(fold_val_f1):.3f}")

# ============================ Model selection ==================================
        best_model_idx = np.argmax([model['val_xgb_f1'] for model in total_fold_results['best_models']])
        best_model = total_fold_results['best_models'][best_model_idx]
        
        torch.save(best_model['cnn_state'], model_save_path) if model_save_path else None
        self.cnn_model.load_state_dict(best_model['cnn_state'])
        self.classifier = best_model['classifier']
        self.ip = best_model['pipeline']

        self.total_fold_results = total_fold_results
        return total_fold_results
    
    def train_process_plot(self, save=True, val_loss_log=False):
# ============================ train and val loss ============================
        plt.figure(figsize=(12, 12))
        total_fold_results = self.total_fold_results
        plt.subplot(1, 2, 1)
        for idx, loss in enumerate(total_fold_results['train_loss']):
            sns.lineplot(x=range(len(loss)), y=loss, label='Fold{} train_loss'.format(idx+1))
        plt.title('Train Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.subplot(1, 2, 2)
        if val_loss_log:
            for idx, loss in enumerate(total_fold_results['val_loss']):
                sns.lineplot(x=range(len(loss)), y=np.log(loss), label='Fold{} val_log_loss'.format(idx+1))
        else:
            for idx, loss in enumerate(total_fold_results['val_loss']):
                sns.lineplot(x=range(len(loss)), y=loss, label='Fold{} val_loss'.format(idx+1))
        plt.title('Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.grid(True, linestyle='--', alpha=0.7)
        if save:
            BASE_PATH = r"src/imgs/training"
            if not os.path.exists(BASE_PATH):
                os.makedirs(BASE_PATH)
            plt.savefig(BASE_PATH + r'/train_val_loss.png')
        else:
            plt.show()
# ============================= train and val f1 score ============================
        plt.figure(figsize=(12, 12))
        plt.subplot(1, 2, 1)
        for idx, f1 in enumerate(total_fold_results['train_f1']):
            sns.lineplot(x=range(len(f1)), y=f1, label='Fold{} train_f1'.format(idx+1))
        plt.title('Train F1 Score')
        plt.xlabel('Epochs')
        plt.ylabel('F1 Score')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.subplot(1, 2, 2)
        for idx, f1 in enumerate(total_fold_results['val_f1']):
            sns.lineplot(x=range(len(f1)), y=f1, label='Fold{} val_f1'.format(idx+1))
        plt.title('Validation F1 Score')
        plt.xlabel('Epochs')
        plt.ylabel('F1 Score')
        plt.legend()
        plt.tight_layout()
        plt.grid(True, linestyle='--', alpha=0.7)
        if save:
            BASE_PATH = r"src/imgs/training"
            if not os.path.exists(BASE_PATH):
                os.makedirs(BASE_PATH)
            plt.savefig(BASE_PATH + r'/train_val_f1.png')
        else:
            plt.show()
        
    def evaluate(self, X_test, y_test, batch_size=32, plot=True):

        self.cnn_model.eval()

        test_dataset = SignalDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        
        test_interaction = self.ip.transform(X_test) if self.ip else ValueError("InteractionPipeline not fitted. Call train() first.")

        test_embeddings = []
        test_labels = []

        with torch.no_grad():
            for data, label in test_loader:
                data = data.to(self.device)
                embeddings = self.cnn_model(data.squeeze(2)).cpu().numpy()
                test_embeddings.append(embeddings)
                test_labels.append(label.numpy())

        test_embeddings = np.vstack(test_embeddings)
        test_labels = np.concatenate(test_labels)

        test_feats = test_interaction.values
        test_combined = np.hstack((test_embeddings, test_feats))

        test_preds_proba = self.classifier.predict_proba(test_combined) 
        test_preds = self.classifier.predict(test_combined)

        # NOTE: test the evaluation here
        # test_preds = np.zeros_like(test_preds) 

        if plot:
            plt.figure(figsize=(10, 8))
            skplot.metrics.plot_precision_recall_curve(y_test, test_preds_proba, 
                                                   title="Precision-Recall Curve",
                                                   cmap='viridis')
            plt.grid(True, linestyle='--', alpha=0.7)
            BASE_PATH = r'src/imgs/evaluation/'
            if not os.path.exists(os.path.dirname(BASE_PATH)):
                os.makedirs(os.path.dirname(BASE_PATH))
            plt.savefig(BASE_PATH + r'precision_recall_curve.png')
            plt.close('all')
            
        return {
            'f1': f1_score(test_labels, test_preds, average='weighted'),
            'predictions': test_preds,
            'true_labels': test_labels
        }
    
    def predict(self, X):
        self.cnn_model.eval()
        
        # Convert input data to tensor
        X_tensor = torch.tensor(X.values, dtype=torch.float32).unsqueeze(1).to(self.device)
        
        # Process all data at once
        with torch.no_grad():
            embeddings = self.cnn_model(X_tensor).cpu().numpy()
            
        # Process with interaction pipeline
        interaction = self.ip.transform(X) if self.ip else ValueError("InteractionPipeline not fitted. Call train() first.")
        interaction_feats = interaction.values
        
        # Combine CNN embeddings with interaction features
        combined = np.hstack((embeddings, interaction_feats))
        
        # Make predictions
        predictions = self.classifier.predict(combined)
        return predictions


