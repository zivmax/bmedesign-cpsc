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
from autogluon.tabular import TabularDataset, TabularPredictor

warnings.filterwarnings("ignore")
SEED = 42
PALETTE = 'coolwarm'
ALPHA = 0.5
np.random.seed(SEED)

class CNNCore(nn.Module):
    def __init__(self, input_length, embedding_dim, kernel_sizes=[3, 5, 7], num_filters=32, drop_out=0.2):
        super(CNNCore, self).__init__()

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

class NNClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim1=128, hidden_dim2=64, dropout_rate=0.2):
        super(NNClassifier, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim1)
        self.layer2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.layer3 = nn.Linear(hidden_dim2, hidden_dim2)
        self.layer4 = nn.Linear(hidden_dim2, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim1)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim2)
        self.batch_norm3 = nn.BatchNorm1d(hidden_dim2)
        
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.batch_norm1(x)
        x = self.dropout(x)
        
        x = F.relu(self.layer2(x))
        x = self.batch_norm2(x)
        x = self.dropout(x)

        x = F.relu(self.layer3(x))
        x = self.batch_norm3(x)
        x = self.dropout(x)
        
        x = self.layer4(x)
        return x

class HybridModel:
    def __init__(self, 
                 cnn_params,
                 classifier_params,
                 device='cuda'if torch.cuda.is_available() else 'cpu'):
        self.device = device 
        self.cnn_params = cnn_params
        self.cnn_model = CNNCore(**cnn_params).to(device)
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
              early_stopping=None,
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
        stopping_thershold = np.inf if not early_stopping else early_stopping

        for fold, (train_idx, val_idx) in fold_iter:
            fold_start = time.time()
            cnn_model = CNNCore(**self.cnn_params).to(self.device)
            n_classes = len(np.unique(target.iloc[train_idx]))
            classifier_layer = NNClassifier(self.cnn_params['embedding_dim'], n_classes).to(self.device)
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
                    if patience_counter >= stopping_thershold:
                        if verbose:
                            tqdm.write(f"Fold {fold+1} early stopping at epoch {epoch+1}")
                        break

# ============================ XGB embedding ==================================
            print('==================XGB Training Begin !==================')
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
        
        torch.save(best_model['cnn_state'], f"{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')}_Hybrid_CNN.pth") if model_save_path else None
        self.cnn_model.load_state_dict(best_model['cnn_state'])
        self.classifier = best_model['classifier']
        self.ip = best_model['pipeline']

        self.total_fold_results = total_fold_results
        return total_fold_results
    
    def train_process_plot(self, save=True, val_loss_log=False, filter_coeff=None):
# ============================ train and val loss ============================
        plt.figure(figsize=(16,6))
        total_fold_results = self.total_fold_results
        plt.subplot(1, 2, 1)
        for idx, loss in enumerate(total_fold_results['train_loss']):
            sns.lineplot(x=range(len(loss)), y=loss, label='Fold{} train_loss'.format(idx+1),
                         palette=PALETTE, alpha=ALPHA)
        plt.title('Train Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.subplot(1, 2, 2)
        if val_loss_log:
            for idx, loss in enumerate(total_fold_results['val_loss']):
                if filter_coeff:
                    loss = [l for l in loss if l < np.mean(loss) + filter_coeff * np.std(loss)]
                sns.lineplot(x=range(len(loss)), y=np.log(loss), label='Fold{} val_log_loss'.format(idx+1),
                             palette=PALETTE,alpha=ALPHA)
        else:
            for idx, loss in enumerate(total_fold_results['val_loss']):
                if filter_coeff:
                    loss = [l for l in loss if l < np.mean(loss) + filter_coeff * np.std(loss)]
                sns.lineplot(x=range(len(loss)), y=loss, label='Fold{} val_loss'.format(idx+1),
                             palette=PALETTE,alpha=ALPHA)
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
            plt.savefig(BASE_PATH + r'/hybrid_train_val_loss.png')
        else:
            plt.show()
# ============================= train and val f1 score ============================
        plt.figure(figsize=(16, 6))
        plt.subplot(1, 2, 1)
        for idx, f1 in enumerate(total_fold_results['train_f1']):
            sns.lineplot(x=range(len(f1)), y=f1, label='Fold{} train_f1'.format(idx+1),
                         palette=PALETTE, alpha=ALPHA)
        plt.title('Train F1 Score')
        plt.xlabel('Epochs')
        plt.ylabel('F1 Score')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.subplot(1, 2, 2)
        for idx, f1 in enumerate(total_fold_results['val_f1']):
            sns.lineplot(x=range(len(f1)), y=f1, label='Fold{} val_f1'.format(idx+1),
                         palette=PALETTE, alpha=ALPHA)
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
            plt.savefig(BASE_PATH + r'/hybrid_train_val_f1.png')
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
                                                   cmap=PALETTE)
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
    
    def predict(self, X, batch_size=16):
        self.cnn_model.eval()

        all_embeddings = []
        num_samples = len(X)

        for i in range(0, num_samples, batch_size):
            batch_end = min(i + batch_size, num_samples)
            batch_data = X.iloc[i:batch_end]

            X_tensor = torch.tensor(batch_data.values, dtype=torch.float32).unsqueeze(1).to(self.device)

            with torch.no_grad():
                batch_embeddings = self.cnn_model(X_tensor).cpu().numpy()
                all_embeddings.append(batch_embeddings)

        embeddings = np.vstack(all_embeddings)

        interaction = self.ip.transform(X) if self.ip else ValueError("InteractionPipeline not fitted. Call train() first.")
        interaction_feats = interaction.values

        combined = np.hstack((embeddings, interaction_feats))

        predictions = self.classifier.predict(combined)
        return combined, predictions


class AutoGModel:
    def __init__(self, 
                 cnn_params,
                 autogluon_params=None,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.cnn_params = cnn_params
        self.cnn_model = CNNCore(**cnn_params).to(device)
        self.autogluon_params = autogluon_params or {
            'presets': 'best_quality',
            'time_limit': 1800,
            'feature_generator': 'auto'
        }
        self.predictor = None
        self.ip = None
        
    def train(self, signal_df, target, 
              batch_size=128, 
              num_epochs=100, 
              learning_rate=0.0015,
              weight_decay=1e-5,
              early_stopping=None,
              splits=5,
              verbose=True,
              model_save_path=None):
        skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=SEED)
        total_fold_results = {
            'train_loss': [],
            'train_f1': [],
            'val_loss': [],
            'val_f1': [],
            'best_models': []
        }
        
        stopping_threshold = np.inf if not early_stopping else early_stopping
        
        fold_iter = tqdm(enumerate(skf.split(signal_df, target)), 
                         total=splits, 
                         desc="Folds", 
                         position=0) if verbose else enumerate(skf.split(signal_df, target))
        train_start = time.time()
        for fold, (train_idx, val_idx) in fold_iter:
            fold_start = time.time()
            cnn_model = CNNCore(**self.cnn_params).to(self.device)
            n_classes = len(np.unique(target.iloc[train_idx]))
            classifier_layer = nn.Linear(self.cnn_params['embedding_dim'], n_classes).to(self.device)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(list(cnn_model.parameters()) + list(classifier_layer.parameters()),
                                        learning_rate,
                                        weight_decay=weight_decay)

            X_train, y_train = signal_df.iloc[train_idx], target.iloc[train_idx]
            X_val, y_val = signal_df.iloc[val_idx], target.iloc[val_idx]
            
            ip = InteractionPipeline()
            ip.fit(X_train)
            
            train_dataset = SignalDataset(X_train, y_train)
            val_dataset = SignalDataset(X_val, y_val)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
            fold_train_losses = []
            fold_train_f1 = []
            fold_val_losses = []
            fold_val_f1 = []
            best_val_loss = np.inf
            patience_counter = 0
            best_fold_model = None
            
            epoch_iter = tqdm(range(num_epochs), 
                              desc=f"Fold {fold+1} Epochs", 
                              position=1, 
                              leave=False) if verbose else range(num_epochs)
            
            for epoch in epoch_iter:
                cnn_model.train()
                classifier_layer.train()
                train_loss = 0.0
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
                    train_preds.extend(predicted.cpu().numpy())
                    train_labels_list.extend(label.cpu().numpy())
                    
                    if verbose:
                        batch_iter.set_postfix({'loss': f"{loss.item():.4f}"})
                
                train_loss /= len(train_loader)
                f1_train = f1_score(train_labels_list, train_preds, average='weighted')
                fold_train_losses.append(train_loss)
                fold_train_f1.append(f1_train)
                
                # Validation phase
                cnn_model.eval()
                classifier_layer.eval()
                val_loss = 0.0
                val_preds = []
                val_labels_list = []
                
                val_batch_iter = tqdm(enumerate(val_loader), 
                                      total=len(val_loader),
                                      desc='Validation', 
                                      position=2, 
                                      leave=False) if verbose else enumerate(val_loader)
                
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
                        val_preds.extend(predicted.cpu().numpy())
                        val_labels_list.extend(label.cpu().numpy())
                        
                        if verbose:
                            val_batch_iter.set_postfix({'loss': f"{loss.item():.4f}"})
                
                val_loss /= len(val_loader)
                f1_val = f1_score(val_labels_list, val_preds, average='weighted')
                fold_val_losses.append(val_loss)
                fold_val_f1.append(f1_val)
                
                if verbose:
                    epoch_iter.set_postfix({
                        'train_loss': f"{train_loss:.4f}",
                        'train_f1': f"{f1_train:.4f}",
                        'val_loss': f"{val_loss:.4f}",
                        'val_f1': f"{f1_val:.4f}"
                    })
                
                # Check for early stopping
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
                    if patience_counter >= stopping_threshold:
                        if verbose:
                            tqdm.write(f"Fold {fold+1} early stopping at epoch {epoch+1}")
                        break
            
            best_fold_model['pipeline'] = ip
            total_fold_results['train_loss'].append(fold_train_losses)
            total_fold_results['train_f1'].append(fold_train_f1)
            total_fold_results['val_loss'].append(fold_val_losses)
            total_fold_results['val_f1'].append(fold_val_f1)
            total_fold_results['best_models'].append(best_fold_model)
            
            fold_time = time.time() - fold_start
            if verbose:
                tqdm.write(f"Fold {fold+1}/{splits} completed in {fold_time:.2f}s | "
                          f"Best val loss: {best_val_loss:.4f} | "
                          f"Best model val F1: {best_fold_model['val_f1']:.4f}")
        
        best_model_idx = np.argmax([model['val_f1'] for model in total_fold_results['best_models']])
        best_model = total_fold_results['best_models'][best_model_idx]
        
        if model_save_path:
            torch.save(best_model['cnn_state'], model_save_path + f"{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')}_AutoG_CNN.pth")
            print("==================Best CNN Model Saved !==================")
        
        print('==================AutoG Stage Begin !==================')
        self.cnn_model.load_state_dict(best_model['cnn_state'])
        self.ip = best_model['pipeline']
        self.total_fold_results = total_fold_results
        
        self.cnn_model.eval()
        
        full_dataset = SignalDataset(signal_df, target)
        full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)
        
        embeddings_list = []
        with torch.no_grad():
            for data, _ in tqdm(full_loader, desc="Generating embeddings"):
                data = data.to(self.device)
                batch_embeddings = self.cnn_model(data.squeeze(2)).cpu().numpy()
                embeddings_list.append(batch_embeddings)
        
        embeddings = np.vstack(embeddings_list)
        
        interaction_feats = self.ip.transform(signal_df).values
        
        combined = np.hstack((embeddings, interaction_feats))
        combined_df = pd.DataFrame(combined)
        combined_df['target'] = target.reset_index(drop=True)
        
        auto_path = model_save_path + f"{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')}_AutoG/" if model_save_path else None
        self.predictor = TabularPredictor(label='target',
                                          path=auto_path).fit(
            combined_df,
            **self.autogluon_params
        )
        print('==================AutoG Finished !==================')
        print(f"Total training time: {str(time.time() - train_start)}s" )
        return self.predictor.leaderboard(), self.predictor.model_best
    

    def train_process_plot(self, save=True, val_loss_log=False, filter_coeff=None):
        # ============================ train and val loss ============================
        plt.figure(figsize=(16,6))
        total_fold_results = self.total_fold_results
        plt.subplot(1, 2, 1)
        for idx, loss in enumerate(total_fold_results['train_loss']):
            sns.lineplot(x=range(len(loss)), y=loss, label=f'Fold{idx+1} train_loss',
                         palette=PALETTE, alpha=ALPHA)
        plt.title('Train Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.subplot(1, 2, 2)
        if val_loss_log:
            for idx, loss in enumerate(total_fold_results['val_loss']):
                if filter_coeff:
                    loss = [l for l in loss if l < np.mean(loss) + filter_coeff * np.std(loss)]
                sns.lineplot(x=range(len(loss)), y=np.log(loss), label=f'Fold{idx+1} val_log_loss',
                             palette=PALETTE, alpha=ALPHA)
        else:
            for idx, loss in enumerate(total_fold_results['val_loss']):
                if filter_coeff:
                    loss = [l for l in loss if l < np.mean(loss) + filter_coeff * np.std(loss)]
                sns.lineplot(x=range(len(loss)), y=loss, label=f'Fold{idx+1} val_loss',
                             palette=PALETTE, alpha=ALPHA)
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
            plt.savefig(BASE_PATH + r'/autog_train_val_loss.png')
        else:
            plt.show()
            
        # ============================= train and val f1 score ============================
        plt.figure(figsize=(16, 6))
        plt.subplot(1, 2, 1)
        for idx, f1 in enumerate(total_fold_results['train_f1']):
            sns.lineplot(x=range(len(f1)), y=f1, label=f'Fold{idx+1} train_f1',
                         palette=PALETTE, alpha=ALPHA)
        plt.title('Train F1 Score')
        plt.xlabel('Epochs')
        plt.ylabel('F1 Score')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.subplot(1, 2, 2)
        for idx, f1 in enumerate(total_fold_results['val_f1']):
            sns.lineplot(x=range(len(f1)), y=f1, label=f'Fold{idx+1} val_f1',
                         palette=PALETTE, alpha=ALPHA)
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
            plt.savefig(BASE_PATH + r'/autog_train_val_f1.png')
        else:
            plt.show()

    
    def predict(self, X, batch_size=16):
        self.cnn_model.eval()

        all_embeddings = []
        num_samples = len(X)

        for i in range(0, num_samples, batch_size):
            batch_end = min(i + batch_size, num_samples)
            batch_data = X.iloc[i:batch_end]

            X_tensor = torch.tensor(batch_data.values, dtype=torch.float32).unsqueeze(1).to(self.device)

            with torch.no_grad():
                batch_embeddings = self.cnn_model(X_tensor).cpu().numpy()
                all_embeddings.append(batch_embeddings)

        embeddings = np.vstack(all_embeddings)

        interaction = self.ip.transform(X) if self.ip else ValueError("InteractionPipeline not fitted. Call train() first.")
        interaction_feats = interaction.values

        combined = np.hstack((embeddings, interaction_feats))
        combined_df = pd.DataFrame(combined)

        predictions = self.predictor.predict(combined_df)
        prediction_proba = self.predictor.predict_proba(combined_df)

        return combined_df, predictions, prediction_proba
    
    def evaluate(self, X_test, y_test, plot=True):

        combined_df, predictions, _ = self.predict(X_test)
        combined_df['target'] = y_test.reset_index(drop=True)
        
        print('\n==================AutoG Evaluation Begin !==================\n')
        eval_results = self.predictor.evaluate(combined_df)
        
        if plot:
            plt.figure(figsize=(10, 8))
            y_pred_proba = self.predictor.predict_proba(combined_df.drop('target', axis=1))
            skplot.metrics.plot_precision_recall_curve(y_test, y_pred_proba, 
                                                     title="Precision-Recall Curve",
                                                     cmap=PALETTE)
            plt.grid(True, linestyle='--', alpha=0.7)
            BASE_PATH = r'src/imgs/evaluation/'
            if not os.path.exists(os.path.dirname(BASE_PATH)):
                os.makedirs(os.path.dirname(BASE_PATH))
            plt.savefig(BASE_PATH + r'autog_precision_recall_curve.png')
            plt.close('all')
        print('\n==================AutoG Evaluation Finished !==================\n')
        return eval_results