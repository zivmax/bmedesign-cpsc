import multiprocessing
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import time
import datetime
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from pandas import DataFrame, Series # Added Series
import pandas as pd # Added pandas
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib # For saving IP pipeline

# Assuming these are in src.utils or similar, adjust path if needed
from utils.dataset import SignalDataset
from utils.pipeline import InteractionPipeline


warnings.filterwarnings("ignore")
SEED = 42
PALETTE = 'coolwarm'
ALPHA = 0.5

# CNNCore and NNClassifier class definitions remain unchanged
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

# Worker function for multiprocessing
def _train_fold_worker(args):
    # Unpack arguments
    (fold_num, train_idx, val_idx, signal_df_numpy, target_numpy, 
     cnn_params_dict, classifier_params_dict, nn_classifier_input_dim, num_classes,
     common_train_params, device_str) = args

    # Seed for this process (optional, but good for reproducibility if randomness is involved in model init or data loading)
    np.random.seed(SEED + fold_num)
    torch.manual_seed(SEED + fold_num)
    if device_str != 'cpu':
        torch.cuda.manual_seed_all(SEED + fold_num)

    # Convert numpy arrays back to DataFrames/Series
    # Assuming signal_df columns are just numerical indices if not passed
    signal_df_fold = pd.DataFrame(signal_df_numpy) 
    target_fold = pd.Series(target_numpy)
    
    device = torch.device(device_str)
    if common_train_params.get('verbose', False):
        print(f"[Fold {fold_num+1}] Starting on device: {device_str}")

    cnn_model = CNNCore(**cnn_params_dict).to(device)
    # Ensure num_classes is correctly determined (e.g., 2 for binary, or len(np.unique(target_fold)) if multi-class)
    classifier_layer = NNClassifier(nn_classifier_input_dim, num_classes).to(device) 
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(list(cnn_model.parameters()) + list(classifier_layer.parameters()),
                                 common_train_params['learning_rate'],
                                 weight_decay=common_train_params['weight_decay'])

    X_train_fold, y_train_fold = signal_df_fold.iloc[train_idx], target_fold.iloc[train_idx]
    X_val_fold, y_val_fold = signal_df_fold.iloc[val_idx], target_fold.iloc[val_idx]

    ip = InteractionPipeline()
    ip.fit(X_train_fold)
    train_interaction = ip.transform(X_train_fold)
    val_interaction = ip.transform(X_val_fold)

    train_dataset = SignalDataset(X_train_fold, y_train_fold)
    val_dataset = SignalDataset(X_val_fold, y_val_fold)
    train_loader = DataLoader(train_dataset, batch_size=common_train_params['batch_size'], shuffle=True, num_workers=0) # num_workers=0 for simplicity in multiprocessing
    val_loader = DataLoader(val_dataset, batch_size=common_train_params['batch_size'], shuffle=True, num_workers=0)

    fold_train_losses, fold_train_f1s, fold_train_accuracies = [], [], []
    fold_val_losses, fold_val_f1s, fold_val_accuracies = [], [], []
    best_val_loss_nn = np.inf
    patience_counter = 0
    best_fold_nn_model_state = None # For NN part

    # Epoch loop for NN training
    for epoch in range(common_train_params['num_epochs']):
        cnn_model.train()
        classifier_layer.train()
        epoch_train_loss, epoch_train_correct, epoch_train_total = 0.0, 0, 0
        epoch_train_preds_list, epoch_train_labels_list = [], []

        for data, label in train_loader:
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            embeddings = cnn_model(data.squeeze(2))
            logits = classifier_layer(embeddings)
            if label.dim() > 1: label = label.squeeze(1)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            epoch_train_total += label.size(0)
            epoch_train_correct += (predicted == label).sum().item()
            epoch_train_preds_list.extend(predicted.cpu().numpy())
            epoch_train_labels_list.extend(label.cpu().numpy())
        
        avg_epoch_train_loss = epoch_train_loss / len(train_loader) if len(train_loader) > 0 else 0
        epoch_train_accuracy = epoch_train_correct / epoch_train_total if epoch_train_total > 0 else 0
        epoch_train_f1 = f1_score(epoch_train_labels_list, epoch_train_preds_list, average='weighted', zero_division=0)
        fold_train_losses.append(avg_epoch_train_loss)
        fold_train_f1s.append(epoch_train_f1)
        fold_train_accuracies.append(epoch_train_accuracy)

        # Validation for NN
        cnn_model.eval()
        classifier_layer.eval()
        epoch_val_loss, epoch_val_correct, epoch_val_total = 0.0, 0, 0
        epoch_val_preds_list, epoch_val_labels_list = [], []
        with torch.no_grad():
            for data, label in val_loader:
                data, label = data.to(device), label.to(device)
                embeddings = cnn_model(data.squeeze(2))
                if label.dim() > 1: label = label.squeeze(1)
                logits = classifier_layer(embeddings)
                loss = criterion(logits, label)
                epoch_val_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                epoch_val_total += label.size(0)
                epoch_val_correct += (predicted == label).sum().item()
                epoch_val_preds_list.extend(predicted.cpu().numpy())
                epoch_val_labels_list.extend(label.cpu().numpy())

        avg_epoch_val_loss = epoch_val_loss / len(val_loader) if len(val_loader) > 0 else float('inf')
        epoch_val_accuracy = epoch_val_correct / epoch_val_total if epoch_val_total > 0 else 0
        epoch_val_f1 = f1_score(epoch_val_labels_list, epoch_val_preds_list, average='weighted', zero_division=0)
        fold_val_losses.append(avg_epoch_val_loss)
        fold_val_f1s.append(epoch_val_f1)
        fold_val_accuracies.append(epoch_val_accuracy)
        
        if common_train_params.get('verbose_epoch', False): # Control epoch verbosity
             print(f"[Fold {fold_num+1} Epoch {epoch+1}] Train Loss: {avg_epoch_train_loss:.4f}, Val Loss: {avg_epoch_val_loss:.4f}, Val F1 (NN): {epoch_val_f1:.4f}")

        if avg_epoch_val_loss < best_val_loss_nn:
            best_val_loss_nn = avg_epoch_val_loss
            patience_counter = 0
            best_fold_nn_model_state = {
                'cnn_state': {k: v.cpu() for k, v in cnn_model.state_dict().items()},
                'classifier_layer_state': {k: v.cpu() for k, v in classifier_layer.state_dict().items()},
                'epoch': epoch,
                'val_loss_nn': avg_epoch_val_loss,
                'val_f1_nn': epoch_val_f1,
            }
        else:
            patience_counter += 1
            if patience_counter >= common_train_params['early_stopping']:
                if common_train_params.get('verbose', False):
                    print(f"Fold {fold_num+1} NN early stopping at epoch {epoch+1} on device {device_str}")
                break
    
    # XGBoost part
    final_fold_model_details = {'val_xgb_f1': -1} # Default for failed fold
    if best_fold_nn_model_state:
        cnn_model.load_state_dict(best_fold_nn_model_state['cnn_state'])
        # classifier_layer.load_state_dict(best_fold_nn_model_state['classifier_layer_state']) # Optional
        cnn_model.eval()

        train_embeddings_list = []
        for data, _ in train_loader:
            data = data.to(device)
            with torch.no_grad(): e = cnn_model(data.squeeze(2)).cpu().numpy()
            train_embeddings_list.append(e)
        train_embeddings = np.vstack(train_embeddings_list)
        train_combined = np.hstack((train_embeddings, train_interaction.values))
        
        xgb_gpu_options = {}
        if device_str != 'cpu': # Assumes device_str is 'cuda:X'
            xgb_gpu_options = {'tree_method': 'hist', 'device': device_str}
        else:
            # Ensure 'approx' is a valid CPU method or use 'hist' if available for CPU too
            xgb_gpu_options = {'tree_method': 'hist'} # 'hist' is generally good for CPU too

        classifier_xgb = XGBClassifier(**classifier_params_dict, **xgb_gpu_options, random_state=SEED + fold_num)
        classifier_xgb.fit(train_combined, y_train_fold)

        val_embeddings_list = []
        for data, _ in val_loader:
            data = data.to(device)
            with torch.no_grad(): e = cnn_model(data.squeeze(2)).cpu().numpy()
            val_embeddings_list.append(e)
        val_embeddings = np.vstack(val_embeddings_list)
        val_combined = np.hstack((val_embeddings, val_interaction.values))
        
        val_preds_xgb = classifier_xgb.predict(val_combined)
        val_xgb_f1 = f1_score(y_val_fold, val_preds_xgb, average='weighted', zero_division=0)

        final_fold_model_details = best_fold_nn_model_state.copy() # Start with NN model details
        final_fold_model_details['classifier_xgb_state'] = classifier_xgb # Store XGBoost model object
        final_fold_model_details['val_xgb_f1'] = val_xgb_f1
        final_fold_model_details['pipeline_state'] = ip # Store fitted InteractionPipeline
    
    if common_train_params.get('verbose', False):
        val_f1_nn_print = best_fold_nn_model_state.get('val_f1_nn', -1) if best_fold_nn_model_state else -1
        print(f"[Fold {fold_num+1}] Finished on {device_str}. Val F1 (NN): {val_f1_nn_print:.4f}, Val F1 (XGB): {final_fold_model_details.get('val_xgb_f1', -1):.4f}")

    return {
        'fold_num': fold_num,
        'train_losses': fold_train_losses, 'train_f1s': fold_train_f1s, 'train_accuracies': fold_train_accuracies,
        'val_losses': fold_val_losses, 'val_f1s': fold_val_f1s, 'val_accuracies': fold_val_accuracies, # These are from NN part
        'best_model_details': final_fold_model_details, # Contains all states and metrics for this fold
        'device_used': device_str
    }


class HybridModel:
    def __init__(self, 
                 cnn_params,
                 classifier_params,
                 device=None): # Allow device to be None initially
        
        resolved_device_str = 'cpu' # Default to CPU
        if device is None: # If no device preference, try CUDA, then CPU
            if torch.cuda.is_available():
                try:
                    # Try a quick CUDA operation to check if it's truly available and not busy
                    torch.zeros(1).to('cuda') # Small tensor to test CUDA
                    resolved_device_str = 'cuda'
                    print("CUDA is available. Main HybridModel instance will prefer CUDA if not overridden by training.")
                except RuntimeError as e:
                    print(f"CUDA available but got runtime error during test: {e}. Defaulting main instance to CPU.")
                    # resolved_device_str remains 'cpu'
            else:
                print("CUDA not available. Main HybridModel instance will use CPU.")
                # resolved_device_str remains 'cpu'
        elif device == 'cuda' and not torch.cuda.is_available():
            print("Warning: Requested 'cuda' but torch.cuda.is_available() is False. Main instance falling back to CPU.")
            # resolved_device_str remains 'cpu'
        elif device == 'cuda': # Requested 'cuda' and it might be available
            try:
                torch.zeros(1).to('cuda')
                resolved_device_str = 'cuda'
                print("CUDA is available as requested. Main HybridModel instance set to CUDA.")
            except RuntimeError as e:
                print(f"Requested 'cuda', but got runtime error: {e}. Main instance falling back to CPU.")
                # resolved_device_str remains 'cpu'
        else: # Explicitly 'cpu' or some other device string
            resolved_device_str = device
            print(f"Main HybridModel instance device explicitly set to: {resolved_device_str}")

        self.device = resolved_device_str
        self.cnn_params = cnn_params
        
        # Initialize cnn_model on the resolved device
        try:
            self.cnn_model = CNNCore(**cnn_params).to(self.device)
        except RuntimeError as e:
            if "CUDA-capable device(s) is/are busy or unavailable" in str(e) and self.device.startswith("cuda"):
                print(f"CUDA error during CNNCore initialization on {self.device}. Attempting to initialize on CPU instead for the main model instance.")
                self.device = "cpu" # Fallback to CPU for the main instance
                self.cnn_model = CNNCore(**cnn_params).to(self.device)
                print(f"Main HybridModel CNNCore successfully initialized on CPU after CUDA failure.")
            else:
                raise # Re-raise if it's a different error or not a CUDA busy error

        self.classifier_params = classifier_params
        # self.xgb_gpu_options is not directly used here for self.classifier init,
        # as XGBoost is instantiated in the worker.
        self.fold_models = []
        self.fold_pipelines = []
        self.model_save_path = None
        self.classifier = None 
        self.ip = None
        self.total_fold_results = None


    def train(self, signal_df: DataFrame,
              target: Series, # Changed to Series for consistency
              batch_size=16,
              num_epochs=300,
              learning_rate=0.0005,
              weight_decay=1e-5,
              early_stopping=10,
              splits=5,
              verbose=True, # Overall verbosity
              verbose_epoch=False, # Per-epoch verbosity within folds
              model_save_path=None,
              gpu_ids: list[int] = None):

        if model_save_path is None:
            default_save_dir = os.path.join('src', 'model_weights')
            if not os.path.exists(default_save_dir): os.makedirs(default_save_dir)
            model_save_path = os.path.join(default_save_dir, f'hybrid_model_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}')
        self.model_save_path = model_save_path # Base path for saving components

        if gpu_ids and torch.cuda.is_available():
            valid_gpu_ids = [gid for gid in gpu_ids if gid < torch.cuda.device_count()]
            if not valid_gpu_ids:
                print("Warning: Specified GPU IDs are invalid or no GPUs available. Falling back to CPU.")
                available_devices_str = ['cpu']
            else:
                available_devices_str = [f'cuda:{gid}' for gid in valid_gpu_ids]
        elif torch.cuda.is_available() and not gpu_ids: # Use first available GPU if none specified
             available_devices_str = [f'cuda:0']
             if verbose: print(f"No GPU IDs specified, using default: {available_devices_str[0]}")
        else: # CPU
            available_devices_str = ['cpu']
            if gpu_ids and verbose: print("Warning: GPUs specified but torch.cuda.is_available() is False. Using CPU.")
        
        # Determine the number of parallel processes
        # If fewer devices than folds, devices will be reused by workers.
        # If more devices than folds, only 'splits' number of processes will be created if num_parallel_processes is capped by splits.
        # Or, if you want to use all specified devices up to 'splits' folds:
        num_parallel_processes = min(len(available_devices_str), splits) 
        
        if verbose:
            print(f"Starting training with {splits} folds.")
            # The actual devices used by workers will cycle through available_devices_str
            print(f"Will use up to {num_parallel_processes} parallel processes. Devices available for workers: {available_devices_str}")


        skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=SEED)
        
        signal_df_numpy = signal_df.to_numpy() 
        target_numpy = target.to_numpy()

        nn_classifier_input_dim = self.cnn_params['embedding_dim'] 
        num_classes = len(np.unique(target_numpy))

        common_train_params = {
            'batch_size': batch_size, 'num_epochs': num_epochs, 
            'learning_rate': learning_rate, 'weight_decay': weight_decay,
            'early_stopping': early_stopping, 'verbose': verbose, 'verbose_epoch': verbose_epoch
        }

        fold_worker_args = []
        for i, (train_idx, val_idx) in enumerate(skf.split(signal_df_numpy, target_numpy)): # Use numpy for skf.split
            # Assign device to fold worker, cycling through available_devices_str
            device_for_fold = available_devices_str[i % len(available_devices_str)] 
            args_for_fold = (
                i, train_idx, val_idx, signal_df_numpy, target_numpy,
                self.cnn_params, self.classifier_params, 
                nn_classifier_input_dim, num_classes,
                common_train_params, device_for_fold
            )
            fold_worker_args.append(args_for_fold)
        
        # Ensure multiprocessing context is appropriate, especially for CUDA. 'spawn' is often safer.
        ctx = multiprocessing.get_context('spawn')
        # Cap pool processes by num_parallel_processes, which is min(len(available_devices_str), splits)
        with ctx.Pool(processes=num_parallel_processes) as pool:
            all_fold_run_results = pool.map(_train_fold_worker, fold_worker_args)
        
        self.total_fold_results = {
            'train_loss': [res['train_losses'] for res in all_fold_run_results],
            'train_accuracy': [res['train_accuracies'] for res in all_fold_run_results],
            'train_f1': [res['train_f1s'] for res in all_fold_run_results],
            'val_loss': [res['val_losses'] for res in all_fold_run_results], # NN val losses
            'val_accuracy': [res['val_accuracies'] for res in all_fold_run_results], # NN val accuracies
            'val_f1': [res['val_f1s'] for res in all_fold_run_results], # NN val F1s
            'best_models_details': [res['best_model_details'] for res in all_fold_run_results]
        }

        # Filter out folds that might have failed (val_xgb_f1 = -1)
        valid_best_models = [m for m in self.total_fold_results['best_models_details'] if m.get('val_xgb_f1', -1) > -1]

        if not valid_best_models:
            print("Error: Training failed for all folds or no valid models were produced.")
            return self.total_fold_results # Or raise an error

        best_overall_model_details = max(valid_best_models, key=lambda x: x.get('val_xgb_f1'))
        
        # Find the original full result for the best model to get fold_num and device_used for printing
        best_model_full_res_idx = -1
        for idx, res_details in enumerate(self.total_fold_results['best_models_details']):
            if res_details is best_overall_model_details: # Check object identity
                best_model_full_res_idx = idx
                break
        
        best_model_fold_num_print = all_fold_run_results[best_model_full_res_idx]['fold_num'] + 1 if best_model_full_res_idx != -1 else "N/A"
        best_model_device_print = all_fold_run_results[best_model_full_res_idx]['device_used'] if best_model_full_res_idx != -1 else "N/A"


        if verbose:
            print(f"Best fold XGB F1: {best_overall_model_details.get('val_xgb_f1'):.4f} from fold {best_model_fold_num_print} on device {best_model_device_print}")

        # Save components of the best overall model
        cnn_save_path = self.model_save_path + "_Hybrid_CNN.pth"
        torch.save(best_overall_model_details['cnn_state'], cnn_save_path)
        if verbose: print(f"Best Hybrid CNN Model state saved to {cnn_save_path}")

        xgb_model_to_save = best_overall_model_details['classifier_xgb_state']
        xgb_save_path = self.model_save_path + "_Hybrid_XGB.json"
        xgb_model_to_save.save_model(xgb_save_path)
        if verbose: print(f"Best Hybrid XGBoost Model saved to {xgb_save_path}")

        ip_pipeline_to_save = best_overall_model_details['pipeline_state']
        if hasattr(ip_pipeline_to_save, 'steps'):
            ip_save_path = self.model_save_path + "_Hybrid_IP.joblib"
            joblib.dump(ip_pipeline_to_save, ip_save_path)
            if verbose: print(f"Best Hybrid Interaction Pipeline saved to {ip_save_path}")
        
        # Load the best model components into self for the main HybridModel object
        self.cnn_model.load_state_dict(best_overall_model_details['cnn_state'])
        self.cnn_model.to(self.device) # Move to the main model's designated device
        self.cnn_model.eval()
        self.classifier = best_overall_model_details['classifier_xgb_state']
        self.ip = best_overall_model_details['pipeline_state']
        
        # Populate self.fold_models and self.fold_pipelines for potential ensemble use
        self.fold_models = []
        self.fold_pipelines = []
        for fold_details_item in self.total_fold_results['best_models_details']: # Iterate over the list of dicts
            if fold_details_item and fold_details_item.get('val_xgb_f1', -1) > -1: # Check if fold was successful
                fold_cnn_instance = CNNCore(**self.cnn_params) 
                fold_cnn_instance.load_state_dict(fold_details_item['cnn_state'])
                fold_cnn_instance.eval()
                
                self.fold_models.append({
                    'cnn_model': fold_cnn_instance, 
                    'classifier': fold_details_item['classifier_xgb_state']
                })
                self.fold_pipelines.append(fold_details_item['pipeline_state'])
        if verbose: print(f"Best model components loaded into HybridModel instance. {len(self.fold_models)} fold models stored.")
        return self.total_fold_results

    def train_process_plot(self, save=True, val_loss_log=False, filter_coeff=None):
        if not self.total_fold_results:
            print("No training results to plot. Run train() first.")
            return

        # Ensure base_path for saving plots is defined
        base_path = os.path.join("src", "imgs", "training")
        if save and not os.path.exists(base_path): 
            os.makedirs(base_path)

        # Plotting for train/val loss (NN)
        plt.figure(figsize=(16,6))
        plt.subplot(1, 2, 1)
        for idx, loss_list in enumerate(self.total_fold_results['train_loss']):
            if loss_list: # Check if list is not empty
                sns.lineplot(x=range(len(loss_list)), y=loss_list, label=f'Fold{idx+1} train_loss', palette=PALETTE, alpha=ALPHA)
        plt.title('Train Loss (NN)')
        plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend(); plt.grid(True, linestyle='--', alpha=0.7)

        plt.subplot(1, 2, 2)
        for idx, loss_list in enumerate(self.total_fold_results['val_loss']): # NN val losses
            if not loss_list: continue # Skip empty list
            processed_loss_list = loss_list
            if filter_coeff:
                mean_loss = np.mean([l for l in loss_list if l is not None and not np.isinf(l) and not np.isnan(l)])
                std_loss = np.std([l for l in loss_list if l is not None and not np.isinf(l) and not np.isnan(l)])
                if not (np.isinf(mean_loss) or np.isnan(mean_loss) or np.isinf(std_loss) or np.isnan(std_loss)): # Ensure mean/std are valid
                    processed_loss_list = [l for l in loss_list if l is not None and l < mean_loss + filter_coeff * std_loss]
            
            # Filter out potential inf/nan before log if val_loss_log is True
            if val_loss_log:
                y_data_plot = [np.log(l) for l in processed_loss_list if l is not None and l > 0]
                x_data_plot = [i for i, l in enumerate(processed_loss_list) if l is not None and l > 0]
            else:
                y_data_plot = [l for l in processed_loss_list if l is not None]
                x_data_plot = [i for i, l in enumerate(processed_loss_list) if l is not None]

            if not x_data_plot: continue # Skip if no valid data to plot

            label_text = f'Fold{idx+1} val_log_loss' if val_loss_log else f'Fold{idx+1} val_loss'
            sns.lineplot(x=x_data_plot, y=y_data_plot, label=label_text, palette=PALETTE, alpha=ALPHA)
        plt.title('Validation Loss (NN)')
        plt.xlabel('Epochs'); plt.ylabel('Log Loss' if val_loss_log else 'Loss'); plt.legend(); plt.tight_layout(); plt.grid(True, linestyle='--', alpha=0.7)
        
        if save:
            plt.savefig(os.path.join(base_path, 'hybrid_train_val_loss_parallel.png'))
        plt.close() 
            
        # Plot for F1 scores (NN)
        plt.figure(figsize=(16, 6))
        plt.subplot(1, 2, 1)
        for idx, f1_list in enumerate(self.total_fold_results['train_f1']):
            if f1_list: sns.lineplot(x=range(len(f1_list)), y=f1_list, label=f'Fold{idx+1} train_f1 (NN)', palette=PALETTE, alpha=ALPHA)
        plt.title('Train F1 Score (NN)'); plt.xlabel('Epochs'); plt.ylabel('F1 Score'); plt.legend(); plt.grid(True, linestyle='--', alpha=0.7)

        plt.subplot(1, 2, 2)
        for idx, f1_list in enumerate(self.total_fold_results['val_f1']): # NN val F1s
            if f1_list: sns.lineplot(x=range(len(f1_list)), y=f1_list, label=f'Fold{idx+1} val_f1 (NN)', palette=PALETTE, alpha=ALPHA)
        plt.title('Validation F1 Score (NN)'); plt.xlabel('Epochs'); plt.ylabel('F1 Score'); plt.legend(); plt.tight_layout(); plt.grid(True, linestyle='--', alpha=0.7)

        if save:
            plt.savefig(os.path.join(base_path, 'hybrid_train_val_f1_parallel.png'))
        plt.close()
        
        # Plot for accuracies (NN)
        plt.figure(figsize=(16, 6))
        plt.subplot(1, 2, 1)
        for idx, acc_list in enumerate(self.total_fold_results['train_accuracy']):
            if acc_list: sns.lineplot(x=range(len(acc_list)), y=acc_list, label=f'Fold{idx+1} train_accuracy (NN)', palette=PALETTE, alpha=ALPHA)
        plt.title('Train Accuracy (NN)'); plt.xlabel('Epochs'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(True, linestyle='--', alpha=0.7)

        plt.subplot(1, 2, 2)
        for idx, acc_list in enumerate(self.total_fold_results['val_accuracy']): # NN val accuracies
            if acc_list: sns.lineplot(x=range(len(acc_list)), y=acc_list, label=f'Fold{idx+1} val_accuracy (NN)', palette=PALETTE, alpha=ALPHA)
        plt.title('Validation Accuracy (NN)'); plt.xlabel('Epochs'); plt.ylabel('Accuracy'); plt.legend(); plt.tight_layout(); plt.grid(True, linestyle='--', alpha=0.7)
        
        if save:
            plt.savefig(os.path.join(base_path, 'hybrid_train_val_accuracy_parallel.png'))
            print(f"Training plots saved to {base_path}")
        plt.close() # Close the last plot figure

    def _ensemble_predict(self, X: DataFrame, batch_size=16):
        if not self.fold_models:
            print("Warning: No fold models available for ensemble prediction. Attempting fallback to single model.")
            if self.cnn_model and self.classifier and self.ip:
                # This is a simplified fallback, ideally predict() handles this.
                # For _ensemble_predict, we strictly expect fold_models.
                # Re-evaluate if this fallback is needed here or should be solely in predict().
                # For now, let's assume _ensemble_predict requires fold_models.
                 raise RuntimeError("No fold models available for _ensemble_predict. This method is for ensemble mode only.")
            else:
                raise RuntimeError("No fold models and no single model loaded for fallback in _ensemble_predict.")

        all_fold_predictions_list = []
        # Determine a common device for prediction if possible, or handle device for each fold_cnn_model
        # For simplicity, using self.device (main model's device) for predictions.
        # Fold CNN models in self.fold_models might be on CPU if saved that way.
        
        start_time = time.time()

        for fold_idx, (fold_model_dict, fold_pipeline) in enumerate(zip(self.fold_models, self.fold_pipelines)):
            # Ensure the fold's CNN model is on the correct device for prediction
            current_cnn_model = fold_model_dict['cnn_model'].to(self.device) 
            current_cnn_model.eval()
            current_xgb_classifier = fold_model_dict['classifier'] # XGBoost model, device handled by XGBoost itself
            
            fold_embeddings_list = []
            num_samples = len(X)
            for i in range(0, num_samples, batch_size):
                batch_end = min(i + batch_size, num_samples)
                # Assuming X is a DataFrame
                batch_data_values = X.iloc[i:batch_end].values 
                X_tensor = torch.tensor(batch_data_values, dtype=torch.float32).unsqueeze(1).to(self.device)
                with torch.no_grad():
                    batch_embeddings = current_cnn_model(X_tensor).cpu().numpy() # Embeddings to CPU for hstack
                    fold_embeddings_list.append(batch_embeddings)
            
            if not fold_embeddings_list: continue # Should not happen if X is not empty

            fold_embeddings = np.vstack(fold_embeddings_list)
            fold_interaction = fold_pipeline.transform(X) # Use this fold's pipeline
            fold_interaction_feats = fold_interaction.values
            fold_combined = np.hstack((fold_embeddings, fold_interaction_feats))
            
            fold_predictions = current_xgb_classifier.predict(fold_combined)
            fold_predictions = np.where(fold_predictions > 0.5, 1, 0) if fold_predictions.dtype == float else fold_predictions
            all_fold_predictions_list.append(fold_predictions.astype(int))
        
        end_time = time.time()
        time_per_prediction = (end_time - start_time) / len(X) if len(X) > 0 and all_fold_predictions_list else 0
        
        if not all_fold_predictions_list: # If all folds failed or X was empty
            # Return an empty array of appropriate shape or handle error
            return np.array([]).reshape(0, len(X) if X is not None and not X.empty else 0), 0

        return np.array(all_fold_predictions_list), time_per_prediction

    def predict(self, X: DataFrame, batch_size=16):
        start_time_total = time.time()
        
        if not self.fold_models and not (self.cnn_model and self.classifier and self.ip):
             raise RuntimeError("Model not trained or loaded. Call train() or load_weights() first.")

        if not self.fold_models: 
            if self.cnn_model is None or self.classifier is None or self.ip is None:
                 raise RuntimeError("Single model components not fully loaded/available for prediction.")
            print("Predicting with single loaded model (not ensemble).")
            self.cnn_model.eval()
            self.cnn_model.to(self.device) 
            
            all_embeddings = []
            num_samples = len(X)
            for i in range(0, num_samples, batch_size):
                batch_end = min(i + batch_size, num_samples)
                batch_data_values = X.iloc[i:batch_end].values
                X_tensor = torch.tensor(batch_data_values, dtype=torch.float32).unsqueeze(1).to(self.device)
                with torch.no_grad():
                    batch_embeddings = self.cnn_model(X_tensor).cpu().numpy()
                    all_embeddings.append(batch_embeddings)
            
            if not all_embeddings: # Handle empty X
                return pd.DataFrame(), np.array([]), 0

            embeddings = np.vstack(all_embeddings)
            interaction = self.ip.transform(X)
            interaction_feats = interaction.values
            combined_features = np.hstack((embeddings, interaction_feats))
            
            predictions = self.classifier.predict(combined_features)
            predictions = np.where(predictions > 0.5, 1, 0) if predictions.dtype == float else predictions
            final_predictions = predictions.astype(int)
            
            time_per_pred = (time.time() - start_time_total) / len(X) if len(X) > 0 else 0
            return combined_features, final_predictions, time_per_pred

        # Proceed with ensemble prediction
        all_fold_predictions_array, _ = self._ensemble_predict(X, batch_size) # Time already calculated per step if needed
        
        if all_fold_predictions_array.shape[0] == 0:
            print("Warning: _ensemble_predict returned no predictions. Check fold model training.")
            # Fallback or error, for now, let's assume an issue and return empty.
            return pd.DataFrame(), np.array([]), 0


        # Majority voting: input (num_folds, num_samples), output (num_samples,)
        if all_fold_predictions_array.ndim == 2 and all_fold_predictions_array.shape[1] > 0:
             # Ensure all elements are integers for bincount
            ensemble_predictions = np.apply_along_axis(lambda x: np.bincount(x.astype(int)).argmax(), axis=0, arr=all_fold_predictions_array)
        elif all_fold_predictions_array.shape[1] == 0 and all_fold_predictions_array.shape[0] > 0 : # No samples but folds exist
            ensemble_predictions = np.array([])
        else: # No folds or unexpected shape
             print(f"Warning: Unexpected shape from _ensemble_predict: {all_fold_predictions_array.shape}. Cannot perform majority vote.")
             return pd.DataFrame(), np.array([]), 0


        # For returning 'combined' features, use the primary loaded model (best one from train or load_weights)
        self.cnn_model.eval() 
        self.cnn_model.to(self.device)
        primary_embeddings_list = []
        num_samples = len(X)
        if num_samples > 0:
            for i in range(0, num_samples, batch_size):
                batch_end = min(i + batch_size, num_samples)
                batch_data_values = X.iloc[i:batch_end].values
                X_tensor = torch.tensor(batch_data_values, dtype=torch.float32).unsqueeze(1).to(self.device)
                with torch.no_grad():
                    batch_embeddings = self.cnn_model(X_tensor).cpu().numpy()
                    primary_embeddings_list.append(batch_embeddings)
            primary_embeddings = np.vstack(primary_embeddings_list)
            
            if self.ip is None: raise ValueError("Primary InteractionPipeline (self.ip) not available.")
            primary_interaction = self.ip.transform(X)
            primary_interaction_feats = primary_interaction.values
            combined_for_return = np.hstack((primary_embeddings, primary_interaction_feats))
        else: # Handle empty X DataFrame
            combined_for_return = pd.DataFrame() # Or np.array([]).reshape(0, feature_dim) if dim is known

        total_time_per_prediction = (time.time() - start_time_total) / len(X) if len(X) > 0 else 0
        
        return combined_for_return, ensemble_predictions.astype(int), total_time_per_prediction

    def evaluate(self, X_test: DataFrame, y_test: Series, batch_size=32, plot=True):
        # Ensure y_test is numpy array for metric functions
        y_test_numpy = y_test.to_numpy() if isinstance(y_test, pd.Series) else np.asarray(y_test)

        if X_test.empty:
            print("Warning: X_test is empty for evaluation.")
            return {'precision': 0, 'recall': 0, 'f1': 0, 'time_per_prediction': 0, 
                    'predictions': np.array([]), 'true_labels': y_test_numpy, 'confusion_matrix': np.zeros((2,2))}


        _, predictions, time_per_pred = self.predict(X_test, batch_size=batch_size)

        if len(predictions) != len(y_test_numpy):
            print(f"Warning: Length mismatch between predictions ({len(predictions)}) and true labels ({len(y_test_numpy)}). Evaluation might be incorrect.")
            # Pad or truncate if necessary, or error out. For now, proceed with caution.
            # This often indicates an issue in predict or data handling for empty X.
            # If predictions is empty due to empty X, metrics will be zero.
            if len(predictions) == 0 and len(y_test_numpy) > 0: # No predictions for non-empty y_test
                 predictions = np.zeros_like(y_test_numpy) # Default to all zeros for calculation
            # Or, if y_test_numpy is empty and predictions are not (should not happen with empty X)
            # elif len(y_test_numpy) == 0 and len(predictions) > 0:
            #      y_test_numpy = np.zeros_like(predictions)


        precision = precision_score(y_test_numpy, predictions, average='binary', zero_division=0)
        recall = recall_score(y_test_numpy, predictions, average='binary', zero_division=0)
        f1 = f1_score(y_test_numpy, predictions, average='binary', zero_division=0)
        
        # Ensure confusion matrix can be computed (e.g. if predictions or y_test_numpy is empty)
        try:
            cm = confusion_matrix(y_test_numpy, predictions, labels=np.unique(np.concatenate((y_test_numpy, predictions))) if len(np.unique(np.concatenate((y_test_numpy, predictions)))) > 0 else [0,1] )
            if cm.shape != (2,2) and len(np.unique(y_test_numpy)) <= 2 : # Handle cases where only one class is predicted or present
                # Rebuild a 2x2 CM if possible, assuming binary case [0,1]
                unique_labels = np.unique(np.concatenate((y_test_numpy, predictions)))
                if len(unique_labels) == 1: # Only one class present/predicted
                    temp_cm = np.zeros((2,2), dtype=int)
                    cls = unique_labels[0]
                    if cls in [0,1]: # Ensure it's one of the expected binary labels
                        # Count how many times this single class was predicted correctly/incorrectly
                        # This is tricky without knowing the "other" class.
                        # For simplicity, if only class 0 is present and predicted, it's TN. If only 1, TP.
                        # This part of CM reconstruction can be complex.
                        # A simpler approach for a fixed binary [0,1] problem:
                        tn, fp, fn, tp = 0,0,0,0
                        for i in range(len(predictions)):
                            if y_test_numpy[i] == 0 and predictions[i] == 0: tn +=1
                            elif y_test_numpy[i] == 0 and predictions[i] == 1: fp +=1
                            elif y_test_numpy[i] == 1 and predictions[i] == 0: fn +=1
                            elif y_test_numpy[i] == 1 and predictions[i] == 1: tp +=1
                        cm = np.array([[tn, fp], [fn, tp]])
                elif cm.size == 0 : # If unique labels resulted in empty cm
                     cm = np.zeros((2,2), dtype=int)


        except ValueError: # If labels arg fails due to no common labels or empty inputs
            cm = np.zeros((2,2), dtype=int) # Default CM

        print(f"Evaluation Results (using {'Ensemble of '+str(len(self.fold_models))+' folds' if self.fold_models else 'Single Model'}):")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  Time per prediction: {time_per_pred:.6f} seconds")
        print(f"  Confusion Matrix:\n{cm}")

        if plot:
            # Plotting logic (e.g., confusion matrix)
            # import scikitplot as skplt # Ensure imported
            # skplt.metrics.plot_confusion_matrix(y_test_numpy, predictions, normalize=True)
            # plt.show()
            pass # Placeholder

        return {
            'precision': precision, 'recall': recall, 'f1': f1,
            'time_per_prediction': time_per_pred,
            'predictions': predictions,
            'true_labels': y_test_numpy,
            'confusion_matrix': cm
        }

    def load_weights(self, model_path_prefix):
        cnn_path = model_path_prefix + "_Hybrid_CNN.pth"
        xgb_path = model_path_prefix + "_Hybrid_XGB.json"
        ip_path = model_path_prefix + "_Hybrid_IP.joblib"

        if os.path.exists(cnn_path):
            self.cnn_model.load_state_dict(torch.load(cnn_path, map_location=self.device))
            self.cnn_model.to(self.device) 
            self.cnn_model.eval() 
            print(f"CNN model loaded from {cnn_path} to device {self.device}")
        else:
            raise FileNotFoundError(f"CNN model file not found: {cnn_path}")

        if os.path.exists(xgb_path):
            self.classifier = XGBClassifier() 
            self.classifier.load_model(xgb_path)
            print(f"XGBoost model loaded from {xgb_path}")
        else:
            raise FileNotFoundError(f"XGBoost model file not found: {xgb_path}")

        if os.path.exists(ip_path):
            self.ip = joblib.load(ip_path)
            print(f"InteractionPipeline loaded from {ip_path}")
        else:
            print(f"Warning: InteractionPipeline file not found: {ip_path}. self.ip remains None.")
            self.ip = None 
        
        self.fold_models = []
        self.fold_pipelines = []
        print("Model weights and pipeline loaded (single best model configuration). Ensemble folds cleared.")