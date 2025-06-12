from model import HybridModel
from utils.pipeline import LabelPipeline
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os

if __name__ == '__main__':
    SEED = 42
    np.random.seed(SEED)

    cnn_params = {
        'input_length': 4000, 'embedding_dim': 256,
        'kernel_sizes': [3, 3, 5, 5], 'num_filters': 64, 'drop_out': 0.2,            
    }
    classifier_params = {
        'n_estimators':500, 'max_depth': 7, 'learning_rate': 0.05,      
        'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 0.9,    
        'gamma': 0, 'reg_alpha': 0.1, 'reg_lambda': 1.0,          
        'objective': 'binary:logistic', 'eval_metric': 'logloss'
    }
        
    train_df_path = "data/traindata.csv"
    test_df_path = "data/testdata.csv"
    unlabeled_preds_path = "data/unlabeled_predictions.csv"

    if not os.path.exists(train_df_path): raise FileNotFoundError(f"traindata.csv not found at {train_df_path}")
    train_df = pd.read_csv(train_df_path)
    
    if os.path.exists(test_df_path):
        test_df = pd.read_csv(test_df_path) 
    else:
        print(f"Warning: testdata.csv not found at {test_df_path}. Test prediction will not be possible.")
        test_df = None
    
    lp = LabelPipeline(train_df) 
    
    labels_for_pseudo = None
    if os.path.exists(unlabeled_preds_path):
        try:
            temp_df = pd.read_csv(unlabeled_preds_path, header=None, delimiter=',')
            if not temp_df.empty:
                # Case 1: CSV has 19000 columns (wide format).
                # We take the first row of predictions.
                if temp_df.shape[1] == 19000:
                    labels_for_pseudo = temp_df.iloc[0]  # Results in a Pandas Series of length 19000
                    print(f"Info: Loaded {len(labels_for_pseudo)} pseudo-labels from the first row of {unlabeled_preds_path} (detected {temp_df.shape[1]} columns).")
                # Case 2: CSV has 19000 rows and one or more columns (tall format, e.g., (19000,1) or (19000,2)).
                # We take all rows from the first column of predictions.
                elif temp_df.shape[0] == 19000 and temp_df.shape[1] >= 1:
                    labels_for_pseudo = temp_df.iloc[:, 0] # Results in a Pandas Series of length 19000
                    print(f"Info: Loaded {len(labels_for_pseudo)} pseudo-labels from the first column of {unlabeled_preds_path} (detected {temp_df.shape[0]} rows, {temp_df.shape[1]} columns).")
                # Case 3: Unexpected shape.
                else:
                    print(f"Warning: {unlabeled_preds_path} has an unexpected shape {temp_df.shape}. Expected 19000 columns (wide) OR 19000 rows (tall, taking first column). Proceeding without pseudo-labels.")
                    labels_for_pseudo = None
            else:
                print(f"Warning: {unlabeled_preds_path} is empty. Proceeding without pseudo-labels.")
                labels_for_pseudo = None
        except Exception as e:
            print(f"Warning: Could not read or process {unlabeled_preds_path}: {e}. Proceeding without pseudo-labels.")
            labels_for_pseudo = None
    else:
        print(f"Info: {unlabeled_preds_path} not found. Proceeding without pseudo-labeling if applicable by LabelPipeline.")

    labeled_signals, targets = lp.add_labels(labels_for_pseudo, cutedge=(500, 1000)) 
    
    if not isinstance(labeled_signals, pd.DataFrame):
        print("Warning: labeled_signals is not a DataFrame. Attempting conversion.")
        labeled_signals = pd.DataFrame(labeled_signals)
    if not isinstance(targets, pd.Series):
        print("Warning: targets is not a Series. Attempting conversion.")
        targets = pd.Series(targets)
        
    print(f"Total labeled signals: {labeled_signals.shape[0]}, Total targets: {targets.shape[0]}")
    if labeled_signals.empty or targets.empty:
        raise ValueError("Data loading or processing resulted in empty signals or targets. Cannot proceed.")
    
    X_train_cv, X_final_val, y_train_cv, y_final_val = train_test_split(
        labeled_signals, targets, test_size=0.2, random_state=SEED, stratify=targets, shuffle=True
    )
    print(f"Data for K-fold CV: {X_train_cv.shape[0]} samples. Final validation set: {X_final_val.shape[0]} samples.")
    
    model = HybridModel(cnn_params=cnn_params, classifier_params=classifier_params) 
    
    model_weights_dir = "models"
    if not os.path.exists(model_weights_dir):
        os.makedirs(model_weights_dir)
    model_save_base_path = os.path.join(model_weights_dir, "hybrid_model_parallel_final")

    gpu_ids_for_training = [0, 1] 

    print("\nStarting model training with K-fold cross-validation...")
    model.train(X_train_cv, y_train_cv, 
                learning_rate=0.00015, 
                num_epochs=200, batch_size=32, splits=5, 
                early_stopping=50, 
                model_save_path=model_save_base_path,
                gpu_ids=gpu_ids_for_training,
                verbose=True, verbose_epoch=False) 
    
    print("\nModel training finished.")
    model.train_process_plot(save=True, val_loss_log=False)

    if not X_final_val.empty:
        print("\nEvaluating on the final held-out validation set:")
        eval_results_final_val = model.evaluate(X_final_val, y_final_val, plot=True)
        print(f"Final Validation F1 Score: {eval_results_final_val['f1']:.4f}")
    else:
        print("Skipping evaluation on final held-out set as it's empty.")

    if test_df is not None and not test_df.empty:
        print("\nPredicting on test set...")
        _, test_pred, test_time_per_pred = model.predict(test_df)
        print(f"Test prediction time per sample: {test_time_per_pred:.6f} seconds")
        
        predictions_save_path = 'data/predictions_parallel.csv'
        pd.DataFrame(test_pred).to_csv(predictions_save_path, index=False, header=False)
        print(f"Test predictions saved to {predictions_save_path}")
    else:
        print("Skipping prediction on test set as test_df is not available or empty.")

    print("\nMain script finished. For detailed evaluation of the saved model using specific data, you can adapt and run eval.py.")



