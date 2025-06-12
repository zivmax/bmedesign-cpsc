\
import csv
import os
import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from model import HybridModel
from utils.pipeline import LabelPipeline

def evaluate_model(model_save_load_path, cnn_params, classifier_params, data_path="data/", results_csv_path="evaluation_results.csv"):
    """
    Loads a trained model, evaluates it on validation data, and saves the results to a CSV file.

    Args:
        model_save_load_path (str): Path prefix for loading the model weights.
        cnn_params (dict): Parameters for the CNNCore model.
        classifier_params (dict): Parameters for the XGBoost classifier.
        data_path (str): Path to the directory containing traindata.csv and unlabeled_predictions.csv.
        results_csv_path (str): Path to save the evaluation results CSV.
    """
    SEED = 42

    # Load data
    train_df = pd.read_csv(os.path.join(data_path, 'traindata.csv'))
    # Assuming unlabeled_predictions.csv is needed by LabelPipeline
    labels_df = pd.read_csv(os.path.join(data_path, 'unlabeled_predictions.csv'), delimiter=',')
    
    lp = LabelPipeline(train_df)
    # Corrected call to add_labels, assuming it expects a DataFrame for labels
    labeled_signals, targets = lp.add_labels(labels_df, cutedge=(500, 1000))
    
    print(f"Total labeled signals: {labeled_signals.shape[0]}, Total targets: {targets.shape[0]}")
    
    _, X_val, _, y_val = train_test_split(
        labeled_signals, targets, test_size=0.2, random_state=SEED
    )

    # Initialize model
    model = HybridModel(cnn_params=cnn_params, classifier_params=classifier_params)

    # Load trained model weights
    print(f"Loading model from: {model_save_load_path}")
    model.load_weights(model_save_load_path)
    print("Model loaded successfully.")

    # Evaluate the model on the validation set
    print("Evaluating model on validation set...")
    eval_results = model.evaluate(X_val, y_val, plot=False)

    # Prepare results for CSV output
    results_to_save = {
        'Timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'ModelPath': model_save_load_path,
        'Precision': eval_results.get('precision'),
        'Recall': eval_results.get('recall'),
        'F1_Score': eval_results.get('f1'),
        'Time_Per_Prediction_Seconds': eval_results.get('time_per_prediction')
    }

    # Write results to CSV
    file_exists = os.path.isfile(results_csv_path)
    with open(results_csv_path, 'a', newline='') as csvfile:
        fieldnames = ['Timestamp', 'ModelPath', 'Precision', 'Recall', 'F1_Score', 'Time_Per_Prediction_Seconds']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()  # Write header only if file is new
        writer.writerow(results_to_save)
    
    print(f"Evaluation results saved to {results_csv_path}")
    print(f"Validation F1 Score: {eval_results['f1']}")

if __name__ == '__main__':
    # These parameters should match those used during training
    cnn_params_eval = {
        'input_length': 4000,
        'embedding_dim': 256,
        'kernel_sizes': [3, 3, 5, 5],  
        'num_filters': 64,         
        'drop_out': 0.2,            
    }
    classifier_params_eval = {
        'n_estimators':500,         
        'max_depth': 7,             
        'learning_rate': 0.05,      
        'subsample': 0.8,           
        'colsample_bytree': 0.8,    
        'min_child_weight': 0.9,    
        'gamma': 0,                 
        'reg_alpha': 0.1,           
        'reg_lambda': 1.0,          
        'objective': 'binary:logistic',  
        'eval_metric': 'logloss'
    }
    
    model_path = "models/hybrid_model_final"
    
    evaluate_model(model_path, cnn_params_eval, classifier_params_eval)
