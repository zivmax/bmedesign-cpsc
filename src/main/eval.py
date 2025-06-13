import pandas as pd
import os
import matplotlib.pyplot as plt # Add this import
from ..utils.pipeline import LabelPipeline # SignalAugmentationPipeline no longer directly used here
from .model import HybridModel

# Parameters from main.py
# CNN parameters from main.py
CNN_PARAMS = {
    "input_length": 4000,
    "embedding_dim": 256,
    "kernel_sizes": [3, 3, 5, 5],
    "num_filters": 64,
    "drop_out": 0.2,
}

# Classifier parameters (XGBoost) from main.py - not strictly needed for loading, but good for consistency
CLASSIFIER_PARAMS = {
    "n_estimators": 500,
    "max_depth": 7,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 0.9,
    "gamma": 0,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
}

MODEL_PATH_PREFIX = "models/hybrid_model_parallel_final"
TEST_DATA_PATH = "data/testdata.csv"
# OUTPUT_CSV_PATH is defined below, before it's used.
BATCH_SIZE = 32 # From main.py's model.train call, or choose a sensible default for eval
DEVICE = None # Autodetect, similar to main.py's HybridModel instantiation

def load_data(file_path: str) -> tuple[pd.DataFrame, pd.Series]:
    """Loads data from a CSV file.
    Assumes the CSV has a 'target' column and the rest are features.
    """
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    if 'target' not in df.columns:
        raise ValueError(f"Test data CSV file '{file_path}' must contain a 'target' column.")
    
    X = df.drop(columns=['target'])
    y = df['target']
    print(f"Data loaded: {len(X)} samples.")
    return X, y

def main():
    # Hardcoded output path, as args are removed.
    current_output_csv_path = "data/evaluation_results.csv"

    # Use hardcoded CNN_PARAMS
    cnn_params = CNN_PARAMS
    print(f"Using CNN parameters from main.py: {cnn_params}")

    train_df_path = "data/traindata.csv"
    unlabeled_preds_path = "data/unlabeled_predictions.csv"

    train_df = pd.read_csv(train_df_path)

    lp = LabelPipeline(train_df)
    temp_df = pd.read_csv(unlabeled_preds_path, header=None)
    labels_for_pseudo = None

    # Check if the DataFrame is not empty and has at least one column
    if temp_df.shape[0] == 19000:
        labels_for_pseudo = temp_df.iloc[
            :, 0
        ]  # Results in a Pandas Series from the first column
        print(
            f"Info: Loaded {len(labels_for_pseudo)} pseudo-labels from the first column of {unlabeled_preds_path} (detected {temp_df.shape[1]} columns)."
        )
    else:
        print(
            f"Warning: {unlabeled_preds_path} is empty or has no columns. No pseudo-labels loaded."
        )

    labeled_signals, targets = lp.add_labels(labels_for_pseudo, cutedge=(500, 1000))

    # Use hardcoded CLASSIFIER_PARAMS (though not strictly needed for loading)
    classifier_params = CLASSIFIER_PARAMS
    
    print(f"Initializing HybridModel...")
    try:
        model = HybridModel(cnn_params=cnn_params, 
                            classifier_params=classifier_params, 
                            device=DEVICE)
    except Exception as e:
        print(f"Error initializing HybridModel: {e}")
        print("This could be due to incorrect cnn_params or issues with the HybridModel class itself.")
        return

    print(f"Loading model weights from prefix: {MODEL_PATH_PREFIX}")
    try:
        model.load_weights(MODEL_PATH_PREFIX) # This method in model.py should print success/failure
    except Exception as e:
        print(f"Error loading model weights: {e}")
        print("Ensure the model_path_prefix is correct and all model files (.pth, .json, .joblib) exist.")
        return

    print(f"Loading test data from: {TEST_DATA_PATH}")
    try:
        X_test, y_test = lp.add_labels(labels_for_pseudo, cutedge=(500, 1000))

    except Exception as e:
        print(f"Error loading test data: {e}")
        return
    
    if X_test.empty:
        print("Test data is empty. Cannot perform evaluation.")
        return

    print("Starting evaluation...")
    try:
        eval_results = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE, plot=False)
    except Exception as e:
        print(f"Error during model evaluation: {e}")
        print("This might be due to issues in HybridModel's predict or evaluate methods, or data mismatches.")
        return

    if not eval_results:
        print("Evaluation did not return any results. Cannot proceed.")
        return

    f1 = eval_results.get("f1")
    precision = eval_results.get("precision")
    recall = eval_results.get("recall")
    time_per_pred_seconds = eval_results.get("time_per_prediction")
    cm = eval_results.get("confusion_matrix")

    if any(m is None for m in [f1, precision, recall, time_per_pred_seconds, cm]):
        print("Evaluation results dictionary is missing one or more required keys (f1, precision, recall, time_per_prediction, confusion_matrix).")
        print(f"Received keys: {list(eval_results.keys())}")
        return
        
    time_per_pred_ms = time_per_pred_seconds * 1000

    print("\n--- Evaluation Summary ---")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Time per inference (avg over test set): {time_per_pred_ms:.4f} ms")
    print(f"Confusion Matrix:\n{cm}")

    # Save results to CSV
    results_df = pd.DataFrame({
        "model_path_prefix": [MODEL_PATH_PREFIX],
        "test_data_path": [TEST_DATA_PATH],
        "f1_score": [f1],
        "precision": [precision],
        "recall": [recall],
        "time_per_inference_ms": [time_per_pred_ms],
        "batch_size": [BATCH_SIZE]
    })

    output_dir = os.path.dirname(current_output_csv_path)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        except OSError as e:
            print(f"Error creating output directory {output_dir}: {e}")
            print(f"Attempting to save to current directory: {os.path.basename(current_output_csv_path)}")
            current_output_csv_path = os.path.basename(current_output_csv_path)


    try:
        results_df.to_csv(current_output_csv_path, index=False)
        print(f"Evaluation results saved to {current_output_csv_path}")
    except Exception as e:
        print(f"Error saving evaluation results to CSV: {e}")

    # Robustness evaluation is now handled by run_robustness.py
    print("\nRobustness evaluation can be run separately using: python -m src.main.run_robustness")

if __name__ == "__main__":
    # This structure assumes that when you run `python src/main/eval.py` from the root
    # of your project (/home/zivmax/bmedesign-cpsc), Python's import system
    # correctly handles `from .model import HybridModel`.
    # If you encounter import errors, you might need to adjust Python's path
    # or how you run the script (e.g., `python -m src.main.eval`).
    main()
