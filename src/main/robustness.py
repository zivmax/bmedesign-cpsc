import pandas as pd
import os
from ..utils.pipeline import LabelPipeline
from .model import HybridModel
import pandas as pd
import os
import matplotlib.pyplot as plt
from ..utils.pipeline import SignalAugmentationPipeline

def evaluate_robustness(model, X_test: pd.DataFrame, y_test: pd.Series, batch_size: int, output_directory: str, target_snr_db: int = 0, ax=None, original_signal_sample_for_plot: pd.Series | None = None): # Added original_signal_sample_for_plot
    """
    Performs robustness evaluation on the model using noisy data.

    Args:
        model: The trained model object with an 'evaluate' method.
        X_test: DataFrame of test features.
        y_test: Series of test labels.
        batch_size: Batch size for model evaluation.
        output_directory: Directory to save plots and noisy results CSV.
        target_snr_db: Target Signal-to-Noise Ratio in dB for generating noisy data.
        ax: Matplotlib axis object for plotting.
        original_signal_sample_for_plot: The specific original signal sample to plot.
    """
    print(f"\n--- Evaluating for SNR: {target_snr_db}dB ---")

    try:
        print("Generating noisy test data for robustness check...")
        X_test_noisy = pd.DataFrame() # Initialize as empty

        if isinstance(X_test, pd.DataFrame) and not X_test.empty:
            X_test_T = X_test.T 
            y_test_df_for_pipeline = y_test.to_frame()

            sap = SignalAugmentationPipeline(
                labeled_df=X_test_T,
                labeled_target_df=y_test_df_for_pipeline,
                noise_level=target_snr_db,
                window_length=1, 
                lag=0,           
                diff=False       
            )
            
            X_test_noisy_T = sap.noise_df
            
            if X_test_noisy_T is not None and not X_test_noisy_T.empty:
                X_test_noisy = X_test_noisy_T.T 
                X_test_noisy.columns = X_test.columns 
                X_test_noisy.index = X_test.index     
                print(f"Applied noise to achieve target SNR of {target_snr_db}dB to {len(X_test_noisy)} test samples.")

                # Plotting original vs noisy for the selected sample
                # original_signal_sample_for_plot is X_test.iloc[0] (or another chosen sample) passed from main
                # X_test_noisy.iloc[0] is its corresponding noisy version
                if original_signal_sample_for_plot is not None and \
                   not X_test_noisy.empty and \
                   ax is not None:
                    ax.plot(original_signal_sample_for_plot.values, label="Original Signal (Sample 0)")
                    ax.plot(X_test_noisy.iloc[0].values, label=f"Noisy Signal (SNR={target_snr_db}dB)", alpha=0.7)
                    ax.set_title(f"Original vs. Noisy Signal (SNR: {target_snr_db}dB)")
                    ax.set_xlabel("Time Step")
                    ax.set_ylabel("Amplitude")
                    ax.legend()
            else:
                print("Warning: SignalAugmentationPipeline returned empty or None noise_df.")
        else:
            print("Warning: X_test is not a suitable DataFrame (empty or not DataFrame type) for noise addition.")

        if X_test_noisy.empty:
            print("Noisy test data is empty or could not be generated. Skipping robustness evaluation.")
            return None # Return None if no results

        print("Starting robustness evaluation (with noisy data)...")
        eval_results_noisy = model.evaluate(X_test_noisy, y_test, batch_size=batch_size, plot=False)

        if not eval_results_noisy:
            print("Robustness evaluation (noisy data) did not return any results. Cannot proceed.")
            return None # Return None if no results

        f1_noisy = eval_results_noisy.get("f1")
        precision_noisy = eval_results_noisy.get("precision")
        recall_noisy = eval_results_noisy.get("recall")
        cm_noisy = eval_results_noisy.get("confusion_matrix")

        if any(m is None for m in [f1_noisy, precision_noisy, recall_noisy, cm_noisy]):
            print("Robustness evaluation results dictionary is missing one or more required keys (f1, precision, recall, confusion_matrix).")
            print(f"Received keys: {list(eval_results_noisy.keys())}")
            return None # Return None if results are incomplete

        print(f"F1 Score (Noisy, {target_snr_db}dB): {f1_noisy:.4f}")
        print(f"Precision (Noisy, {target_snr_db}dB): {precision_noisy:.4f}")
        print(f"Recall (Noisy, {target_snr_db}dB): {recall_noisy:.4f}")
        print(f"Confusion Matrix (Noisy, {target_snr_db}dB):\n{cm_noisy}")

        # Results are now returned to be aggregated by the main function
        results_dict = {
            "eval_type": "robustness_snr",
            "target_snr_db": target_snr_db,
            "f1_score": f1_noisy,
            "precision": precision_noisy,
            "recall": recall_noisy,
        }
        
        return results_dict

    except ImportError as e: 
        print(f"ImportError during robustness evaluation: {e}. Check pipeline dependencies.")
    except AttributeError as e:
        print(f"AttributeError during robustness evaluation: {e}. This might be due to an issue with an object's method or attribute.")
    except Exception as e:
        print(f"An unexpected error occurred during robustness evaluation with noisy data (SNR: {target_snr_db}dB): {e}")
        import traceback
        traceback.print_exc()
        return None # Return None on error


# Default parameters (can be overridden by command-line arguments)
# Copied from eval.py for consistency in model loading
CNN_PARAMS = {
    "input_length": 4000,
    "embedding_dim": 256,
    "kernel_sizes": [3, 3, 5, 5],
    "num_filters": 64,
    "drop_out": 0.2,
}

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

DEFAULT_MODEL_PATH_PREFIX = "models/hybrid_model_parallel_final"
DEFAULT_TRAIN_DATA_PATH = "data/traindata.csv" # For LabelPipeline initialization
DEFAULT_UNLABELED_PREDS_PATH = "data/unlabeled_predictions.csv" # For pseudo-labeling in test set generation
DEFAULT_OUTPUT_DIR = "data" 
DEFAULT_BATCH_SIZE = 32
TARGET_SNR_DB_LIST = [20, 0, -20] # Define the list of SNR values to test
DEFAULT_CUTEDGE_START = 500
DEFAULT_CUTEDGE_END = 1000
DEVICE = None # Autodetect

def main():

    print(f"Using CNN parameters: {CNN_PARAMS}")
    print(f"Using Classifier parameters: {CLASSIFIER_PARAMS}") # Though not strictly needed for loading

    # Initialize LabelPipeline (needed for generating X_test, y_test consistently with eval.py)
    print(f"Loading training data for LabelPipeline from: {DEFAULT_TRAIN_DATA_PATH}")
    try:
        train_df = pd.read_csv(DEFAULT_TRAIN_DATA_PATH)
        lp = LabelPipeline(train_df) # Initialize with cut_edge defaults from LabelPipeline itself if not specified
    except Exception as e:
        print(f"Error loading train_df or initializing LabelPipeline: {e}")
        return

    # Load pseudo-labels for test set generation
    labels_for_pseudo = None
    if os.path.exists(DEFAULT_UNLABELED_PREDS_PATH):
        try:
            temp_df = pd.read_csv(DEFAULT_UNLABELED_PREDS_PATH, header=None)
            if not temp_df.empty and temp_df.shape[0] > 0: # Basic check
                 # Assuming pseudo labels are in the first column if file is not empty
                labels_for_pseudo = temp_df.iloc[:, 0]
                print(f"Info: Loaded {len(labels_for_pseudo)} pseudo-labels from {DEFAULT_UNLABELED_PREDS_PATH}.")
            else:
                print(f"Warning: {DEFAULT_UNLABELED_PREDS_PATH} is empty or has no columns. No pseudo-labels loaded for test data generation.")
        except Exception as e:
            print(f"Error loading pseudo-labels from {DEFAULT_UNLABELED_PREDS_PATH}: {e}")
    else:
        print(f"Warning: Unlabeled predictions file not found at {DEFAULT_UNLABELED_PREDS_PATH}. No pseudo-labels loaded.")

    # Generate X_test, y_test using LabelPipeline
    # This mimics how X_test, y_test are generated in eval.py
    print("Generating test data (X_test, y_test) using LabelPipeline...")
    try:
        # The lp.add_labels method uses the dataframe it was initialized with (train_df)
        # to create the structure, then applies labels.
        # This implies X_test here is derived from the structure of train_df.
        X_test, y_test = lp.add_labels(labels_for_pseudo, cutedge=(DEFAULT_CUTEDGE_START, DEFAULT_CUTEDGE_END))
        if X_test.empty:
            print("Generated X_test is empty. Cannot perform robustness evaluation.")
            return
        print(f"Generated X_test with {len(X_test)} samples.")
    except Exception as e:
        print(f"Error generating X_test, y_test using LabelPipeline: {e}")
        return

    # Select the specific original signal sample for plotting across all SNRs
    # Ensure X_test is not empty before trying to access iloc[0] - already checked above
    original_signal_to_plot = X_test.iloc[0].copy()
    print(f"Selected X_test.iloc[0] as the consistent original signal for plotting.")

    # Initialize and load the model
    print(f"Initializing HybridModel...")
    try:
        model = HybridModel(cnn_params=CNN_PARAMS, 
                            classifier_params=CLASSIFIER_PARAMS, 
                            device=DEVICE)
    except Exception as e:
        print(f"Error initializing HybridModel: {e}")
        return

    print(f"Loading model weights from prefix: {DEFAULT_MODEL_PATH_PREFIX}")
    try:
        model.load_weights(DEFAULT_MODEL_PATH_PREFIX)
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return

    # Ensure output directory exists
    current_output_dir = DEFAULT_OUTPUT_DIR # Use the hardcoded default
    if current_output_dir and not os.path.exists(current_output_dir):
        try:
            os.makedirs(current_output_dir)
            print(f"Created output directory: {current_output_dir}")
        except OSError as e:
            print(f"Error creating output directory {current_output_dir}: {e}. Using current directory if possible.")
            current_output_dir = "."


    # Call the robustness evaluation function FOR MULTIPLE SNR LEVELS
    # snr_levels_to_test = [20, 0, -20] # Use the defined list
    all_snr_results = []

    fig, axes = plt.subplots(len(TARGET_SNR_DB_LIST), 1, figsize=(15, 6 * len(TARGET_SNR_DB_LIST)), sharex=True)
    if len(TARGET_SNR_DB_LIST) == 1: # Handle case of single SNR level for axes indexing
        axes = [axes]

    print("\n--- Starting Robustness Evaluation for Multiple SNR Levels ---")
    for i, snr_db in enumerate(TARGET_SNR_DB_LIST):
        # Pass the specific axis for the current SNR level and the chosen original signal sample
        results = evaluate_robustness(
            model,
            X_test, # Full X_test is still needed for evaluation and generating full X_test_noisy
            y_test,
            DEFAULT_BATCH_SIZE,
            current_output_dir,
            target_snr_db=snr_db,
            ax=axes[i],
            original_signal_sample_for_plot=original_signal_to_plot # Pass the selected sample
        )
        if results:
            all_snr_results.append(results)

    # Save the consolidated plot
    plot_save_path = "src/main/original_vs_noisy_signals_subplots.png"
    try:
        fig.tight_layout()
        fig.savefig(plot_save_path)
        print(f"\nConsolidated comparison plot saved to {plot_save_path}")
        plt.close(fig)
    except Exception as e:
        print(f"Error saving consolidated plot: {e}")

    # Save all results to a single CSV
    if all_snr_results:
        results_df = pd.DataFrame(all_snr_results)
        csv_save_path = "src/main/evaluation_results_noisy_multi_snr.csv"
        try:
            results_df.to_csv(csv_save_path, index=False)
            print(f"All robustness evaluation results saved to {csv_save_path}")
        except Exception as e:
            print(f"Error saving multi-SNR results to CSV: {e}")
    else:
        print("No robustness evaluation results were collected to save.")


if __name__ == "__main__":
    main()
