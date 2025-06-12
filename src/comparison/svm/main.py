import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    classification_report,
)
import sys
import os

# Adjust the Python path to include the parent directory of 'src'
current_script_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_script_path)
src_dir = os.path.dirname(os.path.dirname(current_dir))  # up to comparison, then src
project_root = os.path.dirname(src_dir)
sys.path.insert(0, project_root)

from src.utils.pipeline import LabelPipeline  # type: ignore

if __name__ == "__main__":
    SEED = 42
    np.random.seed(SEED)

    base_path = project_root

    train_df_path = os.path.join(base_path, "data/traindata.csv")
    test_df_path = os.path.join(base_path, "data/testdata.csv")
    unlabeled_preds_path = os.path.join(base_path, "data/unlabeled_predictions.csv")

    print(f"Loading training data from: {train_df_path}")
    try:
        train_df = pd.read_csv(train_df_path)
    except FileNotFoundError:
        print(f"Error: Training data file not found at {train_df_path}. Exiting.")
        sys.exit(1)

    print(f"Loading pseudo-labels from: {unlabeled_preds_path}")
    try:
        temp_df = pd.read_csv(unlabeled_preds_path, header=None)
    except FileNotFoundError:
        print(
            f"Error: Pseudo-labels file not found at {unlabeled_preds_path}. Exiting."
        )
        sys.exit(1)

    labels_for_pseudo = None
    if temp_df.shape[0] > 0 and temp_df.shape[1] > 0:
        labels_for_pseudo = temp_df.iloc[:, 0]
        print(
            f"Info: Loaded {len(labels_for_pseudo)} pseudo-labels from {unlabeled_preds_path}."
        )
    else:
        print(
            f"Warning: {unlabeled_preds_path} is empty or has no columns. No pseudo-labels loaded."
        )
        sys.exit(
            "Exiting: Pseudo-labels are required for SVM baseline based on pseudo-labeled data."
        )

    lp = LabelPipeline(train_df)
    labeled_signals, targets = lp.add_labels(labels_for_pseudo, cutedge=(500, 1000))

    print("Original pseudo-labels distribution:")
    print(targets.value_counts(normalize=True))

    if not isinstance(labeled_signals, pd.DataFrame):
        labeled_signals = pd.DataFrame(labeled_signals)
    if not isinstance(targets, pd.Series):
        targets = pd.Series(targets)

    print(
        f"Total labeled signals: {labeled_signals.shape[0]}, Total targets: {targets.shape[0]}"
    )
    if labeled_signals.empty or targets.empty:
        raise ValueError(
            "Data loading or processing resulted in empty signals or targets."
        )

    X_train, X_val, y_train_pseudo, y_val_pseudo = train_test_split(
        labeled_signals,
        targets,
        test_size=0.2,
        random_state=SEED,
        stratify=targets,
        shuffle=True,
    )

    print(f"SVM training set size: {X_train.shape[0]}")
    print(f"SVM validation set size: {X_val.shape[0]}")

    # SVM Model
    # Using a simple RBF kernel SVM. Parameters can be tuned.
    svm_model = SVC(kernel="rbf", random_state=SEED, C=1.0, gamma="scale")

    print("\nTraining SVM on the training portion of pseudo-labeled data...")
    svm_model.fit(X_train, y_train_pseudo)

    # Evaluate on Training Set (Pseudo-labels)
    print(
        "\nEvaluating SVM on the training portion of pseudo-labeled data (for reference):"
    )
    y_train_pred_svm = svm_model.predict(X_train)
    print(
        f"Training F1 Score: {f1_score(y_train_pseudo, y_train_pred_svm, average='binary', zero_division=0):.4f}"
    )
    print(
        f"Training Precision Score: {precision_score(y_train_pseudo, y_train_pred_svm, average='binary', zero_division=0):.4f}"
    )
    print(
        f"Training Recall Score: {recall_score(y_train_pseudo, y_train_pred_svm, average='binary', zero_division=0):.4f}"
    )
    print(f"Training Accuracy: {accuracy_score(y_train_pseudo, y_train_pred_svm):.4f}")
    print("Training Classification Report:")
    print(
        classification_report(
            y_train_pseudo,
            y_train_pred_svm,
            target_names=["Class 0", "Class 1"],
            zero_division=0,
        )
    )

    # Evaluate on Validation Set (Pseudo-labels)
    print("\nEvaluating SVM on the validation portion of pseudo-labeled data:")
    y_val_pred_svm = svm_model.predict(X_val)
    val_f1 = f1_score(y_val_pseudo, y_val_pred_svm, average="binary", zero_division=0)
    val_precision = precision_score(
        y_val_pseudo, y_val_pred_svm, average="binary", zero_division=0
    )
    val_recall = recall_score(
        y_val_pseudo, y_val_pred_svm, average="binary", zero_division=0
    )
    val_accuracy = accuracy_score(y_val_pseudo, y_val_pred_svm)
    print(f"Validation F1 Score: {val_f1:.4f}")
    print(f"Validation Precision Score: {val_precision:.4f}")
    print(f"Validation Recall Score: {val_recall:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print("Validation Classification Report:")
    print(
        classification_report(
            y_val_pseudo,
            y_val_pred_svm,
            target_names=["Class 0", "Class 1"],
            zero_division=0,
        )
    )

    # Save validation metrics
    validation_metrics_data = {
        "Metric": ["F1 Score", "Precision", "Recall", "Accuracy"],
        "Score": [val_f1, val_precision, val_recall, val_accuracy],
    }
    validation_metrics_df = pd.DataFrame(validation_metrics_data)
    validation_metrics_save_path = os.path.join(
        current_dir, "svm_validation_evaluation_metrics.csv"
    )
    validation_metrics_df.to_csv(validation_metrics_save_path, index=False)
    print(f"SVM validation evaluation metrics saved to {validation_metrics_save_path}")

    # Prediction and Evaluation on Actual Test Set (REMOVED as per request)
    # test_df = None
    # try:
    #     print(f"\\nLoading test data from: {test_df_path}")
    #     test_df = pd.read_csv(test_df_path)
    # except FileNotFoundError:
    #     print(f"Warning: Test data file not found at {test_df_path}. Skipping predictions on test set.")

    # if test_df is not None and not test_df.empty:
    #     print("Processing test data for SVM prediction...")
    #
    #     y_test_true = None
    #     if 'label' in test_df.columns:
    #         print("Found 'label' column in test data. Will use for evaluation.")
    #         y_test_true = test_df['label'].to_numpy()
    #         test_signals_for_svm = test_df.drop(columns=['label'])
    #     else:
    #         print("Warning: No 'label' column in test_df. Using all columns for SVM input. Cannot evaluate test set performance.")
    #         test_signals_for_svm = test_df.copy()

    #     if test_signals_for_svm.shape[1] == X_train.shape[1]:
    #         print(f"Predicting on test set with {test_signals_for_svm.shape[0]} samples...")
    #         test_pred_svm = svm_model.predict(test_signals_for_svm)
    #
    #         if y_test_true is not None:
    #             if len(y_test_true) == len(test_pred_svm):
    #                 print("\\nEvaluating SVM on the actual test data:")
    #                 test_f1 = f1_score(y_test_true, test_pred_svm, average='binary', zero_division=0)
    #                 test_precision = precision_score(y_test_true, test_pred_svm, average='binary', zero_division=0)
    #                 test_recall = recall_score(y_test_true, test_pred_svm, average='binary', zero_division=0)
    #                 test_accuracy = accuracy_score(y_test_true, test_pred_svm)
    #
    #                 print(f"Test F1 Score: {test_f1:.4f}")
    #                 print(f"Test Precision Score: {test_precision:.4f}")
    #                 print(f"Test Recall Score: {test_recall:.4f}")
    #                 print(f"Test Accuracy: {test_accuracy:.4f}")
    #                 print("Test Classification Report:")
    #                 print(classification_report(y_test_true, test_pred_svm, target_names=['Class 0', 'Class 1'], zero_division=0))

    #                 metrics_data = {
    #                     'Metric': ['F1 Score', 'Precision', 'Recall', 'Accuracy'],
    #                     'Score': [test_f1, test_precision, test_recall, test_accuracy]
    #                 }
    #                 metrics_df = pd.DataFrame(metrics_data)
    #                 metrics_save_path = os.path.join(current_dir, 'svm_test_evaluation_metrics.csv')
    #                 metrics_df.to_csv(metrics_save_path, index=False)
    #                 print(f"SVM test evaluation metrics saved to {metrics_save_path}")
    #             else:
    #                 print(f"Warning: Mismatch in length between true test labels ({len(y_test_true)}) "
    #                       f"and predicted test labels ({len(test_pred_svm)}). Skipping test evaluation and saving of metrics.")
    #         else:
    #             print("No true labels available for the test set. Cannot calculate or save evaluation metrics.")

    #     else:
    #         print(f"Warning: Test data feature mismatch. Expected {X_train.shape[1]} features, got {test_signals_for_svm.shape[1]}. Skipping test set prediction.")
    #
    # else:
    #     print("Skipping prediction on external test set as test_df is not available or empty.")

    print("\\nSVM baseline script finished.")
