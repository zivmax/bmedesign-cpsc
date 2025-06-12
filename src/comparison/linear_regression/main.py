import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    classification_report,
)
from sklearn.preprocessing import StandardScaler
import sys
import os

# Adjust the Python path to include the parent directory of 'src'
current_script_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_script_path)
# Navigate up to 'src' directory: linear_regression -> comparison -> src
src_dir = os.path.dirname(os.path.dirname(current_dir))
# Navigate up to project root: src -> project_root
project_root = os.path.dirname(src_dir)
sys.path.insert(0, project_root)

from src.utils.pipeline import LabelPipeline  # type: ignore

if __name__ == "__main__":
    SEED = 42
    np.random.seed(SEED)

    base_path = project_root

    train_df_path = os.path.join(base_path, "data/traindata.csv")
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
            "Exiting: Pseudo-labels are required for Logistic Regression baseline based on pseudo-labeled data."
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

    # Scale features for Logistic Regression
    scaler = StandardScaler()
    labeled_signals_scaled = scaler.fit_transform(labeled_signals)

    X_train, X_val, y_train_pseudo, y_val_pseudo = train_test_split(
        labeled_signals_scaled,
        targets,
        test_size=0.2,
        random_state=SEED,
        stratify=targets,
        shuffle=True,
    )

    print(f"Logistic Regression training set size: {X_train.shape[0]}")
    print(f"Logistic Regression validation set size: {X_val.shape[0]}")

    # Logistic Regression Model
    log_reg_model = LogisticRegression(
        random_state=SEED, solver="liblinear", max_iter=1000
    )

    print(
        "\nTraining Logistic Regression on the training portion of pseudo-labeled data..."
    )
    log_reg_model.fit(X_train, y_train_pseudo)

    # Evaluate on Training Set (Pseudo-labels)
    print(
        "\nEvaluating Logistic Regression on the training portion of pseudo-labeled data (for reference):"
    )
    y_train_pred_log_reg = log_reg_model.predict(X_train)
    print(
        f"Training F1 Score: {f1_score(y_train_pseudo, y_train_pred_log_reg, average='binary', zero_division=0):.4f}"
    )
    print(
        f"Training Precision Score: {precision_score(y_train_pseudo, y_train_pred_log_reg, average='binary', zero_division=0):.4f}"
    )
    print(
        f"Training Recall Score: {recall_score(y_train_pseudo, y_train_pred_log_reg, average='binary', zero_division=0):.4f}"
    )
    print(
        f"Training Accuracy: {accuracy_score(y_train_pseudo, y_train_pred_log_reg):.4f}"
    )
    print("Training Classification Report:")
    print(
        classification_report(
            y_train_pseudo,
            y_train_pred_log_reg,
            target_names=["Class 0", "Class 1"],
            zero_division=0,
        )
    )

    # Evaluate on Validation Set (Pseudo-labels)
    print(
        "\nEvaluating Logistic Regression on the validation portion of pseudo-labeled data:"
    )
    y_val_pred_log_reg = log_reg_model.predict(X_val)
    val_f1 = f1_score(
        y_val_pseudo, y_val_pred_log_reg, average="binary", zero_division=0
    )
    val_precision = precision_score(
        y_val_pseudo, y_val_pred_log_reg, average="binary", zero_division=0
    )
    val_recall = recall_score(
        y_val_pseudo, y_val_pred_log_reg, average="binary", zero_division=0
    )
    val_accuracy = accuracy_score(y_val_pseudo, y_val_pred_log_reg)
    print(f"Validation F1 Score: {val_f1:.4f}")
    print(f"Validation Precision Score: {val_precision:.4f}")
    print(f"Validation Recall Score: {val_recall:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print("Validation Classification Report:")
    print(
        classification_report(
            y_val_pseudo,
            y_val_pred_log_reg,
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
        current_dir, "logistic_regression_validation_evaluation_metrics.csv"
    )
    validation_metrics_df.to_csv(validation_metrics_save_path, index=False)
    print(
        f"Logistic Regression validation evaluation metrics saved to {validation_metrics_save_path}"
    )

    print("\nLogistic Regression baseline script finished.")
