import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
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
# Navigate up to 'src' directory: gaussian_mixture -> comparison -> src
src_dir = os.path.dirname(os.path.dirname(current_dir))
# Navigate up to project root: src -> project_root
project_root = os.path.dirname(src_dir)
sys.path.insert(0, project_root)

from src.utils.pipeline import LabelPipeline  # type: ignore


def map_clusters_to_labels(cluster_labels, true_labels):
    """
    Maps GMM cluster labels to true class labels.
    Assumes binary classification and n_components=2 for GMM.
    """
    # Ensure true_labels is a numpy array for consistent indexing
    true_labels = np.asarray(true_labels)
    cluster_labels = np.asarray(cluster_labels)

    # Create a mapping for cluster 0
    cluster0_mask = cluster_labels == 0
    if np.sum(cluster0_mask) == 0:  # No samples in cluster 0
        if np.sum(cluster_labels == 1) > 0:  # Check if cluster 1 has samples
            # If cluster 0 is empty, cluster 1 gets its mode, cluster 0 gets the other label
            label_for_cluster1 = pd.Series(true_labels[cluster_labels == 1]).mode()[0]
            label_for_cluster0 = 1 - label_for_cluster1
        else:  # Both clusters are empty (highly unlikely with data)
            label_for_cluster0 = 0
            label_for_cluster1 = 1
    else:
        label_for_cluster0 = pd.Series(true_labels[cluster0_mask]).mode()[0]

    # Create a mapping for cluster 1
    cluster1_mask = cluster_labels == 1
    if np.sum(cluster1_mask) == 0:  # No samples in cluster 1
        # If cluster 1 is empty, it gets the label not taken by cluster 0
        label_for_cluster1 = 1 - label_for_cluster0
    else:
        mode_cluster1 = pd.Series(true_labels[cluster1_mask]).mode()[0]
        if mode_cluster1 == label_for_cluster0:
            label_for_cluster1 = 1 - label_for_cluster0  # Assign the other label
        else:
            label_for_cluster1 = mode_cluster1

    # Fallback: if both clusters mapped to the same label, try to resolve
    # This can happen if data isn't well separated or one true class is very dominant in both clusters
    if (
        label_for_cluster0 == label_for_cluster1
        and np.sum(cluster0_mask) > 0
        and np.sum(cluster1_mask) > 0
    ):
        # Option 1: cluster 0 -> 0, cluster 1 -> 1
        mapping1 = {0: 0, 1: 1}
        preds1 = np.array(
            [mapping1.get(l, l) for l in cluster_labels]
        )  # .get for safety if unexpected cluster label
        acc1 = accuracy_score(true_labels, preds1)

        # Option 2: cluster 0 -> 1, cluster 1 -> 0
        mapping2 = {0: 1, 1: 0}
        preds2 = np.array([mapping2.get(l, l) for l in cluster_labels])
        acc2 = accuracy_score(true_labels, preds2)

        if acc1 >= acc2:
            label_for_cluster0 = 0
            label_for_cluster1 = 1
        else:
            label_for_cluster0 = 1
            label_for_cluster1 = 0

    return {0: label_for_cluster0, 1: label_for_cluster1}


if __name__ == "__main__":
    SEED = 42
    np.random.seed(SEED)

    base_path = project_root

    train_df_path = os.path.join(base_path, "data/traindata.csv")
    # test_df_path = os.path.join(base_path, "data/testdata.csv") # Not used for prediction
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
            "Exiting: Pseudo-labels are required for GMM baseline based on pseudo-labeled data."
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

    print(f"GMM training set size: {X_train.shape[0]}")
    print(f"GMM validation set size: {X_val.shape[0]}")

    # Gaussian Mixture Model
    gmm_model = GaussianMixture(
        n_components=2, random_state=SEED, covariance_type="full"
    )

    print("\nTraining GMM on the training portion of pseudo-labeled data...")
    gmm_model.fit(X_train)  # Fit GMM

    # Predict cluster assignments for the training data
    train_cluster_labels = gmm_model.predict(X_train)

    print("Mapping GMM clusters to pseudo-labels using training data...")
    cluster_to_class_mapping = map_clusters_to_labels(
        train_cluster_labels, y_train_pseudo
    )
    print(f"GMM cluster to class mapping: {cluster_to_class_mapping}")

    y_train_pred_gmm = np.array(
        [cluster_to_class_mapping[label] for label in train_cluster_labels]
    )

    print(
        "\nEvaluating GMM on the training portion of pseudo-labeled data (for reference):"
    )
    print(
        f"Training F1 Score: {f1_score(y_train_pseudo, y_train_pred_gmm, average='binary', zero_division=0):.4f}"
    )
    print(
        f"Training Precision Score: {precision_score(y_train_pseudo, y_train_pred_gmm, average='binary', zero_division=0):.4f}"
    )
    print(
        f"Training Recall Score: {recall_score(y_train_pseudo, y_train_pred_gmm, average='binary', zero_division=0):.4f}"
    )
    print(f"Training Accuracy: {accuracy_score(y_train_pseudo, y_train_pred_gmm):.4f}")
    print("Training Classification Report:")
    print(
        classification_report(
            y_train_pseudo,
            y_train_pred_gmm,
            target_names=["Class 0", "Class 1"],
            zero_division=0,
        )
    )

    # Predict clusters on the validation set
    print("\nPredicting GMM clusters on the validation set...")
    val_cluster_labels = gmm_model.predict(X_val)
    y_val_pred_gmm = np.array(
        [cluster_to_class_mapping[label] for label in val_cluster_labels]
    )

    print("\nEvaluating GMM on the validation portion of pseudo-labeled data:")
    val_f1 = f1_score(y_val_pseudo, y_val_pred_gmm, average="binary", zero_division=0)
    val_precision = precision_score(
        y_val_pseudo, y_val_pred_gmm, average="binary", zero_division=0
    )
    val_recall = recall_score(
        y_val_pseudo, y_val_pred_gmm, average="binary", zero_division=0
    )
    val_accuracy = accuracy_score(y_val_pseudo, y_val_pred_gmm)
    print(f"Validation F1 Score: {val_f1:.4f}")
    print(f"Validation Precision Score: {val_precision:.4f}")
    print(f"Validation Recall Score: {val_recall:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print("Validation Classification Report:")
    print(
        classification_report(
            y_val_pseudo,
            y_val_pred_gmm,
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
    # Save in the current script's directory (src/comparison/gaussian_mixture/)
    validation_metrics_save_path = os.path.join(
        current_dir, "gmm_validation_evaluation_metrics.csv"
    )
    validation_metrics_df.to_csv(validation_metrics_save_path, index=False)
    print(f"GMM validation evaluation metrics saved to {validation_metrics_save_path}")

    print("\nGMM baseline script finished.")
