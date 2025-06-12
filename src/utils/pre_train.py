import pandas as pd
import numpy as np
from mantis.architecture import Mantis8M
from mantis.trainer import MantisTrainer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

# Load data
# Replaced h5py loading with pandas for the CSV file.
# Assumes the script is run from the workspace root, so 'data/traindata.csv' is the correct path.
# Assumes the CSV has no header row.
df = pd.read_csv('data/traindata.csv', header=None)
traindata = df.values  # Data is (samples, features), matching previous shape after transpose.

# Data partitioning
# First 500 records are atrial fibrillation (label=1), 501-1000 are non-atrial fibrillation (label=0), the rest are unlabeled data
labels = np.array([1] * 500 + [0] * 500 + [None] * (20000 - 1000))
labeled_indices = np.where(labels != None)[0]  # Indices of labeled data
unlabeled_indices = np.where(labels == None)[0]  # Indices of unlabeled data

# Extract labeled data
X_labeled = traindata[labeled_indices]
y_labeled = labels[labeled_indices].astype(int)

# Extract unlabeled data
X_unlabeled = traindata[unlabeled_indices]

# Data standardization
scaler = StandardScaler()
X_labeled = scaler.fit_transform(X_labeled)
X_unlabeled = scaler.transform(X_unlabeled)

# Split training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_labeled, y_labeled, test_size=0.2, random_state=42, stratify=y_labeled)

# Adjust data shape to 3D
# Convert (batch_size, sequence_length) to (batch_size, num_channels, sequence_length)
X_train = X_train[:, np.newaxis, :]  # Add a pseudo channel dimension
X_val = X_val[:, np.newaxis, :]
X_unlabeled = X_unlabeled[:, np.newaxis, :]

# Load Mantis model
device = 'cuda'  # Set to 'cuda' if GPU is available, otherwise use 'cpu'
network = Mantis8M(device=device)
network = network.from_pretrained("paris-noah/Mantis-8M")

# Initialize MantisTrainer
model = MantisTrainer(device=device, network=network)

# Feature extraction
print("Extracting features from training and validation data...")
X_train_features = model.transform(X_train)  # 3D input
X_val_features = model.transform(X_val)      # 3D input

# Train using scikit-learn or other classifiers
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(random_state=42)
classifier.fit(X_train_features, y_train)

# Validate model performance
y_val_pred = classifier.predict(X_val_features)
print("Validation Classification Report:")
print(classification_report(y_val, y_val_pred, target_names=['Non-AF', 'AF']))

# Predict unlabeled data
print("Predicting unlabeled data...")
X_unlabeled_features = model.transform(X_unlabeled)  # 3D input
y_unlabeled_pred = classifier.predict(X_unlabeled_features)

# Save predictions to a CSV file using pandas
predictions_df = pd.DataFrame(y_unlabeled_pred)
csv_output_path = 'test/unlabeled_predictions.csv'
predictions_df.to_csv(csv_output_path, header=False, index=False)

print(f"Prediction results saved to '{csv_output_path}'")