import h5py
import numpy as np
from mantis.architecture import Mantis8M
from mantis.trainer import MantisTrainer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

# 加载数据
with h5py.File('./data/traindata.mat', 'r') as f:
    # 加载变量 'traindata'
    traindata = np.array(f['traindata']).T  # 转置为 (20000, 4000)，每行为一条信号

# 数据划分
# 前 500 条为房颤 (label=1)，501-1000 条为非房颤 (label=0)，其余为无标签数据
labels = np.array([1] * 500 + [0] * 500 + [None] * (20000 - 1000))
labeled_indices = np.where(labels != None)[0]  # 有标签数据索引
unlabeled_indices = np.where(labels == None)[0]  # 无标签数据索引

# 提取有标签数据
X_labeled = traindata[labeled_indices]
y_labeled = labels[labeled_indices].astype(int)

# 提取无标签数据
X_unlabeled = traindata[unlabeled_indices]

# 数据标准化
scaler = StandardScaler()
X_labeled = scaler.fit_transform(X_labeled)
X_unlabeled = scaler.transform(X_unlabeled)

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X_labeled, y_labeled, test_size=0.2, random_state=42, stratify=y_labeled)

# 调整数据形状为 3D
# 将 (batch_size, sequence_length) 转换为 (batch_size, num_channels, sequence_length)
X_train = X_train[:, np.newaxis, :]  # 添加一个伪通道维度
X_val = X_val[:, np.newaxis, :]
X_unlabeled = X_unlabeled[:, np.newaxis, :]

# 加载 Mantis 模型
device = 'cuda'  # 如果有 GPU 可用，设置为 'cuda'，否则使用 'cpu'
network = Mantis8M(device=device)
network = network.from_pretrained("paris-noah/Mantis-8M")

# 初始化 MantisTrainer
model = MantisTrainer(device=device, network=network)

# 特征提取
print("Extracting features from training and validation data...")
X_train_features = model.transform(X_train)  # 3D 输入
X_val_features = model.transform(X_val)      # 3D 输入

# 使用 scikit-learn 或其他分类器进行训练
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(random_state=42)
classifier.fit(X_train_features, y_train)

# 验证模型性能
y_val_pred = classifier.predict(X_val_features)
print("Validation Classification Report:")
print(classification_report(y_val, y_val_pred, target_names=['Non-AF', 'AF']))

# 对无标签数据进行预测
print("Predicting unlabeled data...")
X_unlabeled_features = model.transform(X_unlabeled)  # 3D 输入
y_unlabeled_pred = classifier.predict(X_unlabeled_features)

# 将预测结果保存到文件
import scipy.io as sio
output = {
    'unlabeled_predictions': y_unlabeled_pred
}
sio.savemat('./data/unlabeled_predictions.mat', output)

print("Prediction results saved to './data/unlabeled_predictions.mat'")