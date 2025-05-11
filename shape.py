# import h5py


# # 1. 加载 .mat 文件
# file_path = "./data/unlabeled_predictions.mat"  # 替换为你的 .mat 文件路径
# with h5py.File(file_path, "r") as f:
#     # 2. 列出所有变量
#     print("MAT 文件中的变量：")
#     print(list(f.keys()))

#     # 3. 遍历变量，查看形状和前几条数据
#     for var_name in f.keys():
#         data = f[var_name]
#         print(f"\n变量 '{var_name}' 的信息：")
#         print(f"  形状：{data.shape}")
        
#         # 4. 打印前几条数据（如果数据量较大）
#         if data.ndim == 1:  # 一维数据
#             print(f"  前几条数据：{data[:5]}")
#         elif data.ndim == 2:  # 二维数据
#             print(f"  前几条数据：\n{data[:5, :5]}")  # 前 5 行 5 列
#         else:  # 多维数据
#             print("  数据维度超过 2D，请根据需要手动查看。")


import numpy as np

# 加载预测结果
import scipy.io as sio
data = sio.loadmat('./data/unlabeled_predictions.mat')
predictions = data['unlabeled_predictions']

# 保存成 CSV 文件
np.savetxt('./data/unlabeled_predictions.csv', predictions, delimiter=',', fmt='%d')
print("Predictions saved to './data/unlabeled_predictions.csv'")