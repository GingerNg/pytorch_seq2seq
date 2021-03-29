import numpy as np


def cos_dist(vector1, vector2: np.array):
    # print(vector1, vector2)
    num = float(np.sum(vector1 * vector2))  # 若为行向量则 A * B.T
    denom = np.linalg.norm(vector1) * np.linalg.norm(vector2)
    cos = num / denom  # 余弦值
    return 0.5 + 0.5 * cos  # 归一化


def euclidean_dist(vector1, vector2):
    """欧式距离"""
    vector1, vector2 = np.mat(vector1), np.mat(vector2)
    return np.sqrt((vector1-vector2)*((vector1-vector2).T))[0][0]


#
