from sklearn.model_selection import train_test_split
from sklearn import utils
import random


class Slicer(object):
    """[切片器, 分片]

    Args:
        object ([type]): [description]
    """

    def __init__(self):
        pass

    def split(self):
        pass


class SklearnSlicer(Slicer):
    def __init__(self):
        super().__init__()

    def split(self, data, fea_col_name, label_col_name):
        # 划分训练集和测试集
        X = data[fea_col_name]
        y = data[label_col_name]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=1)
        return X_train, X_test, y_train, y_test


class ShuffleSlicer(Slicer):
    def __init__(self):
        super().__init__()

    def split(self, df, dev=False):
        if isinstance(df, list):
            df = random.shuffle(df)
        else:
            df = utils.shuffle(df)
        if dev:
            fold_len = df.shape[0]//5
            train_df = df[0:fold_len*3]
            dev_df = df[fold_len*3:fold_len*4]
            test_df = df[fold_len*4:]
            # train_df = df[0:100]
            # dev_df = df[100:200]
            # test_df = df[200:300]
            return train_df, dev_df, test_df
        else:
            train_len = df.shape[0]//5 * 4
            train_df = df[0:train_len]
            test_df = df[train_len:]
            return train_df, test_df
