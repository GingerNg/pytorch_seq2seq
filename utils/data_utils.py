import numpy as np
import pandas as pd


def df2np_array(pd_series):
    """[df["label"] --> series]

    Args:
        pd_series ([series]): [description]

    Returns:
        [type]: [description]
    """
    return np.array(pd_series.tolist())


def df2list(data_df):
    return data_df.values.tolist()


def dict2df(d):
    """[summary]

    Args:
        d ([dict]): [description]

    Returns:
        [type]: [description]
    """
    # df = pd.DataFrame({'A': ['a', 'b', 'a', 'c', 'a', 'c', 'b', 'c'],
    #                'B': [[2,1], [8,1], [1,1], [4,2], [3,2], [2,2], [5,2], [9,8]]})
    return pd.DataFrame(d)


def list2df(datas, names):
    """[summary]

    Args:
        datas ([list]): [description]
        names ([list]): [description]

    Returns:
        [type]: [description]
    """
    # name = ['sentence']
    data_pd = pd.DataFrame(columns=names, data=datas)
    return data_pd
