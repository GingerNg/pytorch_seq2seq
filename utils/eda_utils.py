import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus']=False   #这两行需要手动设置

class EDA:
    #    def __init__(self,file_path):
    #        self.file_path=file_path
    # 获得数据
    def get_data(file_path):
        file_type = file_path.split('.')[1]
        if file_type == 'txt':
            df = pd.read_table(file_path)
        elif file_type == 'csv':
            df = pd.read_csv(file_path)
        elif file_type == 'xls' or file_type == 'xlsx':
            df = pd.read_excel(file_path)
        return df
    # 获取每列数据类型

    def get_col_type(df):
        #        df=get_data(file_path)
        label_col = list(set(df.columns)-set(df.describe().columns))
        print(label_col)
        other_col = list(set(df.columns)-set(label_col))
        print(other_col)
        num_col = []
        for col in other_col:
            if len(set(df[col])) < 10:
                label_col.append(col)
            else:
                num_col.append(col)
        return label_col, num_col
    # 绘图看数据分布

    def eda_plot(df, label_col, num_col):
        df_count = df.shape[0]
        df_col = df.columns.tolist()
        k = 0
        for col in df_col:
            # 对数值型数据进行直方图，箱线图，小提琴图
            if col in num_col:
                if math.floor(len(set(df[col]))*100/df_count) < 50:
                    df = df[(df[col].notnull())].sort_values(
                        col, ascending=True).reset_index(drop=True)
                    k = k + 1
                    plt.figure(k)
                    plt.subplot(1, 3, 1)
                    plt.hist(df[col])
                    plt.subplot(1, 3, 2)
                    plt.boxplot(df[col])
                    plt.gca().set_title('  %s  分布' % col)
                    plt.subplot(1, 3, 3)
                    plt.violinplot(df[col])
                    plt.tight_layout()

            elif col in label_col:
                # 类别型数据画柱形图和饼图
                if math.floor(len(set(df[col]))*100/df_count) < 5:
                    # k=k+1
                    # df_0=pd.DataFrame(df[col].index.values.tolist(),columns=[col])
                    df_0 = df.groupby([col]).agg({col: ['count']})
                    df_1 = df_0.iloc[:, 0].tolist()
                    #df_0['count'] = df.groupby([col]).agg({col:['count']})
                    x_axle = range(len(set(df[col].dropna())))
                    y_axle = df_1
                    x_label = list(set(df[col].dropna()))
                    fig, (ax1, ax2) = plt.subplots(2)
                    ax1.bar(x_axle, y_axle)
                    ax1.set_xticks(x_axle)
                    ax1.set_xticklabels(x_label)
                    ax1.set_title(' %s  分布' % (col))
                    ax2.pie(y_axle, labels=x_label, autopct='%1.2f%%')
        else:
            pass


if __name__ == "__main__":
    df = EDA.get_data(
        "/home/ginger/Projects/ND/ppt_server/ppt_auto_layout/datasets/data_ppt/v4/db/37_work_template/ppt_scd.csv")
    # print(df)
    label_col, num_col = EDA.get_col_type(df)
    EDA.eda_plot(df, label_col, num_col)
