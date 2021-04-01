from sklearn.metrics import f1_score, precision_score


def f1_index(y_true, y_pred):
    return f1_score(y_true, y_pred)


def micro_f1_index(y_true, y_pred):
    # 通过先计算总体的TP，FN和FP的数量，再计算F1
    return f1_score(y_true, y_pred, average="micro")


def macro_f1_index(y_true, y_pred):
    # 各类平均
    return f1_score(y_true, y_pred, average="macro")


if __name__ == "__main__":
    preds = [1, 2, 1, 1, 3]
    labels = [1, 2, 3, 1, 3]
    print(macro_f1_index(preds, labels))
