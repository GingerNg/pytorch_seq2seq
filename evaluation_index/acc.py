from sklearn.metrics import accuracy_score
import numpy as np

def acc_index(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

if __name__ == "__main__":
    preds = np.array([1, 2, 1, 1, 3])
    labels = np.array([1, 2, 3, 1, 3])
    r = labels == preds
    print(sum(r))