from scipy.stats import ks_2samp
import numpy as np

def compute_KS(y_pred, S):
    """
    y_pred: np.array of predicted scores
    S: np.array of sensitive attribute
    """
    groups = []
    for s in np.unique(S):
        groups.append(y_pred[S.squeeze() == s])
    ks = 0.
    for i in range(len(groups)):
        for j in range(len(groups)):
            if j>i:
                ks_stat, p_value = ks_2samp(groups[i], groups[j])
                if ks < ks_stat:
                    ks = ks_stat
    return ks

