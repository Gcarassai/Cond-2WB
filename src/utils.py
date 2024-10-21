# PCA
from sklearn.decomposition import PCA

def choose_pca(benchmark,DIM):
    pca = PCA(n_components=2)

    class Identity:
        pass

    if benchmark.bar_sampler is not None:
        pca.fit(benchmark.bar_sampler.sample(100000).cpu().detach().numpy())
    elif benchmark.gauss_bar_sampler is not None:
        pca.fit(benchmark.gauss_bar_sampler.sample(100000).cpu().detach().numpy())
    else:
        pca = Identity()
        pca.transform = lambda x: x
        
    # No PCA for dim=2
    if DIM == 2:
        pca = Identity()
        pca.transform = lambda x: x

    return pca

# Split datasets

from sklearn.model_selection import train_test_split
import pandas as pd

def split_data(data_all_x, data_all_s, data_all_y, test_OT_size=0.3, test_models_size=0.3, ot_valid_size=0.1, random_state=42, verbose=True):
    """
    Function that creates data splits to train OT, and test on insurance models
    :param data_all_x: X features data
    :param data_all_s: sensitive data
    :param data_all_y: target data
    :param test_OT_size: size of the test set for OT
    :param test_models_size: size of the test set for the models
    :param ot_valid_size: size of the validation set for OT
    :param random_state: random state
    :return: train and test splits for OT, unfair and fair models
    """

    # convert to pandas dataframe if not already
    if not isinstance(data_all_x, pd.DataFrame):
        data_all_x = pd.DataFrame(data_all_x)
        data_all_s = pd.DataFrame(data_all_s)
        data_all_y = pd.DataFrame(data_all_y)

    X_train_OT, X_test_OT, S_train_OT, S_test_OT, Y_train_OT, Y_test_OT = train_test_split(data_all_x, data_all_s, data_all_y, test_size=test_OT_size, random_state=random_state, shuffle = True)
    X_train_unfair, X_test_unfair, S_train_unfair, S_test_unfair, Y_train_unfair, Y_test_unfair = train_test_split(X_train_OT, S_train_OT, Y_train_OT, train_size=(1-test_models_size)*test_OT_size/(1-test_OT_size) ,test_size=test_models_size*test_OT_size/(1-test_OT_size), random_state=random_state, shuffle = True)
    X_train_fair, X_test_fair, S_train_fair, S_test_fair, Y_train_fair, Y_test_fair = train_test_split(X_test_OT, S_test_OT, Y_test_OT, test_size=test_models_size, random_state=random_state, shuffle = True)
    X_train_OT, X_valid_OT, S_train_OT, S_valid_OT = train_test_split(X_train_OT, S_train_OT, test_size=ot_valid_size, random_state=random_state, shuffle = True)

    if verbose:
        print("length OT train:", len(X_train_OT))
        print("length OT test:", len(X_test_OT))
        print("length unfair train:", len(X_train_unfair))
        print("length unfair test:", len(X_test_unfair))
        print("length fair train:", len(X_train_fair))
        print("length fair test:", len(X_test_fair))
        print("length OT valid:", len(X_valid_OT))

    return X_train_OT, X_test_OT, X_valid_OT, S_train_OT, S_test_OT,S_valid_OT, Y_train_OT, Y_test_OT, X_train_unfair, X_test_unfair, S_train_unfair, S_test_unfair, Y_train_unfair, Y_test_unfair, X_train_fair, X_test_fair, S_train_fair, S_test_fair, Y_train_fair, Y_test_fair
