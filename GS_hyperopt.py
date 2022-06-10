import os
import argparse
import pandas as pd
import numpy as np
import xgboost as xgb
from math import ceil
from math import sqrt
from math import acos
from operator import itemgetter
from itertools import product
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import gc
#import seaborn as sns
from scipy import stats
from collections import defaultdict
from tqdm import tqdm
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe , space_eval
import time

def load_data(vector_filename, ion_type):
    # Read file
    if vector_filename.split(".")[-1] == "pkl":
        vectors = pd.read_pickle(vector_filename)
    elif vector_filename.split(".")[-1] == "h5":
        # vectors = pd.read_hdf(vector_filename, key='table', stop=1000)
        vectors = pd.read_hdf(vector_filename, key="table")
    else:
        print("Unsuported feature vector format")
        exit(1)

    # Extract targets for given ion type
    target_names = list(vectors.columns[vectors.columns.str.contains("targets")])
    if not "targets_{}".format(ion_type) in target_names:
        print("Targets for {} could not be found in vector file.".format(ion_type))
        print("Vector file only contains these targets: {}".format(target_names))
        exit(1)

    targets = vectors.pop("targets_{}".format(ion_type))
    target_names.remove("targets_{}".format(ion_type))
    for n in target_names:
        vectors.pop(n)

    # Get psmids
    psmids = vectors.pop("psmid")

    return (vectors, targets, psmids)

def get_params_combinations(params):
    keys, values = zip(*params.items())
    combinations = [dict(zip(keys, v)) for v in product(*values)]
    return(combinations)

def get_best_params(df, params_grid):
    params = {}
    best = df[df['test-rmse-mean'] == df['test-rmse-mean'].min()]
    best_rmse = df['test-rmse-mean'].min()
    for p in params_grid.keys():
        params[p] = best[p].iloc[0]
    # num_boost_round = best['boosting-round'].iloc[0]
    return(params, best_rmse)

def gridsearch(xtrain, params, params_grid):
    cols = ['boosting-round', 'test-rmse-mean', 'test-rmse-std', 'train-rmse-mean', 'train-rmse-std']
    cols.extend(sorted(params_grid.keys()))
    result = pd.DataFrame(columns=cols)

    count = 1
    combinations = get_params_combinations(params_grid)

    for param_overrides in combinations:
        print("Working on combination {}/{}".format(count, len(combinations)))
        count += 1
        params.update(param_overrides)
        tmp = xgb.cv(params, xtrain, nfold=5, num_boost_round=200, early_stopping_rounds=10, verbose_eval=10)
        tmp['boosting-round'] = tmp.index
        for param in param_overrides.keys():
            tmp[param] = param_overrides[param]
        result = result.append(tmp)

    print("Grid search ready!\n")

    return(result)

def ms2pip_pearson(true, pred):
    """
    Return pearson of tic-normalized, log-transformed intensities, 
    the MS2PIP way.
    """
    #tic_norm = lambda x: x / np.sum(x)
    # log_transform = lambda x: np.log2(x + 0.001)
    corr = pearsonr(
        true, 
        pred
    )[0]
    return (corr)

def spectral_angle(true, pred, epsilon=1e-7):
    """
    Return square root normalized spectral angle.
    See https://doi.org/10.1074/mcp.O113.036475
    """
    
    de_log = lambda x: (2**x)-0.001
    l2_normalize = lambda x: x / sqrt(max(sum(x**2), epsilon))
    
    pred_norm = l2_normalize(de_log(pred))
    true_norm = l2_normalize(de_log(true))
    
    spectral_angle = 1 - (2 * acos(np.dot(pred_norm, true_norm)) / np.pi)

    return (spectral_angle)


class Scorer:
    def __init__(self,psmids):
        self.psmids = psmids

    def psm_score(self,targets, predictions):
        tmp = pd.DataFrame(columns=["psmids", "targets", "predictions"])
        tmp["psmids"] = np.array(self.psmids)
        tmp["targets"] = np.array(targets)
        tmp["predictions"] = np.array(predictions)
        tmp2 = tmp.groupby("psmids").agg({'predictions': list, 'targets': list}).reset_index()
        spectral_corr = []
        pearson_corr = []
        for spectra in range(0, len(tmp2["psmids"])):
            spectral_corr.append(spectral_angle(np.array(tmp2.targets.loc[spectra]), np.array(tmp2.predictions.loc[spectra])))
            pearson_corr.append(ms2pip_pearson(np.array(tmp2.targets.loc[spectra]), np.array(tmp2.predictions.loc[spectra])))
        return (pearson_corr, spectral_corr) 

vectors, targets, psmids = load_data("immuno/immunopeptide_spectra_rankTICnormalized_featvector.h5", "Y")

upeps = psmids.unique()
np.random.shuffle(upeps)
test_psms = upeps[:int(len(upeps) * 0.1)]

train_vectors = vectors[~psmids.isin(test_psms)]
train_targets = targets[~psmids.isin(test_psms)]
train_psmids = psmids[~psmids.isin(test_psms)]

test_vectors = vectors[psmids.isin(test_psms)]
test_targets = targets[psmids.isin(test_psms)]
test_psmids = psmids[psmids.isin(test_psms)]

xtrain = xgb.DMatrix(train_vectors, label=train_targets)
xtest = xgb.DMatrix(test_vectors, label=test_targets)
evallist = [(xtrain, 'train'),(xtest, 'test')]

space= {
    'eta': hp.loguniform('eta', np.log(0.01), np.log(1)),
    'max_depth': hp.quniform('max_depth', 3, 18, 1),
    'max_leaves': hp.quniform('max_leaves', 5, 500, 1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0),
    'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
    'gamma' : hp.uniform ('gamma', 0.0,1),
    'min_child_weight' : hp.quniform('min_child_weight', 0, 500, 1),
    'subsample': hp.quniform('subsample', 0.5, 1 ,0.1),
    'reg_alpha': hp.quniform('reg_alpha', 0, 5 ,0.1)
}

def objective(space):
    params = {
    "nthread": 64,
    "objective": "reg:squarederror",
    "eval_metric": 'rmse',
    "eta": space["eta"],
    "max_depth": int(space['max_depth']),
    "grow_policy":"lossguide",
    "max_leaves":int(space["max_leaves"]),
    "min_child_weight": int(space["min_child_weight"]),
    "gamma": space['gamma'],
    "reg_lambda" : space['reg_lambda'],
    "colsample_bytree": space['colsample_bytree'],
    "subsample" : space["subsample"],
    "reg_alpha" : space["reg_alpha"]
    }
    print(params)
    tmp = xgb.cv(params, xtrain, nfold=4, num_boost_round=400, early_stopping_rounds=10, verbose_eval=True)
    rmse = tmp['test-rmse-mean'].min()

    print ("rmse:", rmse)
    return {'loss': rmse, 'status': STATUS_OK, "params": params}

trials = Trials()
best_hyperparams = fmin(fn = objective,
                        space = space,
                        algo = tpe.suggest,
                        max_evals = 25,
                        trials = trials)


print("Best hyperparameters:", best_hyperparams)

hyperoptimization_results = {'test-rmse-mean': [x['loss'] for x in trials.results]}
for key in trials.results[0]["params"].keys():
    hyperoptimization_results[key] = [x['params'][key] for x in trials.results]

df = pd.DataFrame(hyperoptimization_results)

df.to_csv("immunopeptide_rankTIC_resultsY")

df_sorted = df.sort_values(by=['test-rmse-mean'], ascending=True)

params = {
    "nthread": df_sorted.iloc[0]["nthread"],
    "objective": df_sorted.iloc[0]["objective"],
    "eval_metric": df_sorted.iloc[0]["eval_metric"],
    "eta": df_sorted.iloc[0]["eta"],
    "max_depth": df_sorted.iloc[0]["max_depth"],
    "grow_policy": df_sorted.iloc[0]["grow_policy"],
    "max_leaves": df_sorted.iloc[0]["max_leaves"],
    "min_child_weight":  df_sorted.iloc[0]["min_child_weight"],
    "gamma":  df_sorted.iloc[0]["gamma"],
    "subsample":  df_sorted.iloc[0]["subsample"],
    "reg_lambda" : df_sorted.iloc[0]["reg_lambda"],
    "colsample_bytree": df_sorted.iloc[0]["colsample_bytree"],
    "reg_alpha" : df_sorted.iloc[0]["reg_alpha"]
}

bst2 = xgb.train(params, xtrain, 400, evallist, maximize=False, early_stopping_rounds=10)

bst2.save_model("models/model_immunopeptide_rankTIC_HCDy.xgboost")
