import pandas as pd
import numpy as np
import torch
import datetime
import os
import pickle
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from models_rd import Raindrop_Mod

from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score, accuracy_score, brier_score_loss, precision_recall_curve
from sklearn.model_selection import ParameterGrid
from sklearn.calibration import CalibratedClassifierCV
from sklearn.calibration import calibration_curve, CalibrationDisplay
from expl_perf_drop.explainers import CGExplainerDR
from expl_perf_drop.utils import Graph

from expl_perf_drop.models import select as model_select
from tqdm import tqdm

from sklearn.metrics._ranking import _binary_clf_curve
import numpy as np
from sklearn.metrics import confusion_matrix

def precision_recall_curve(y_true, y_pred, pos_label=None,
                           sample_weight=None,pi0=None):
    """Compute precision-recall (with optional calibration) pairs for different probability thresholds
    This implementation is a modification of scikit-learn "precision_recall_curve" function that adds calibration
    ----------
    y_true : array, shape = [n_samples]
        True binary labels. If labels are not either {-1, 1} or {0, 1}, then
        pos_label should be explicitly given.
    probas_pred : array, shape = [n_samples]
        Estimated probabilities or decision function.
    pos_label : int or str, default=None
        The label of the positive class.
        When ``pos_label=None``, if y_true is in {-1, 1} or {0, 1},
        ``pos_label`` is set to 1, otherwise an error will be raised.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.
    Returns
    -------
    calib_precision : array, shape = [n_thresholds + 1]
        Calibrated Precision values such that element i is the calibrated precision of
        predictions with score >= thresholds[i] and the last element is 1.
    recall : array, shape = [n_thresholds + 1]
        Decreasing recall values such that element i is the recall of
        predictions with score >= thresholds[i] and the last element is 0.
    thresholds : array, shape = [n_thresholds <= len(np.unique(probas_pred))]
        Increasing thresholds on the decision function used to compute
        precision and recall.
    """
    
    fps, tps, thresholds = _binary_clf_curve(y_true, y_pred,
                                             pos_label=pos_label,
                                             sample_weight=sample_weight)
    
   
    
    
    if pi0 is not None:
        pi = np.sum(y_true)/float(np.array(y_true).shape[0])
        ratio = pi*(1-pi0)/(pi0*(1-pi))
        precision = tps / (tps + ratio*fps)
    else:
        precision = tps / (tps + fps)
    
    precision[np.isnan(precision)] = 0
        
    recall = tps / tps[-1]

    # stop when full recall attained
    # and reverse the outputs so recall is decreasing
    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)
    return np.r_[precision[sl], 1], np.r_[recall[sl], 0], thresholds[sl]
def average_precision(y_true, y_pred, pos_label=1, sample_weight=None,pi0=0.1):
        precision, recall, _ = precision_recall_curve(y_true, y_pred, pos_label=pos_label, sample_weight=sample_weight, pi0=pi0)
        return -np.sum(np.diff(recall) * np.array(precision)[:-1])
# # Main flow is this:
# #   1) Load data (embeddings acquired from Raindrop) for both datasets, based on the model used to infer embeddings
# #   2) Define model (mlp_static from Raindrop)
# #   3) Get prediction metrics and visualize AUPRC and calibration
# #   4) Explain model (with expl_perf_drop)
def auprc_torch(model, data: pd.DataFrame, subset_cols=None, weight=None, target_name='outcome'):
    # (1) 사용할 feature/target 분리
    if subset_cols is not None:
        df_feats = data[subset_cols]
    else:
        # default: 모든 열 중 outcome 열만 제외
        df_feats = data.drop(columns=[target_name])
    y_true = data[target_name].values

    # (2) Pandas → NumPy → torch.Tensor
    #     (float 타입으로 변환, 기기의 일치)
    X_np = df_feats.to_numpy(dtype=np.float32)
    X_tensor = torch.from_numpy(X_np)

    # 모델이 이미 특정 디바이스에 올라가 있을 수 있으므로, 일치시켜줍니다.
    device = next(model.parameters()).device
    X_tensor = X_tensor.to(device)

    # (3) 모델 예측 (no_grad 상태, eval 모드)
    model.eval()
    with torch.no_grad():
        output = model(X_tensor)
        probs = torch.sigmoid(output).cpu().numpy().reshape(-1)

    # (4) AUPRC 계산 (scikit-learn)
    return average_precision(y_true, probs, sample_weight=weight)


def auprc_c_data(model, data: pd.DataFrame, subset_cols=None, weight=None, target_name='outcome'):
    return average_precision(data['label'], data['proba'], sample_weight=weight)

def auprc_data(model, data: pd.DataFrame, subset_cols=None, weight=None, target_name='outcome'):
    return average_precision_score(data['label'], data['proba'], sample_weight=weight)

def brier_data(model, data: pd.DataFrame, subset_cols=None, weight=None, target_name='outcome'):
    return brier_score_loss(data['label'], data['proba'], sample_weight=weight)

def brier_torch(model, data: pd.DataFrame, subset_cols=None, weight=None, target_name='outcome'):

    # (1) feature/target 분리
    if subset_cols is not None:
        df_feats = data[subset_cols]
    else:
        df_feats = data.drop(columns=[target_name])
    y_true = data[target_name].values

    # (2) Pandas → NumPy → torch.Tensor
    X_np = df_feats.to_numpy(dtype=np.float32)
    X_tensor = torch.from_numpy(X_np)
    device = next(model.parameters()).device
    X_tensor = X_tensor.to(device)

    # (3) 모델 예측 (no_grad, eval)
    model.eval()
    with torch.no_grad():
        output = model(X_tensor)
        probs = torch.sigmoid(output).cpu().numpy().reshape(-1)

    # (4) Brier Score 계산 (scikit-learn)
    return brier_score_loss(y_true, probs, sample_weight=weight)

def explain_model(exp, model, auprc = auprc_torch, auprc_c = auprc_c_data, brier = brier_torch):
    print("==============")
    print("AUPRC")
    print("==============")
    exp.get_all_possible_shifts()

    perfs_auprc = exp.get_perf_on_sets(model, metric = auprc)
    print(perfs_auprc)

    res_auprc = exp.explain(model, metric = auprc)
    print(res_auprc)

    print(exp.scale(res_auprc, perfs_auprc))

    print("==============")
    print("AUPRC_c")
    print("==============")
    exp.get_all_possible_shifts()

    perfs_auprc_c = exp.get_perf_on_sets(model, metric = auprc_c)
    print(perfs_auprc_c)

    res_auprc_c = exp.explain(model, metric = auprc_c)
    print(res_auprc_c)

    print(exp.scale(res_auprc_c, perfs_auprc_c))

    print("==============")
    print("Brier Score")
    print("==============")
    perfs = exp.get_perf_on_sets(model, metric = brier)
    print(perfs)

    res_brier = exp.explain(model, metric = brier)
    print(res_brier)

    print(exp.scale(res_brier, perfs))

    return res_auprc, exp.scale(res_auprc, perfs_auprc), res_auprc_c, exp.scale(res_auprc_c, perfs_auprc_c), res_brier, exp.scale(res_brier, perfs)




# Add part to load best model and use "mlp_static" (the forecast head) on the embeddings for prediction
#    Load them as : model

def explain_attributions(source, task, d_ob=4):
    target = 'mimic' if source =='eicu' else 'eicu'

    model = 0
    # Add part to load data(embeddings acquired from raindrop models), and outcomes (from arr_outcomes_{data}.npy)
    #    Load them as : train_vars_source_df, test_vars_source_df, train_vars_target_df, test_vars_target_df
    train_vars_source_df = pd.read_feather(f'./rd_results/predictions_train_{source}_{task}_source.feather')
    source_valid_df = pd.read_feather(f'./rd_results/predictions_val_{source}_{task}_source.feather')

    train_vars_source_df = pd.concat([train_vars_source_df, source_valid_df])
    test_vars_source_df = pd.read_feather(f'./rd_results/predictions_test_{source}_{task}_source.feather')

    train_vars_target_df = pd.read_feather(f'./rd_results/predictions_train_{target}_{task}_target.feather')
    target_valid_df = pd.read_feather(f'./rd_results/predictions_val_{target}_{task}_target.feather')
    train_vars_target_df = pd.concat([train_vars_target_df, target_valid_df])

    test_vars_target_df = pd.read_feather(f'./rd_results/predictions_test_{target}_{task}_target.feather')


    # Fix var_categories accordingly
    var_categories = { 
        'static' : [f'emb_{i}' for i in range(d_ob*35+16, (d_ob+1)*35+16)], # gotta fix this (Probably 35 vars from d_ob * 35)
        'vital': [f'emb_{i}' for i in range(0, d_ob*6)], # 6 vital vars
        'lab': [f'emb_{i}' for i in range(d_ob*6, d_ob*19)], # 13 lab vars
        'treat': [f'emb_{i}' for i in range(d_ob*19, d_ob*35)], # 16 treat vars
        'outcome': ['label']
    }
    
    train_features = var_categories['vital'] + var_categories['lab'] + var_categories['treat'] + var_categories['static']

    # define the causal graph
    GRAPH = Graph(
        nodes= list(var_categories.keys()),
        edges=[
            ('static', 'vital'),
            ('static', 'lab'),
            ('vital', 'treat'),
            ('lab', 'treat'),
            ('static', 'treat'),
            ('static', 'outcome'),
            ('vital', 'outcome'), 
            ('lab', 'outcome'), 
            ('treat', 'outcome'), 
        ]
    )  

    exp = CGExplainerDR(GRAPH, train_vars_source_df, test_vars_source_df, train_vars_target_df, test_vars_target_df,
            train_features, var_categories, target_name = 'label')

    result_auprc, result_brier, result_brier_scale = explain_model(exp, model, auprc_data, brier_data)
    df = pd.concat([result_auprc, result_brier, result_brier_scale], axis=1)
    df.columns = ['AUPRC', 'Brier', 'Brier (Scaled)']
    df.loc["Total"] = df.sum()

    auprc_source = auprc_data(model, test_vars_source_df)
    brier_source = brier_data(model, test_vars_source_df)
    auprc_target = auprc_data(model, test_vars_target_df)
    brier_target = brier_data(model, test_vars_target_df)

    df.loc["Source"] = [auprc_source, brier_source, brier_source]
    df.loc["Target"] = [auprc_target, brier_target, brier_target]
    df.to_csv(f'./rd_results/attribution_{source}_{task}.csv')

    return df

def explain_attributions_common(model_name, pi0):
    """
    model name like 'stratsvar_pre_False_dann_False_mor'
    """
    
    model = 0
    # Add part to load data(embeddings acquired from raindrop models), and outcomes (from arr_outcomes_{data}.npy)
    #    Load them as : train_vars_source_df, test_vars_source_df, train_vars_target_df, test_vars_target_df
    train_vars_source_df = pd.read_feather(f'./rd_results/predictions_source_train_{model_name}.feather')
    source_valid_df = pd.read_feather(f'./rd_results/predictions_source_val_{model_name}.feather')

    train_vars_source_df = pd.concat([train_vars_source_df, source_valid_df])
    test_vars_source_df = pd.read_feather(f'./rd_results/predictions_source_test_{model_name}.feather')

    train_vars_target_df = pd.read_feather(f'./rd_results/predictions_target_train_{model_name}.feather')
    target_valid_df = pd.read_feather(f'./rd_results/predictions_target_val_{model_name}.feather')
    train_vars_target_df = pd.concat([train_vars_target_df, target_valid_df])

    test_vars_target_df = pd.read_feather(f'./rd_results/predictions_target_test_{model_name}.feather')


    # Fix var_categories accordingly
    var_categories = { 
        'static' : [f'emb_{i}' for i in range(96, 128)], # gotta fix this (Probably 35 vars from d_ob * 35)
        'vital': [f'emb_{i}' for i in range(0, 32)], # 6 vital vars
        'lab': [f'emb_{i}' for i in range(32, 64)], # 13 lab vars
        'treat': [f'emb_{i}' for i in range(64, 96)], # 16 treat vars
        'outcome': ['label']
    }
    
    train_features = var_categories['vital'] + var_categories['lab'] + var_categories['treat'] + var_categories['static']

    # define the causal graph
    GRAPH = Graph(
        nodes= list(var_categories.keys()),
        edges=[
            ('static', 'vital'),
            ('static', 'lab'),
            ('vital', 'treat'),
            ('lab', 'treat'),
            ('static', 'treat'),
            ('static', 'outcome'),
            ('vital', 'outcome'), 
            ('lab', 'outcome'), 
            ('treat', 'outcome'), 
        ]
    )  

    # exp = CGExplainerDR(GRAPH, train_vars_source_df, test_vars_source_df, train_vars_target_df, test_vars_target_df,
    #         train_features, var_categories, target_name = 'label')
    exp = CGExplainerDR(GRAPH, train_vars_source_df, test_vars_source_df, train_vars_target_df, test_vars_target_df,
            train_features, var_categories, target_name = 'label', clip_prob_thres=0.9)

    result_auprc, result_auprc_scale, result_auprc_c, result_auprc_c_scale, result_brier, result_brier_scale = explain_model(exp, model, auprc_data,
                                                                                                                              auprc_c_data, brier_data)

    # after explainer._train_weight_models() has run
    wr = exp.weight_models[frozenset(['static'])]
    # apply to your source_eval_df
    X = exp.source_eval_df[var_categories['vital']].values  # plus static if needed
    w_vital = wr(X)
    print("vital ratio: mean", w_vital.mean(), "std", w_vital.std(), 
      "min", w_vital.min(), "max", w_vital.max(), "median", np.median(w_vital))
    q99 = np.percentile(w_vital, 99)
    print(q99)
    clipped = np.clip(w_vital, None, q99)
    print("vital ratio: mean", clipped.mean(), "std", clipped.std(), 
      "min", clipped.min(), "max", clipped.max(), "median", np.median(clipped))
    
    # exp = CGExplainerDR(GRAPH, train_vars_source_df, test_vars_source_df, train_vars_target_df, test_vars_target_df,
    #         train_features, var_categories, target_name = 'label', calibrate_weight_models=True)

    # result_auprc, result_brier, result_brier_scale = explain_model(exp, model, auprc_data, brier_data)


    df = pd.concat([result_auprc, result_auprc_scale, result_auprc_c, result_auprc_c_scale, result_brier, result_brier_scale], axis=1)
    df.columns = ['AUPRC', 'AUPRC (Scaled)', 'AUPRC_c', 'AUPRC_c (Scaled)', 'Brier', 'Brier (Scaled)']
    df.loc["Total"] = df.sum()

    auprc_source = auprc_data(model, test_vars_source_df)
    auprc_c_source = auprc_c_data(model, test_vars_source_df)
    brier_source = brier_data(model, test_vars_source_df)
    auprc_target = auprc_data(model, test_vars_target_df)
    auprc_c_target = auprc_c_data(model, test_vars_target_df)
    brier_target = brier_data(model, test_vars_target_df)

    df.loc["Source"] = [auprc_source, auprc_source, auprc_c_source, auprc_c_source, brier_source, brier_source]
    df.loc["Target"] = [auprc_target, auprc_target, auprc_c_target, auprc_c_target, brier_target, brier_target]
    df.to_csv(f'./rd_results/attribution_{model_name}_clip.csv')

    return df

# pi0
# aki : 0.20
# cf : 0.04
# mor : 0.1
# los : 0.25


if __name__=="__main__":
    df = explain_attributions_common('surpstratsvar_pre_False_dann_False_weight_False_32_mor', 0.04)