import numpy as np
import pandas as pd

def calculate_training_time(log_path):

    with open(log_path, 'r') as f:
        lines = f.readlines()

    times = list()
    for line in lines:
        times.append(float(line.split("\t")[1].split(" ")[2]))

    time_mean = sum(times[1:])/len(times[1:])
    time_compilation = times[0]

    return time_mean, time_compilation

def first_epoch(log_path):
    with open(log_path, 'r') as f:
        lines = f.readlines()

    epoch_0 = int(lines[0].split("epoch: ")[1].split("\t")[0].strip(" "))

    return epoch_0


#--------------------          Evaluation metrics          --------------------#

def rse(y, y_pred):
    return np.mean((y-y_pred)**2) / np.mean((y-y.mean())**2)

def rmse(y, y_pred):
    return np.sqrt(np.mean((y-y_pred)**2))

def mae(y, y_pred):
    return np.mean(np.abs(y-y_pred))

def compute_metrics(g_u, g_u_pred, round=None, std=False):
    metrics = dict()
    metrics['mae'] = mae(g_u, g_u_pred)
    metrics['rmse'] = rmse(g_u, g_u_pred)
    metrics['rse'] = rse(g_u, g_u_pred)

    if round is not None:
        for key in metrics:
            metrics[key] = np.round(metrics[key], round)

    return metrics


def compute_and_print_metrics(true_train=None, pred_train=None, true_test=None, pred_test=None, round=None):

    compute_train = (true_train is not None) and (pred_train is not None) and (pred_train.shape[0] != 0)
    compute_test = (true_test is not None) and (pred_test is not None)

    if compute_train:
        metrics_train = compute_metrics(true_train, pred_train, round=round)
        metrics_train = pd.DataFrame(metrics_train, index=["train: "])

    if compute_test:
        metrics_test = compute_metrics(true_test, pred_test, round=round)
        metrics_test = pd.DataFrame(metrics_test, index=["test: "])

    if compute_train and compute_test:
        metrics = pd.concat([metrics_train, metrics_test])

    elif compute_train:
        metrics = metrics_train

    elif compute_test:
        metrics = metrics_test

    else:
        raise ValueError(
            "Must provide true and pred for at least training or testing data.")

    return metrics


def rse_in_time(g_u, g_u_pred):    
    """
    Computes relative squared error in time.
    g_u and g_u pred have to have shape [bs, t_len, x_len]
    """         
    rse_ts = np.mean(np.mean(((g_u-g_u_pred))**2, axis=2), axis=0) / np.mean(np.mean(((g_u-g_u.mean())**2), axis=2), axis=0)
    return rse_ts

def rse_by_sample(g_u, g_u_pred):    
    """
    Computes relative squared error by sample.
    g_u and g_u pred have to have shape [bs, t_len, x_len]
    """         
    rse = np.mean(np.mean(((g_u-g_u_pred))**2, axis=2), axis=1) / np.mean(np.mean(((g_u-g_u.mean())**2), axis=2), axis=1)
    return rse

