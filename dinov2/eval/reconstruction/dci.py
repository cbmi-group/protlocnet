import numpy as np
import scipy
from sklearn.model_selection import train_test_split


def compute_importance_matrix(representation, labels, test_ratio, random_state, model_cls, model_args):
  B, D = representation.shape
  train_idx, test_idx = train_test_split(np.arange(B), test_size=int(B * test_ratio), random_state=random_state)
  X_train = representation[train_idx].T
  Y_train = labels[train_idx].T
  X_test = representation[test_idx].T
  Y_test = labels[test_idx].T

  scores = {}
  importance_matrix, pred_trains, pred_tests = compute_importance_gbt(
      X_train, Y_train, X_test, Y_test, model_cls, model_args
  )
  scores["importance_matrix"] = importance_matrix
  scores["pred_trains"] = pred_trains
  scores["pred_tests"] = pred_tests
  return scores


def fit_one_factor(x_train, y_train, x_test, y_test, model_cls, model_args):
  model_args.update({
      'device': 'cuda',
  })
  model = model_cls(**model_args)
  model.fit(x_train.T, y_train)
  return (
      np.abs(model.feature_importances_),
      model.predict(x_train.T),
      model.predict(x_test.T),
  )
      

def compute_importance_gbt(x_train, y_train, x_test, y_test, model_cls, model_args):
  num_factors = y_train.shape[0]
  num_codes = x_train.shape[0]
  importance_matrix = np.zeros(shape=[num_codes, num_factors], dtype=np.float64)
  
  pred_trains = []
  pred_tests = []
  for i in range(num_factors):
    importance_matrix[:, i], pred_train, pred_test = fit_one_factor(
        x_train, y_train[i, :], x_test, y_test[i, :], model_cls[i], model_args[i]
    )
    pred_trains.append(pred_train)
    pred_tests.append(pred_test)
  return importance_matrix, pred_trains, pred_tests

def compute_disentanglement(importance_matrix):
  per_code = 1. - scipy.stats.entropy(importance_matrix.T + 1e-11, base=importance_matrix.shape[1])
  if importance_matrix.sum() == 0.:
    importance_matrix = np.ones_like(importance_matrix)
  code_importance = importance_matrix.sum(axis=1) / importance_matrix.sum()
  return np.sum(per_code*code_importance)
    

def compute_completeness(importance_matrix):
  per_factor = 1. - scipy.stats.entropy(importance_matrix + 1e-11, base=importance_matrix.shape[0])
  if importance_matrix.sum() == 0.:
    importance_matrix = np.ones_like(importance_matrix)
  factor_importance = importance_matrix.sum(axis=0) / importance_matrix.sum()
  return np.sum(per_factor*factor_importance)
