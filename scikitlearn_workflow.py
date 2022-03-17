import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.model_selection
import sklearn.metrics

import sklearn.preprocessing
import sklearn.cluster
import sklearn.decomposition
import sklearn.linear_model

import sklearn.pipeline
import joblib
import statsmodels.api as sm
import os.path
from sklearn.base import BaseEstimator, TransformerMixin

#import sys
#if not '/home/zacharygibbs/documents/' in sys.path: 
#    sys.path.append('/home/zacharygibbs/documents/')
#print(sys.path)
#import sklearn_sciencey.core as sks

def cross_val_fun(name, estimator, X, y, params={}, cv=10, verbose=True, scoring=None, do_rfecv=True):
    mod = {
        'name': name,#'lda',
        'estimator': estimator,#sklearn.discriminant_analysis.LinearDiscriminantAnalysis(),
        'scores':None,
        'score_mean':None,
        'score_std':None,
        'params': params,#{}
        'cv':cv
          }
    mod['estimator'].set_params(**mod['params'])
    mod['estimator'].fit(X, y)
    try:
        mod['confusion_matrix'] = sklearn.metrics.confusion_matrix(y, mod['estimator'].predict(X))
    except:
        pass
    mod['features'] = X.columns
    mod_name = mod['name']
    mod['scores'] = sklearn.model_selection.cross_val_score(mod['estimator'], X, y,
                             scoring=scoring, cv=cv)#10)
    try:
        if do_rfecv:
            mod['features_rfecv'] = rfecv(mod['estimator'], X, y, cv=cv, scoring=scoring)
        else: 
            1/0
    except:
        mod['features_rfecv'] = []
        print(f'{mod_name}: features_rfecv failed')
    mod['score_mean'] = mod['scores'].mean()
    mod['score_std'] = mod['scores'].std()
    if verbose:
        print(f'*********** {mod_name} *************')
        print(mod['scores'])
        print(mod['score_mean'])
        print(mod['score_std'])
    return mod

def rfecv(model, X, y, cv, scoring):
    model.fit(X, y)
    cv_selector = sklearn.feature_selection.RFECV(model,cv=cv, step=1,scoring=scoring)
    cv_selector = cv_selector.fit(X, y)
    rfecv_mask = cv_selector.get_support() #list of booleans
    rfecv_features = [] 
    for bool, feature in zip(rfecv_mask, X.columns):
        if bool:
            rfecv_features.append(feature)
    return rfecv_features


def get_model_selec_from_name(model_selec_hyperparam, name, return_model_index_only=True):
    model_selec_hyperparam_index = [index for index, item in enumerate(model_selec_hyperparam) if item['name']==name][0]
    model = model_selec_hyperparam[model_selec_hyperparam_index]['estimator'] #neural network
    if return_model_index_only:
        return model, model_selec_hyperparam_index
    else:
        return model_selec_hyperparam[model_selec_hyperparam_index]

def get_clf_score_model(model_selec_hyperparam, model_selec_hyperparam_name, model_selec_hyperparam_index, clf, model_selec_scores):
    model_selec_hyperparam[model_selec_hyperparam_index]['params'] = clf.best_params_
    model_selec_hyperparam[model_selec_hyperparam_index]['estimator'] = clf.best_estimator_
    model_selec_hyperparam[model_selec_hyperparam_index]['scores'] = -1*clf.cv_results_['mean_test_score']
    model_selec_hyperparam[model_selec_hyperparam_index]['score_best'] = -1*clf.best_score_
    print(clf.best_estimator_)
    print(model_selec_hyperparam[model_selec_hyperparam_index]['score_best'])
    model_selec_scores.loc[model_selec_hyperparam_name, 'score_hyperparameter'] = model_selec_hyperparam[model_selec_hyperparam_index]['score_best']
    try:
        print(model_selec_scores.drop(['features_rfecv', 'rfecv_len(sel / total)'], axis=1))
    except:
        print(model_selec_scores)
    return model_selec_hyperparam, model_selec_scores

