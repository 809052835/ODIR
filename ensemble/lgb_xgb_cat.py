import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from catboost import CatBoostClassifier


# 二分类任务的lgb
def lgb_binary(X_train, X_val, y_train, y_val, columns, is_unbalance):
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'num_leaves': 63,
        'learning_rate': 0.01,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.9,
        'bagging_seed': 0,
        'bagging_freq': 1,
        'verbose': 5,
        'reg_alpha': 2,
        'reg_lambda': 2,
        'is_unbalance': is_unbalance
    }

    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_evals = lgb.Dataset(X_val, y_val, reference=lgb_train)
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=10000,
                    valid_sets=[lgb_train, lgb_evals],
                    valid_names=['train', 'valid'],
                    early_stopping_rounds=200,
                    verbose_eval=100,
                    )

    pred_y_train = gbm.predict(X_train, num_iteration=gbm.best_iteration)
    pred_y_val = gbm.predict(X_val, num_iteration=gbm.best_iteration)
    # print('训练集的AUC：', roc_auc_score(y_train, pred_y_train), '验证集的AUC：', roc_auc_score(y_val, pred_y_val))

    '''all_data'''
    lgb_train = lgb.Dataset(X_train.append(X_val), y_train.append(y_val))
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=gbm.best_iteration,
                    valid_sets=[lgb_train],
                    valid_names=['train'],
                    verbose_eval=100
                    )

    print(pd.DataFrame({
        'column': columns,
        'importance': gbm.feature_importance(),
    }).sort_values(by='importance'))
    return gbm
