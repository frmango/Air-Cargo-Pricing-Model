import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
import tensorflow as tf
import gpflow
from gpflow.ci_utils import reduce_in_tests
from scipy.cluster.vq import kmeans2
from utils import normalize

def train_models(df):
    X = df.drop(columns=['Revenue']).to_numpy().astype(np.float64)
    y = df['Revenue'].values
    k_folds = 5
    kf = KFold(n_splits=k_folds)

    mse_scores_lr = []
    mse_scores_xgb = []
    mae_scores_lr = []
    mae_scores_xgb = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        X_train, X_test, y_train, y_test, _ = normalize(X_train, X_test, y_train, y_test)

        # Linear Regression
        model_lr = LinearRegression()
        model_lr.fit(X_train, y_train)
        y_pred_lr = model_lr.predict(X_test)
        mse_lr = mean_squared_error(y_test, y_pred_lr)
        mae_lr = mean_absolute_error(y_test, y_pred_lr)
        mse_scores_lr.append(mse_lr)
        mae_scores_lr.append(mae_lr)

        # XGBoost with hyperparameter tuning
        param_grid_xgb = {
            'learning_rate': [0.1, 0.01, 0.001],
            'subsample': [0.5, 0.7],
            'max_depth': [3, 6]
        }
        model_xgb = xgb.XGBRegressor()
        grid_search_xgb = GridSearchCV(model_xgb, param_grid_xgb, cv=3, scoring='neg_mean_squared_error')
        grid_search_xgb.fit(X_train, y_train)
        best_model_xgb = grid_search_xgb.best_estimator_
        y_pred_xgb = best_model_xgb.predict(X_test)
        mse_xgb = mean_squared_error(y_test, y_pred_xgb)
        mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
        mse_scores_xgb.append(mse_xgb)
        mae_scores_xgb.append(mae_xgb)
        print(f'XGBoost Parameters: {grid_search_xgb.best_params_}')
        print(f'XGBoost Mean Squared Error: {- grid_search_xgb.best_score_}')

    print_metrics(mse_scores_lr, mae_scores_lr, mse_scores_xgb, mae_scores_xgb)

def print_metrics(mse_scores_lr, mae_scores_lr, mse_scores_xgb, mae_scores_xgb):
    mean_mse_lr = np.mean(mse_scores_lr)
    mean_mae_lr = np.mean(mae_scores_lr)
    mean_mse_xgb = np.mean(mse_scores_xgb)
    mean_mae_xgb = np.mean(mae
