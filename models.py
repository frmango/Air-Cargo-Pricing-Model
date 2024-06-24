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
    """
    Train and evaluate models using k-fold cross-validation.
    
    Args:
    - df (DataFrame): DataFrame containing features and target variable 'Revenue'.

    Models:
    - Linear Regression
    - XGBoost with hyperparameter tuning
    - Sparse Variational Gaussian Process (SVGP)
    """
    X = df.drop(columns=['Revenue']).to_numpy().astype(np.float64)
    y = df['Revenue'].values
    k_folds = 5
    kf = KFold(n_splits=k_folds)

    mse_scores_lr = []
    mse_scores_xgb = []
    mae_scores_lr = []
    mae_scores_xgb = []
    mse_scores_sgp = []
    mae_scores_sgp = []

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

        # Sparse Variational Gaussian Process (SVGP)
        M = 100  # Number of inducing locations
        N = X_train.shape[0]
        kernel = gpflow.kernels.SquaredExponential()
        Z = kmeans2(X_train, M, minit='points')[0]
        m = gpflow.models.SVGP(kernel, gpflow.likelihoods.Gaussian(), Z, num_data=N)

        # Run minibatch Adam
        minibatch_size = 100
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train.reshape(-1, 1))).repeat().shuffle(N)
        maxiter = reduce_in_tests(20000)
        run_adam(m, maxiter, train_dataset, minibatch_size)

        # Predictions
        y_pred_sgp_mean, y_pred_sgp_var = m.predict_y(X_test)
        y_pred_sgp_mean = y_pred_sgp_mean.numpy()
        mse_sgp = mean_squared_error(y_test, y_pred_sgp_mean)
        mae_sgp = mean_absolute_error(y_test, y_pred_sgp_mean)
        mse_scores_sgp.append(mse_sgp)
        mae_scores_sgp.append(mae_sgp)

    print_metrics(mse_scores_lr, mae_scores_lr, mse_scores_xgb, mae_scores_xgb, mse_scores_sgp, mae_scores_sgp)

def run_adam(model, iterations, train_dataset, minibatch_size):
    """
    Utility function running the Adam optimizer.

    Args:
    - model (GPflow model): The GPflow model to be optimized.
    - iterations (int): Number of iterations for optimization.
    - train_dataset (tf.data.Dataset): The training dataset.
    - minibatch_size (int): The size of the minibatches.
    
    Returns:
    - logf (list): Log of the ELBO (Evidence Lower Bound) at each iteration.
    """
    logf = []
    train_iter = iter(train_dataset.batch(minibatch_size))
    training_loss = model.training_loss_closure(train_iter, compile=True)
    optimizer = tf.optimizers.Adam()

    @tf.function
    def optimization_step():
        optimizer.minimize(training_loss, model.trainable_variables)

    for step in range(iterations):
        optimization_step()
        if step % 10 == 0:
            elbo = -training_loss().numpy()
            logf.append(elbo)
    return logf

def print_metrics(mse_scores_lr, mae_scores_lr, mse_scores_xgb, mae_scores_xgb, mse_scores_sgp, mae_scores_sgp):
    """
    Print the mean evaluation metrics for each model.

    Args:
    - mse_scores_lr (list): List of MSE scores for Linear Regression.
    - mae_scores_lr (list): List of MAE scores for Linear Regression.
    - mse_scores_xgb (list): List of MSE scores for XGBoost.
    - mae_scores_xgb (list): List of MAE scores for XGBoost.
    - mse_scores_sgp (list): List of MSE scores for SVGP.
    - mae_scores_sgp (list): List of MAE scores for SVGP.
    """
    mean_mse_lr = np.mean(mse_scores_lr)
    mean_mae_lr = np.mean(mae_scores_lr)
    mean_mse_xgb = np.mean(mse_scores_xgb)
    mean_mae_xgb = np.mean(mae_scores_xgb)
    mean_mse_sgp = np.mean(mse_scores_sgp)
    mean_mae_sgp = np.mean(mae_scores_sgp)

    print(f'Mean Squared Error (Linear Regression): {mean_mse_lr}')
    print(f'Mean Absolute Error (Linear Regression): {mean_mae_lr}')
    print(f'Mean Squared Error (XGBoost): {mean_mse_xgb}')
    print(f'Mean Absolute Error (XGBoost): {mean_mae_xgb}')
    print(f'Mean Squared Error (SVGP): {mean_mse_sgp}')
    print(f'Mean Absolute Error (SVGP): {mean_mae_sgp}')

