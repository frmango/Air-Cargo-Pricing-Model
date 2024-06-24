import numpy as np

def normalize(X_train, X_test, y_train, y_test):
    """
    Normalize the features and target variables for training and testing datasets.

    Args:
    X_train (ndarray): Training features.
    X_test (ndarray): Testing features.
    y_train (ndarray): Training targets.
    y_test (ndarray): Testing targets.

    Returns:
    tuple: Normalized training features, testing features, training targets, 
           testing targets, and variance of the training targets.
    """
    X_train_mean, X_train_var = X_train.mean(axis=0), X_train.std(axis=0) + 1e-9
    y_train_mean, y_train_var = y_train.mean(axis=0), y_train.std(axis=0) + 1e-9

    X_train = (X_train - X_train_mean) / X_train_var
    X_test = (X_test - X_train_mean) / X_train_var
    y_train = (y_train - y_train_mean) / y_train_var
    y_test = (y_test - y_train_mean) / y_train_var

    return X_train, X_test, y_train, y_test, y_train_var

def print_metrics(mse_scores_lr, mae_scores_lr, mse_scores_xgb, mae_scores_xgb):
    """
    Print the mean squared error (MSE) and mean absolute error (MAE) for Linear Regression and XGBoost models.

    Args:
    mse_scores_lr (list): MSE scores for Linear Regression.
    mae_scores_lr (list): MAE scores for Linear Regression.
    mse_scores_xgb (list): MSE scores for XGBoost.
    mae_scores_xgb (list): MAE scores for XGBoost.
    """
    mean_mse_lr = np.mean(mse_scores_lr)
    mean_mae_lr = np.mean(mae_scores_lr)
    mean_mse_xgb = np.mean(mse_scores_xgb)
    mean_mae_xgb = np.mean(mae_scores_xgb)

    print(f'Mean Squared Error (Linear Regression): {mean_mse_lr}')
    print(f'Mean Absolute Error (Linear Regression): {mean_mae_lr}')
    print(f'Mean Squared Error (XGBoost): {mean_mse_xgb}')
    print(f'Mean Absolute Error (XGBoost): {mean_mae_xgb}')

def transform_feature(df, feature, skewness):
    """
    Transform a feature based on its skewness to reduce skew.

    Args:
    df (DataFrame): DataFrame containing the feature.
    feature (str): Name of the feature to be transformed.
    skewness (float): Skewness of the feature.

    Returns:
    Series: Transformed feature.
    """
    transformed_feature = df[feature].copy()
    if skewness < 0:
        k = df[feature].min() + 1
        transformed_feature = k - transformed_feature
    if abs(skewness) > 1:
        transformed_feature = np.log1p(transformed_feature)
    elif abs(skewness) > 0.5 and abs(skewness) < 1:
        transformed_feature = np.sqrt(transformed_feature)
    return transformed_feature
