import pandas as pd

def load_and_clean_data(filepath):
    ds = pd.read_csv(filepath)
    df = ds.copy()

    # Basic dataset information
    print(f"The dataset has {df.shape[0]} instances and {df.shape[1]} features.")
    print(df.head(10))

    # Check for missing values
    print('Percentage of data without missing values: ', df.dropna().shape[0] / float(df.shape[0]))

    # Drop unuseful features
    df = df.drop(['POS', 'AgentName'], axis=1)

    # Drop constant columns
    constant_columns = [col for col in df.columns if df[col].nunique() == 1]
    df.drop(columns=constant_columns, inplace=True)

    # Remove duplicate rows
    df = df.drop_duplicates()
    print(f"The dataset now has {df.shape[0]} row instances.")

    return df

def encode_and_transform_data(df):
    import numpy as np
    from scipy.stats import skew

    # Encode categorical features
    categorical_features = df.select_dtypes(include=["object"]).columns.tolist()
    for col in categorical_features:
        if col != 'FlownMonth':
            df[col] = pd.factorize(df[col])[0]

    df = pd.get_dummies(df, columns=['FlownMonth'])

    # Transform skewed features
    numeric_data = df.select_dtypes(include=['float64', 'int64']).columns
    for feature in numeric_data:
        skewness = skew(df[feature])
        df[feature] = transform_feature(df, feature, skewness)
    
    return df

def transform_feature(df, feature, skewness):
    import numpy as np

    transformed_feature = df[feature].copy()
    if skewness < 0:
        k = df[feature].min() + 1
        transformed_feature = k - transformed_feature
    if abs(skewness) > 1:
        transformed_feature = np.log1p(transformed_feature)
    elif abs(skewness) > 0.5 and abs(skewness) < 1:
        transformed_feature = np.sqrt(transformed_feature)
    return transformed_feature
