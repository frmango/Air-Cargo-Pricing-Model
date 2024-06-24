from data_preprocessing import load_and_clean_data, encode_and_transform_data
from visualization import plot_univariate_distributions, plot_covariance_matrix
from models import train_models, evaluate_models, train_gp_model

# Load and clean the dataset
df = load_and_clean_data('dataset.csv')

# Encode and transform the dataset
df = encode_and_transform_data(df)

# Plot univariate distributions
plot_univariate_distributions(df)

# Plot covariance matrix
plot_covariance_matrix(df)

# Train and evaluate models
train_models(df)

# Train and evaluate Gaussian Process model
train_gp_model(df)

if __name__ == "__main__":
    main()