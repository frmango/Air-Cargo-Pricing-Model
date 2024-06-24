import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis

def plot_univariate_distributions(df):
    numeric_data = df.select_dtypes(include=['float64', 'int64']).columns
    fig, axes = plt.subplots(nrows=len(numeric_data), ncols=1, figsize=(5, 2 * len(numeric_data)))
    fig.suptitle('Univariate Distributions', y=1.02)

    for i, feature in enumerate(numeric_data):
        plot_univariate_distribution(df, axes[i], feature)

    plt.tight_layout()
    plt.show()

def plot_univariate_distribution(df, axes, feature):
    if df[feature].dtype == 'float64':
        sns.histplot(df[feature], kde=True, bins=100, ax=axes)
        axes.set_title('PDF')
    elif df[feature].dtype == 'int64':
        sns.histplot(df[feature], bins=100, discrete=False, ax=axes)
        axes.set_title('PMF')
    skewness = skew(df[feature])
    kurt = kurtosis(df[feature])
    axes.text(0.95, 0.95, f'Skewness: {skewness:.2f}\nKurtosis: {kurt:.2f}', 
              transform=axes.transAxes, ha='right', va='top', 
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

def plot_covariance_matrix(df):
    numeric_data = df.select_dtypes(include=['float64', 'int64']).columns
    cov_matrix = df[numeric_data].cov()
    sns.heatmap(cov_matrix, annot=False, cmap='vlag', fmt='.2f', linewidths=0.5, vmin=-1, vmax=1)
    plt.title('Covariance Matrix Heatmap')
    plt.show()
