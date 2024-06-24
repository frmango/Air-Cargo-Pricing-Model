# Air Cargo Pricing Model

## Introduction

In the Air Cargo industry, developing effective pricing strategies is essential for maximizing revenue. This project aims to create a model that determines optimal pricing for shipping transactions between airlines and agents. By considering various attributes of the shipment—such as weight, agent, shipment type, and flight day—the model identifies the best service price.

## Rationale

The project is driven by the need to adapt existing tools for new clients within the Air Cargo industry. The objective is to develop scalable code that can handle diverse scenarios while remaining interpretable. Design choices prioritize efficiency within an 8-hour timeframe, focusing on statistical techniques, particularly Bayesian methods, to demonstrate expertise.

## Solution Walkthrough

### Data Exploration and Preprocessing

The dataset contained no missing values, ensuring a solid foundation for analysis. Features were carefully selected to enhance model interpretability. Redundant and constant value columns were removed to reduce noise, and duplicate rows were eliminated to improve data quality and prevent biased model training. Skewness and kurtosis were analyzed to understand the dataset shape and identify outliers.

### Data Transformation

Logarithmic and square root transformations were applied to achieve a normal distribution of variables, facilitating more accurate modeling. Nominal attributes were encoded to ensure compatibility with machine learning models.

### Feature Standardization

The covariance matrix was visualized to understand relationships between variables. Despite high correlations, all features were retained to preserve the dataset's integrity. Feature standardization was performed, particularly important for models such as Gaussian Processes and Linear Regression.

### Model Selection and Tuning

Linear Regression served as the baseline model, establishing a linear relationship between input features and the target variable. This provided a benchmark for evaluating more complex models. XGBoost was chosen for its ability to capture non-linear relationships and complex interactions. Hyperparameters, including learning rate, subsample, and max depth, were meticulously tuned to enhance predictive accuracy.

### Evaluation Metrics

Mean Squared Error (MSE) and Mean Absolute Error (MAE) were employed as regression metrics. MSE emphasizes larger errors and is sensitive to outliers, while MAE treats all errors equally, providing a more intuitive average. Using both metrics offered complementary insights into different aspects of the model's performance.

### Gaussian Processes and Sparse Variational Gaussian Process (SVGP)

Gaussian Processes were explored for their robustness, though they faced scalability issues. To address this, Sparse Variational Gaussian Processes (SVGP) were introduced, which use inducing points to handle large-scale datasets efficiently.

### Conclusion

The Air Cargo Pricing Model employs a systematic approach to data preprocessing, transformation, and model selection, emphasizing interpretability and scalability. By incorporating both linear and ensemble models and utilizing thoughtful evaluation metrics, the model ensures a comprehensive understanding of performance. The integration of Gaussian Processes and SVGP demonstrates adaptability to diverse tasks while addressing scalability concerns.

Although time constraints prevented the creation of additional visualizations for direct model comparisons, thorough performance assessments have been conducted, with results presented in the output section.
