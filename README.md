# Air Cargo Pricing Model

## Introduction
In the Air Cargo industry, effective pricing strategies are crucial for maximizing revenues. This project aims to develop a model that provides optimal pricing for shipping transactions between airlines and agents. The model considers various attributes of the shipment, such as weight, agent, shipment type, and day of the flight, to determine the best price for the service.  

## Rationale
The project's foundation lies in the need to adapt existing tools for new clients in the Air Cargo industry. The goal is to create a scalable code capable of handling diverse scenarios while maintaining interpretability. Design choices were made to ensure efficiency within an 8-hour time limit, focusing on statistical techniques, especially Bayesian methods, to showcase expertise.  

## Solution Walkthrough  
### Data Exploration and Preprocessing
* No missing values were found in the dataset.
* Feature selection was performed to enhance model explainability.
* Redundant and constant value columns were removed to reduce noise.
* Duplicate rows were eliminated to improve data quality and prevent biased model training.
* Skewness and Kurtosis were used to analyze dataset shape and identify outliers.  

### Data Transformation
* Transformation techniques such as logarithmic and square root transformations were applied to achieve a normal distribution of variables.
* Nominal attributes were encoded to facilitate machine learning model compatibility.
### Feature Standardization
* The covariance matrix was visualized to understand relationships between variables.
* All features were retained, considering high correlations within the covariance matrix.
* Feature standardization was performed, especially relevant for Gaussian Processes and Linear Regression.
### Model Selection and Tuning
Linear Regression [Baseline]
* Established a linear relationship between input features and the target variable.
* Serves as a benchmark for evaluating more complex models.
XGBoost
* Ensemble learning technique chosen for capturing non-linear relationships and complex interactions.
* Hyperparameters tuned, including learning rate, subsample, and max depth, to enhance predictive accuracy.
### Evaluation Metrics
* Mean Squared Error (MSE) and Mean Absolute Error (MAE) were chosen as regression metrics.
* MSE emphasizes larger errors and is sensitive to outliers.
* MAE treats all errors equally, providing a more intuitive average.
* Rationale Behind Using Both Metrics:
MSE and MAE offer complementary insights into different aspects of the model's performance.
### Gaussian Processes and Sparse Variational Gaussian Process (SVGP)
* Gaussian Processes explored for robustness but faced scalability issues.
* SVGP introduced inducing points to handle large-scale datasets efficiently.
### Conclusion
This Air Cargo Pricing Model employs a systematic approach to data preprocessing, transformation, and model selection, emphasizing interpretability and scalability. The inclusion of both linear and ensemble models, along with thoughtful evaluation metrics, ensures a comprehensive understanding of the model's performance. The integration of Gaussian Processes and SVGP showcases adaptability to diverse tasks while addressing scalability concerns.  
Regrettably, time constraints prevented the creation of additional visualizations for a direct visual comparison of all three models. However, thorough performance assessments have been conducted, and the results are presented in the output section.
