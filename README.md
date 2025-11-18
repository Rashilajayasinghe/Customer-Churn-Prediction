# Customer-Churn-Prediction
"A machine learning repository dedicated to building and evaluating a binary classification model to predict customer churn in the telecommunications industry.

# Exploratory Data Analysis (EDA) 🔎
The EDA phase involves exploring the dataset to gain insights into the underlying patterns, relationships, and data quality issues, which inform feature engineering and model selection.

The main steps in the EDA process include:

1. Data Loading and Understanding:
* Loading the dataset into the analysis environment (e.g., a Pandas DataFrame).
* Inspecting the data dimensions (rows/columns) and data types.
* Checking for the presence and extent of missing values and duplicate records.
  
2. Univariate Analysis:
* Analyzing each feature in isolation.
* Calculating summary statistics (mean, median, standard deviation, quartiles).
* Visualizing the distribution of numerical features (Histograms, Box Plots) and the frequency of categorical features         (Bar Charts).
  
3. Data Cleaning and Pre-processing:
* Handling the identified missing values (imputation or removal).
* Detecting and treating outliers (removal, capping, or transformation).
* Standardizing data formats and correcting inconsistencies.
4. Bivariate Analysis:
* Exploring the relationship between two variables, especially between independent features and the target variable.
* Using Scatter Plots (numeric vs. numeric), Box Plots (numeric vs. categorical), and Stacked Bar Charts or Cross-tabulations (categorical vs. categorical).

5. Multivariate Analysis:
* Exploring relationships among three or more variables simultaneously.
* Calculating and visualizing the correlation matrix (Heatmap) to identify highly correlated features (multicollinearity).
* Visualizing conditional distributions or using techniques like Pair Plots.

6. Feature Engineering:
* Deriving new, more informative features from existing data (e.g., ratios, aggregations).
* Applying necessary transformations (log, square root) to normalize skewed data.
* Encoding categorical features (e.g., One-Hot Encoding or Label Encoding).


Model Creation and Evaluation Summary

1. Data Preparation and Splitting
2. Initial Modeling and Imbalance Handling
Initial models, such as the Decision Tree Classifier, were trained using specified hyperparameters (e.g., criterion set to Entropy)56. Performance was assessed using the accuracy score and a detailed classification report

ANN Model Building and Training (Using Keras)

1.Data Scaling and Splitting
2.Building the ANN Model

