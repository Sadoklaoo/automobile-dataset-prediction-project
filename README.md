# Automobile Dataset Prediction: Model Evaluation and Hyperparameter Tuning
## Overview
This project applies machine learning techniques to predict automobile-related outcomes using various models. It involves the following steps:

1. Exploratory Data Analysis (EDA)
2. Preprocessing the Data
3. Model Training
4. Model Evaluation using MAE and RMSE
5. Hyperparameter Tuning for improving model performance
6. Model Comparison and Results

## Project Requirements

- Python 3.x or higher
- Libraries:
  -  pandas
  -  numpy
  -  matplotlib
  -  scikit-learn

## Setup Instructions

1. Clone the repository
```bash
git clone https://github.com/Sadoklaoo/itds-assessement
cd
```
2. Install dependencies
```bash
pip install pandas numpy matplotlib scikit-learn
```
3. Download the dataset
Download the dataset: Make sure the dataset automobile.csv is in the working directory. You can download the dataset from [automobile.csv](https://archive.ics.uci.edu/static/public/10/data.csv) and save it as csv file in the same repository.

5. Run the Jupyter Notebook
- Open the notebook in Jupyter or Jupyter Lab:
```bash
jupyter notebook
```
- Run all the cells in the notebook to execute the code.
  

## Setup Instructions
1. Exploratory Data Analysis (EDA):
- The dataset is loaded and cleaned.
- Basic visualizations and statistical analysis are performed to understand the dataset.

2. Data Preprocessing:
- Missing values are handled.
- Categorical variables are encoded.
- The dataset is split into training and test sets.

3. Model Training:
- Multiple models are trained, including:
  - Linear Regression
  - Decision Tree
  - Random Forest
  - KNN (K-Nearest Neighbors)

4. Model Evaluation:
- Each model is evaluated using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) to compare their performance.

5. Hyperparameter Tuning:
- Random Forest and KNN models are tuned using GridSearchCV and RandomizedSearchCV to find the optimal hyperparameters.

6. Results:
- A bar plot is generated to visualize the MAE and RMSE of each model.
- The model performance is summarized, and the best model is selected based on evaluation metrics.

## Example Usage

1. Load and preprocess the dataset:
```python
# Load the dataset
data = pd.read_csv('automobile.csv')

# Perform EDA, preprocessing, and model training as instructed in the notebook
```
2. Train models and evaluate:
```python
# Train models
models = [linear_model, decision_tree, random_forest, knn]

# Evaluate models
for model in models:
    y_pred = model.predict(X_test)
    print(f"{model.__class__.__name__} Evaluation:")
    print(f"MAE: {mean_absolute_error(y_test, y_pred)}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred))}")
```
3. Hyperparameter tuning for KNN (example):
```python
from sklearn.model_selection import GridSearchCV

param_grid = {'n_neighbors': [3, 5, 7, 9]}
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
```

## Output
- Tables showing MAE and RMSE values for each model.
- Bar plot comparing the MAE and RMSE of all models.

## Additional Information
- If you encounter issues with missing values or data discrepancies, ensure that your dataset is correctly formatted.
- You can modify the model hyperparameters or try additional models to improve the results further.




