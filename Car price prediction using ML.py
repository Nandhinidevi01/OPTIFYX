# basic libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sea
born as sns
import plotly.express as px

# import from scikit-learn library
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
import pickle

# ignore warnings
import warnings
warnings.filterwarnings('ignore')

print('Libraries imported successfully!')

# Load the data
df = pd.read_csv('car data.csv')

print('Data loaded successfully')

# display the first 5 rows of the data
df.head()

# shape of data
print(f'Rows = {df.shape[0]}\n')
print(f'Columns = {df.shape[1]}')

# info of data
df.info()

# statistical summary of data
num_sum = df.describe()
palette = sns.color_palette('viridis', as_cmap=True)
num_sum.style.background_gradient(cmap=palette)

# missing values
missing = df.isnull().sum()
print(missing)
​
print('\n There is no missing values in the dataset')

# duplicate values
duplicates = df.duplicated().sum()
print(f"Number of duplicate rows = {duplicates}")

# drop duplicates
print("After dropping duplicates")
df.drop_duplicates(inplace=True)
print(f"Number of duplicate rows = {df.duplicated().sum()}")

# Set up the figure with 1 row and 3 columns
fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

# Create individual box plots using Seaborn
sns.boxplot(ax=axes[0], y=df['Driven_kms'], color='darkblue', showfliers=True)
axes[0].set_title('Driven_kms', fontsize=14)
axes[0].set_ylabel("")  # Remove y-axis label to match uniformity

sns.boxplot(ax=axes[1], y=df['Present_Price'], color='darkgreen', showfliers=True)
axes[1].set_title('Present_Price', fontsize=14)
axes[1].set_ylabel("")

sns.boxplot(ax=axes[2], y=df['Selling_Price'], color='darkred', showfliers=True)
axes[2].set_title('Selling_Price', fontsize=14)
axes[2].set_ylabel("")

# Add a global title for the figure
fig.suptitle('Box Plots of Car Data Columns', fontsize=20, weight='bold')

# Display the combined figure
plt.show()

# outliers using z-score method of Driven_kms Selling_Price and Present_Price columns using numpy
from scipy import stats
z = np.abs(stats.zscore(df[['Driven_kms', 'Selling_Price', 'Present_Price']]))
outliers = (z > 3).sum()
print(f'There are total {outliers.sum()} outliers in Driven_kms, Selling_Price and Present_Price columns using Z-score method')

print('_'*100, '\n')

# outliers using iqr method Driven_kms Selling_Price and Present_Price columns
Q1 = df[['Driven_kms', 'Selling_Price', 'Present_Price']].quantile(0.25)
Q3 = df[['Driven_kms', 'Selling_Price', 'Present_Price']].quantile(0.75)
IQR = Q3 - Q1
outliers = (df[['Driven_kms', 'Selling_Price', 'Present_Price']] < (Q1 - 1.5 * IQR)) | (df[['Driven_kms', 'Selling_Price', 'Present_Price']] > (Q3 + 1.5 * IQR))
print(f'There are total {outliers.sum().sum()} outliers in Driven_kms, Selling_Price and Present_Price columns using IQR method')

# Check the shape of the data before removing inconsistency 
print(f'shape of data before removing inconsistency : Rows = {df.shape[0]} & Columns = {df.shape[1]}\n')

# List of bike-related keywords to identify bike rows
bike_keywords = ['Royal Enfield', 'KTM', 'Hero', 'Yamaha', 'TVS', 'Hyosung', 'UM'] # Remove these bike rows

# Filter out rows where Car_Name contains any of the bike keywords
df = df[~df['Car_Name'].str.contains('|'.join(bike_keywords), case=False, na=False)]

# i will remove the rows that shows imbalancing in my data 
df = df[~((df['Owner'] == 3) | (df['Year'] == 2018) | (df['Car_Name'] == '800'))]

# Check the shape of the data after removing inconsistency 
print(f'shape of data after removing inconsistency : Rows = {df.shape[0]} Columns = {df.shape[1]}\n')

# Display message
print('Successfully remove inconsistency')

# Calculate Age of the Car
current_year = 2025
df['Age_of_car'] = current_year - df['Year']

# Compute Car Depreciation
df['Car_depreciation'] = (df['Present_Price'] - df['Selling_Price']).round(3)

# Depreciation Per Year
df['Depreciation_per_year'] = (df['Car_depreciation'] / df['Age_of_car']).round(3)

# Determine Depreciation Rate
df['Depreciation_rate'] = ((df['Present_Price'] - df['Selling_Price']) / df['Present_Price']).round(3)
df['Depreciation_rate'].fillna(0, inplace=True)  # Handle division by zero

# Extract Car Brand from Car_Name
df['Brand'] = df['Car_Name'].str.split().str[0]

# Create Car Age Category
df['Car_Condition'] = pd.cut(df['Age_of_car'],
                          bins=[0, 3, 8, 15, 23],
                          labels=['new', 'young', 'old', 'very old'])

# Create Mileage Category
df['Car_Mileage'] = pd.cut(df['Driven_kms'],
                              bins=[500, 100000, 300000, 500000],
                              labels=['low', 'medium', 'high'])

print('Feature Engineering completed successfully')

# Convert object type columns to category type
categorical_columns = ['Brand', 'Car_Name', 'Fuel_Type', 'Selling_type', 'Transmission']
df[categorical_columns] = df[categorical_columns].astype('category')

# Display the data types of the updated dataset
print("Data types after conversion : \n")
print(df.dtypes)

# Calculate skewness
skewness = df['Selling_Price'].skew()
print(f'Skewness of Selling_Price: {skewness}')

# Plot the distribution of Selling_Price
sns.histplot(df['Selling_Price'], kde=True)
plt.title('Distribution of Selling_Price')
plt.show()

# Apply square root transformation to Selling_Price
df['Selling_Price'] = np.sqrt(df['Selling_Price'])

# Calculate skewness
skewness = df['Selling_Price'].skew()
print(f'Skewness of Selling_Price: {skewness}')

# Plot the transformed distribution
sns.histplot(df['Selling_Price'], kde=True)
plt.title('Distribution of Log-Transformed Selling_Price')
plt.show()

# Select only numerical features
numerical_features = df.select_dtypes(include=["number"])

# Calculate the correlation matrix for numerical features
correlation_matrix = numerical_features.corr()

# Display the correlation matrix
print("\nCorrelation Matrix:")
correlation_matrix

# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='Set1', fmt='.2f')
plt.title("Correlation Matrix")
plt.show()

# Drop unnecessary columns
df = df.drop(columns=['Car_Name'])

print("Successfully Droped the Column")

# Separate features (X) and target (y)
X = df.drop(columns=['Selling_Price'])
y = df['Selling_Price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define categorical and numerical columns
categorical_cols = ['Fuel_Type', 'Selling_type', 'Transmission', 'Brand', 'Car_Condition', 'Car_Mileage']
numerical_cols = ['Present_Price', 'Driven_kms', 'Age_of_car', 'Car_depreciation']

# Preprocessing pipeline
# Numerical columns: MinMax Scaling
# Categorical columns: OneHot Encoding
preprocessor = ColumnTransformer(
    transformers=[
        ('num', MinMaxScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# Define a list of regression models to compare with additional hyperparameters
models = [
    ('Linear Regression', LinearRegression(), {}),
    ('Ridge Regression', Ridge(), {'regressor__alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}),
    ('Lasso Regression', Lasso(), {'regressor__alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}),
    ('Decision Tree', DecisionTreeRegressor(), {'regressor__max_depth': [None, 10, 20, 30], 'regressor__min_samples_split': [2, 5, 10]}),
    ('Random Forest', RandomForestRegressor(), {'regressor__n_estimators': [50, 100, 200], 'regressor__max_depth': [None, 10, 20], 'regressor__min_samples_split': [2, 5, 10]}),
    ('Gradient Boosting', GradientBoostingRegressor(), {'regressor__n_estimators': [50, 100, 200], 'regressor__learning_rate': [0.01, 0.1, 0.2], 'regressor__max_depth': [3, 5, 10]}),
    ('SVR', SVR(), {'regressor__C': [0.1, 1.0, 10.0], 'regressor__kernel': ['linear', 'rbf'], 'regressor__gamma': ['scale', 'auto']}),
    ('KNN', KNeighborsRegressor(), {'regressor__n_neighbors': [3, 5, 7, 9], 'regressor__weights': ['uniform', 'distance']}),
    ('XGBoost', XGBRegressor(), {'regressor__n_estimators': [50, 100, 200], 'regressor__learning_rate': [0.01, 0.1, 0.2], 'regressor__max_depth': [3, 5, 10]})
]

# Initialize a list to store model performance
results = []

# Loop through each model, train, and evaluate
for name, model, params in models:
    # Create a pipeline with preprocessor and model
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    
    # Use GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(pipeline, param_grid=params, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = grid_search.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Store the results
    results.append({
        'Model': name,
        'Best Parameters': grid_search.best_params_,
        'MSE': mse,
        'MAE': mae,
        'R2 Score': r2
    })
    
    print(f"{name}:")
    print(f"  Best Parameters: {grid_search.best_params_}")
    print(f"  MSE: {mse}")
    print(f"  MAE: {mae}")
    print(f"  R2 Score: {r2}")
    print()

# Convert results to a DataFrame and sort by R2 Score in descending order
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by='R2 Score', ascending=False)

# Print the sorted results
print("Model Performance (Sorted by R2 Score):")
print(results_df)

# Print the best model
best_model_result = results_df.iloc[0]
print("\nBest Model:")
print(f"  Model: {best_model_result['Model']}")
print(f"  Best Parameters: {best_model_result['Best Parameters']}")
print(f"  MSE: {best_model_result['MSE']}")
print(f"  MAE: {best_model_result['MAE']}")
print(f"  R2 Score: {best_model_result['R2 Score']}")

# Extract the best model from the results DataFrame
best_model_name = results_df.iloc[0]['Model']
best_model_params = results_df.iloc[0]['Best Parameters']

# Find the corresponding model object
best_model = None
for name, model, params in models:
    if name == best_model_name:
        best_model = model
        break

# Create a pipeline with the preprocessor and the best model
best_model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', best_model)
])

# Set the best parameters to the model
best_model_pipeline.set_params(**best_model_params)

# Fit the best model pipeline
best_model_pipeline.fit(X_train, y_train)

# Save the best model to a pickle file
with open('best_model.pkl', 'wb') as file:
    pickle.dump(best_model_pipeline, file)

# Load the model from the pickle file
with open('best_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Make predictions using the loaded model
loaded_model_predictions = loaded_model.predict(X_test)

# Evaluate the loaded model
loaded_model_mse = mean_squared_error(y_test, loaded_model_predictions)
loaded_model_mae = mean_absolute_error(y_test, loaded_model_predictions)
loaded_model_r2 = r2_score(y_test, loaded_model_predictions)

print(f"Loaded Model Evaluation:")
print(f"  MSE: {loaded_model_mse}")
print(f"  MAE: {loaded_model_mae}")
print(f"  R2 Score: {loaded_model_r2}")
