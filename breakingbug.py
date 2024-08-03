# import libraries

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from yellowbrick.cluster import KElbowVisualizer
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_absolute_error, mean_squared_error, r2_score
import warnings

# 1. to handle the data
import pandas as pd
import numpy as np

# 2. To Viusalize the data
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from yellowbrick.cluster import KElbowVisualizer
from matplotlib.colors import ListedColormap

# 3. To preprocess the data
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer

# 4. import Iterative imputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Error:  from sklearn.model_selection import train_test_split,GridSearch, cross_val

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

# Error:  import statements are incorrect 
from sklearn import LogisticRegressions
from sklearn import KNN
from sklearn import SVC_Classifier
from sklearn import DecisionTree, plot_tree_regressor
from sklearn import RandomForestRegressor, AdaBoost, GradientBoost
from xgboost import XG
from lightgbm import LGBM
from sklearn import Gaussian

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestRegressor, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.naive_bayes import GaussianNB

7. Metrics
# Error : from sklearn.metrics import accuracy, confusion, classification

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# 8. Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Incorrect import statement

df = pd.read_csv(r"C:\Users\ragha\Downloads\dataset.csv")

# print the first 5 rows of the dataframe
df.head()

# Exploring the data type of each column
df.info()

# Checking the data shape
df.shape

# Id column
df['id'].min(), df['id'].max()

# age column
df['age'].min(), df['age'].max()

# lets summerize the age column
df['age'].describe()

import seaborn as sns

# Define custom colors
custom_colors = ["#FF5733", "#3366FF", "#33FF57"]  # Example colors, you can adjust as needed

# Plot the histogram with custom colors
sns.histplot(df['age'], kde=True, color="#FF5733", palette=custom_colors)

# Plot the mean, Median and mode of age column using sns
sns.histplot(df['age'], kde=True)
plt.axvline(df['age'].mean(), color='Red')
plt.axvline(df['age'].median(), color= 'Green')
plt.axvline(df['age'].mode()[0], color='Blue')

# print the value of mean, median and mode of age column
print('Mean', df['age'].mean())
print('Median', df['age'].median())
print('Mode', df['age'].mode())

# plot the histogram of age column using plotly and coloring this by sex

fig = px.histogram(data_frame=df, x='age', color= 'sex')
fig.show()

# Find the values of sex column
df['sex'].value_counts()

# calculating the percentage fo male and female value counts in the data

male_count = 726
female_count = 194

total_count = male_count + female_count

# calculate percentages
male_percentage = (male_count/total_count)*100
female_percentages = (female_count/total_count)*100

# display the results
print(f'Male percentage i the data: {male_percentage:.2f}%')
print(f'Female percentage in the data : {female_percentages:.2f}%')

# Difference
difference_percentage = ((male_count - female_count)/female_count) * 100
print(f'Males are {difference_percentage:.2f}% more than female in the data.')

726/194

# Find the values count of age column grouping by sex column
df.groupby('sex')['age'].value_counts()

#  Error : df['dataset'].counts()

# find the unique values in the dataset column
df['dataset'].value_counts()

# plot the countplot of dataset column
fig =px.bar(df, x='dataset', color='sex')
fig.show()

# print the values of dataset column groupes by sex
print (df.groupby('sex')['dataset'].value_counts())

# make a plot of age column using plotly and coloring by dataset

fig = px.histogram(data_frame=df, x='age', color= 'dataset')
fig.show()



# Use df['data'] instead of df('data')
# The correct method to calculate the mode is df['data']['age'].mode()

print("___________________________________________________________")
print("Mean of the dataset: ", df.groupby('dataset')['age'].mean())
print("___________________________________________________________")
print("Median of the dataset: ", df.groupby('dataset')['age'].median())
print("___________________________________________________________")
print("Mode of the dataset: ", df.groupby('dataset')['age'].apply(lambda x: x.mode()[0]))
print("___________________________________________________________")

# value count of cp column
df['cp'].value_counts()

# count plot of cp column by sex column
sns.countplot(df, x='cp', hue= 'sex')

# count plot of cp column by dataset column
sns.countplot(df,x='cp',hue='dataset')

# Draw the plot of age column group by cp column

fig = px.histogram(data_frame=df, x='age', color='cp')
fig.show()

# lets summerize the trestbps column
df['trestbps'].describe()

# Dealing with Missing values in trestbps column.
# find the percentage of misssing values in trestbps column
print(f"Percentage of missing values in trestbps column: {df['trestbps'].isnull().sum() /len(df) *100:.2f}%")

# Impute the missing values of trestbps column using iterative imputer
# create an object of iteratvie imputer
imputer1 = IterativeImputer(max_iter=10, random_state=42)

# Fit the imputer on trestbps column
imputer1.fit(df[['trestbps']])

# Transform the data
df['trestbps'] = imputer1.transform(df[['trestbps']])

# Check the missing values in trestbps column
print(f"Missing values in trestbps column: {df['trestbps'].isnull().sum()}")

# First lets see data types or category of columns
df.info()

# let's see which columns has missing values
(df.isnull().sum()/ len(df)* 100).sort_values(ascending=False)

# create an object of iterative imputer
imputer2 = IterativeImputer(max_iter=10, random_state=42)

# Error: imputer2.Fit_transform

imputer = SimpleImputer(strategy='mean')

# Apply the imputer to each column
df['ca'] = imputer.fit_transform(df[['ca']])
df['oldpeak'] = imputer.fit_transform(df[['oldpeak']])
df['chol'] = imputer.fit_transform(df[['chol']])
df['thalch'] = imputer.fit_transform(df[['thalch']])

# let's check again for missing values
(df.isnull().sum()/ len(df)* 100).sort_values(ascending=False)

print(f"The missing values in thal column are: {df['thal'].isnull().sum()}")

df['thal'].value_counts()

df.tail()

#  Error df.null().sum()[df.null()()<0].values(ascending=true)

# find missing values.
(df.isnull().sum() / len(df) * 100).sort_values(ascending=False)


missing_data_cols = df.isnull().sum()[df.isnull().sum()>0].index.tolist()

missing_data_cols

# find categorical Columns
cat_cols = df.select_dtypes(include='object').columns.tolist()
cat_cols

# find Numerical Columns
Num_cols = df.select_dtypes(exclude='object').columns.tolist()
Num_cols

print(f'categorical Columns: {cat_cols}')
print(f'numerical Columns: {Num_cols}')

# FInd columns
categorical_cols = ['thal', 'ca', 'slope', 'exang', 'restecg','thalch', 'chol', 'trestbps']
bool_cols = ['fbs']
numerical_cols = ['oldpeak','age','restecg','fbs', 'cp', 'sex', 'num']

# This function imputes missing values in categorical columnsdef impute_categorical_missing_data(passed_col):
passed_col = categorical_cols

# Indentation Errors: 
# Variable Names: 
# Unused/Incorrect Variables: 
# ogical Errors: Indentation Errors: 
# Variable Names: 
# Unused/Incorrect Variables: 
# Logical Errors: 



def impute_categorical_missing_data(passed_col, df, missing_data_cols, bool_cols):
    # Separate rows with missing and non-missing values for the given column
    df_null = df[df[passed_col].isnull()]
    df_not_null = df[df[passed_col].notnull()]

    # Prepare feature matrix and target variable
    X = df_not_null.drop(passed_col, axis=1)
    y = df_not_null[passed_col]

    # Initialize label and one-hot encoders
    label_encoder = LabelEncoder()
    onehot_encoder = OneHotEncoder(sparse=False, drop='first')

    # Handle categorical features
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = onehot_encoder.fit_transform(X[col].astype(str))
    
    # Encode target variable if it's a boolean column
    if passed_col in bool_cols:
        y = label_encoder.fit_transform(y)

    # Impute missing values for other columns
    other_missing_cols = [col for col in missing_data_cols if col != passed_col]
    imputer = IterativeImputer(estimator=RandomForestRegressor(random_state=16))

    for col in other_missing_cols:
        if X[col].isnull().any():
            X[col] = imputer.fit_transform(X[[col]])

    # Split data for training and evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_classifier = RandomForestClassifier()
    rf_classifier.fit(X_train, y_train)
    
    y_pred = rf_classifier.predict(X_test)
    acc_score = accuracy_score(y_test, y_pred)
    
    print(f"The feature '{passed_col}' has been imputed with {round((acc_score * 100), 2)}% accuracy\n")

    # Impute missing values in df_null
    X_null = df_null.drop(passed_col, axis=1)
    
    for col in X_null.columns:
        if X_null[col].dtype == 'object':
            X_null[col] = onehot_encoder.transform(X_null[col].astype(str))

    for col in other_missing_cols:
        if X_null[col].isnull().any():
            X_null[col] = imputer.transform(X_null[[col]])

    # Predict missing values and update the original dataframe
    if len(df_null) > 0:
        df[passed_col] = rf_classifier.predict(X_null)
        if passed_col in bool_cols:
            df[passed_col] = df[passed_col].map({0: False, 1: True})

    df_combined = pd.concat([df_not_null, df_null])

    return df_combined


import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def impute_continuous_missing_data(passed_col, df, missing_data_cols):
    # Separate rows with missing and non-missing values for the given column
    df_null = df[df[passed_col].isnull()]
    df_not_null = df[df[passed_col].notnull()]

    # Prepare feature matrix and target variable
    X = df_not_null.drop(passed_col, axis=1)
    y = df_not_null[passed_col]

    other_missing_cols = [col for col in missing_data_cols if col != passed_col]

    # Initialize label and one-hot encoders
    label_encoder = LabelEncoder()
    onehot_encoder = OneHotEncoder(sparse=False, drop='first')

    # Encode categorical features
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = onehot_encoder.fit_transform(X[col].astype(str))

    # Impute missing values for other columns
    imputer = IterativeImputer(estimator=RandomForestRegressor(random_state=16))

    for col in other_missing_cols:
        if X[col].isnull().any():
            X[col] = imputer.fit_transform(X[[col]])

    # Split data for training and evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_regressor = RandomForestRegressor()
    rf_regressor.fit(X_train, y_train)
    
    y_pred = rf_regressor.predict(X_test)

    print("MAE =", mean_absolute_error(y_test, y_pred))
    print("RMSE =", mean_squared_error(y_test, y_pred, squared=False))
    print("R2 =", r2_score(y_test, y_pred))

    # Impute missing values in df_null
    X_null = df_null.drop(passed_col, axis=1)

    for col in X_null.columns:
        if X_null[col].dtype == 'object':
            X_null[col] = onehot_encoder.transform(X_null[col].astype(str))

    for col in other_missing_cols:
        if X_null[col].isnull().any():
            X_null[col] = imputer.transform(X_null[[col]])

    if len(df_null) > 0:
        df[passed_col] = rf_regressor.predict(X_null)
    
    df_combined = pd.concat([df_not_null, df_null])

    return df_combined


Indentation Errors:
Variable Naming:
Uninitialized Variables: 

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Example impute functions
def impute_categorical_missing_data(df, col):
    # Assuming some imputation method for categorical data
    df[col].fillna(df[col].mode()[0], inplace=True)
    return df

def impute_continuous_missing_data(df, col):
    # Assuming some imputation method for continuous data
    imputer = IterativeImputer(estimator=RandomForestRegressor(random_state=16))
    df[col] = imputer.fit_transform(df[[col]])
    return df

# Example column lists
categorical_cols = ['gender', 'fbs', 'restecg']  # Update with your actual categorical columns
numeric_cols = ['age', 'cholesterol']  # Update with your actual numeric columns
missing_data_cols = categorical_cols + numeric_cols  # Assuming all are being checked

# Sample DataFrame for demonstration (use your actual DataFrame)
df = pd.DataFrame({
    'gender': ['Male', 'Female', np.nan],
    'fbs': [np.nan, 'Normal', 'High'],
    'restecg': ['Normal', 'Abnormal', np.nan],
    'age': [25, 30, np.nan],
    'cholesterol': [200, np.nan, 240]
})

# Print missing values percentage
for col in missing_data_cols:
    if col in df.columns:
        print(f"Missing Values {col} : {round((df[col].isnull().sum() / len(df)) * 100, 2)}%")
        if col in categorical_cols:
            df = impute_categorical_missing_data(df, col)
        elif col in numeric_cols:
            df = impute_continuous_missing_data(df, col)
    else:
        print(f"Warning: Column '{col}' not found in DataFrame.")

# Display remaining missing values
print(df.isnull().sum().sort_values(ascending=False))


# Split the data into features and target variable
X = df.drop('num', axis=1)
y = df['num']

# Encode categorical columns
label_encoders = {}
for col in X.select_dtypes(include='object').columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Define models and evaluate
models = [
    ('Logistic Regression', LogisticRegression(random_state=42)),
    ('Gradient Boosting', GradientBoostingClassifier(random_state=42)),
    ('KNeighbors Classifier', KNeighborsClassifier()),
    ('Decision Tree Classifier', DecisionTreeClassifier(random_state=42)),
    ('AdaBoost Classifier', AdaBoostClassifier(random_state=42)),
    ('Random Forest', RandomForestClassifier(random_state=42)),
    ('XGboost Classifier', XGBClassifier(random_state=42)),
    ('Support Vector Machine', SVC(random_state=42)),
    ('Naive Bayes Classifier', GaussianNB())
]

best_model = None
best_accuracy = 0.0

for name, model in models:
    pipeline = Pipeline([
        ('model', model)
    ])
    scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
    mean_accuracy = scores.mean()
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Model: {name}")
    print(f"Cross Validation Accuracy: {mean_accuracy:.2f}")
    print(f"Test Accuracy: {accuracy:.2f}")
    print()

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = name

print(f"Best Model: {best_model}")

# Hyperparameter tuning
def hyperparameter_tuning(X, y, models):
    results = {}
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grids = {
        'Logistic Regression': {'C': [0.1, 1, 10, 100]},
        'KNeighbors Classifier': {'n_neighbors': [3, 5, 7, 9]},
        'GaussianNB': {'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]},
        'SVC': {'C': [0.1, 1, 10, 100], 'gamma': [0.1, 1, 10, 100]},
        'Decision Tree': {'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10]},
        'Random Forest': {'n_estimators': [100, 200, 300], 'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10]},
        'XGboost Classifier': {'learning_rate': [0.01, 0.1, 0.2], 'n_estimators': [100, 200, 300], 'max_depth': [3, 5, 7]},
        'Gradient Boosting': {'learning_rate': [0.01, 0.1, 0.2], 'n_estimators': [100, 200, 300], 'max_depth': [3, 5, 7]},
        'AdaBoost Classifier': {'n_estimators': [50, 100, 150]}
    }

    for name, model in models:
        param_grid = param_grids.get(name, {})
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        print(f"Model: {name}")
        print(f"Best Parameters: {best_params}")
        print(f"Best Cross Validation Score: {best_score:.2f}")
        results[name] = best_params

    return results

best_params = hyperparameter_tuning(X, y, models)
print("Best Parameters:", best_params)

# Train and evaluate the final model with the best parameters
final_model = RandomForestClassifier(n_estimators=200, max_depth=30, min_samples_split=5, random_state=42)
final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)
print(f"Final Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
