import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler, LabelEncoder 
# Load data
data = pd.read_csv('data.csv') 
# Handle missing values 
data.fillna(method='ffill', inplace=True) 
# Encoding categorical variables (if any) label_encoders = {} for column in 
data.select_dtypes(include=['object']).columns: 
le = LabelEncoder() 
data[column] = le.fit_transform(data[column]) label_encoders[column] = le 
# Split data into features and target
X = data.drop('acousticness', axis=1) # assuming 'price' is the target variable y = 
data['acousticness']
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
# Feature scaling
scaler = StandardScaler() 
X_train = scaler.fit_transform(X_train) 
X_test = scaler.transform(X_test)
# Plotting distributions sns.histplot(y, 
kde=True) 
plt.title('acousticness Distribution') plt.show() 
# Correlation matrix plt.figure(figsize=(12, 8)) 
sns.heatmap(data.corr(), annot=True, cmap='coolwarm') plt.title('Correlation Matrix') 
plt.show()
from sklearn.linear_model import LinearRegression 
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor 
from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score 
# Initialize models models = { 
'Linear Regression': LinearRegression(), 
'Random Forest': RandomForestRegressor(random_state=42), 
'Gradient Boosting': GradientBoostingRegressor(random_state=42) 
for name, model in models.items():
model.fit(X_train, y_train) print(f"{name} trained.")   
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score 
# Assuming 'models' is a dictionary with model names as keys and model instances as values
results = {} for name, model in models.items(): 
y_pred = model.predict(X_test) results[name] = { 
'RMSE': mean_squared_error(y_test, y_pred, squared=False), 
'MAE': mean_absolute_error(y_test, y_pred), 
'R^2': r2_score(y_test, y_pred) 
} 
# Print evaluation results for name, metrics in 
results.items(): print(f"Model: {name}") for metric, 
value in metrics.items(): print(f"{metric}: {value}") 
print("\n")
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler, LabelEncoder from 
sklearn.linear_model import LinearRegression from sklearn.ensemble import 
RandomForestRegressor, GradientBoostingRegressor 
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score 
from sklearn.model_selection import GridSearchCV 
import matplotlib.pyplot as plt import seaborn as sns 
# Load data
data = pd.read_csv('data.csv') 
# Handle missing values 
data.fillna(method='ffill', inplace=True) 
# Encoding categorical variables label_encoders = {} for column in 
data.select_dtypes(include=['object']).columns: le = LabelEncoder() 
data[column] = le.fit_transform(data[column]) label_encoders[column] = le
# Split data into features and target X = 
data.drop('acousticness', axis=1) y = data['acousticness'] 
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
# Feature scaling
scaler = StandardScaler() 
X_train = scaler.fit_transform(X_train) 
X_test = scaler.transform(X_test) 
# Exploratory Data Analysis sns.histplot(y, kde=True) 
plt.title('acousticness Distribution') plt.show() 
plt.figure(figsize=(12, 8)) 
sns.heatmap(data.corr(), annot=True, cmap='coolwarm') plt.title('Correlation 
Matrix') plt.show() 
# Initialize models models = { 
'Linear Regression': LinearRegression(), 
'Random Forest': RandomForestRegressor(random_state=42), 
'Gradient Boosting': GradientBoostingRegressor(random_state=42) 
} 
# Train models for name, model in models.items(): 
model.fit(X_train, y_train) print(f"{name} trained.") 
# Evaluate models 
results = {} for name, model in models.items(): y_pred 
= model.predict(X_test) results[name] = { 
'RMSE': mean_squared_error(y_test, y_pred, squared=False), 
'MAE': mean_absolute_error(y_test, y_pred), 
'R^2': r2_score(y_test, y_pred) 
} 
# Print evaluation results for name, metrics in 
results.items(): print(f"Model: {name}") for metric, valuein metrics.items(): print(f"{metric}: {value}") print("\n")