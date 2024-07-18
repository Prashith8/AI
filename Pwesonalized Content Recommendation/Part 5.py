import pandas as pd
df = pd.read_csv('amazon.csv')
print(df)
print(df.dtypes)
print(df.head())
print(df.tail())
df.info()
df.describe()
#Check for null values in the DataFrame
print(df.isnull().sum()) #Drop rows 
data = {
'product_id': [1, 2, 3, 4, 5],
'product_name': ['Product A', 'Product B', 'Product C', 'Product D', 'Product E'],
'category': ['Electronics', 'Clothing', 'Electronics', 'Books', 'Books'],
'price': [100, 50, 120, 20, 30],
'quantity_sold': [10, 20, 15, 30, 25]
}
# Create a DataFrame from the data
df = pd.DataFrame(data)
# Aggregate data by category
agg_data = df.groupby('category').agg(
total_quantity_sold=('quantity_sold', 'sum'), total_revenue=('price', lambda x: (x * df['quantity_sold']).sum()) # Total revenue = price * quantity_sold
).reset_index()
print(agg_data)
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# Sample Amazon dataset (replace this with your actual dataset)
data = {
'product_id': [1, 2, 3, 4, 5],
'product_name': ['Product A', 'Product B', 'Product C', 'Product D', 'Product E'],
'category': ['Electronics', 'Clothing', 'Electronics', 'Books', 'Books'],
'price': [100, 50, 120, 20, 30],
'quantity_sold': [10, 20, 15, 30, 25]
}
# Create a DataFrame from the data
df = pd.DataFrame(data)
# Univariate analysis: Histogram of price
plt.figure(figsize=(8, 6))
sns.histplot(df['price'], bins=10, kde=True, color='skyblue')
plt.title('Histogram of Price')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()
# Bivariate analysis: Scatter plot of price vs. quantity_sold
plt.figure(figsize=(8, 6))
sns.scatterplot(x='price', y='quantity_sold', data=df, color='salmon')
plt.title('Scatter Plot of Price vs. Quantity Sold')
plt.xlabel('Price')
plt.ylabel('Quantity Sold')
plt.show()
# Multivariate analysis: Pairplot of price, quantity_sold, and category
sns.pairplot(df[['price', 'quantity_sold', 'category']], hue='category', height=3)
plt.suptitle('Pairplot of Price, Quantity Sold, and Category')
plt.show()
import pandas as pd
# Sample Amazon dataset (replace this with your actual dataset)
data = {
'product_id': [1, 2, 3, 4, 5],
'product_name': ['Product A', 'Product B', 'Product C', 'Product D', 'Product E'],
'category': ['Electronics', 'Clothing', 'Electronics', 'Books', 'Books'],
'price': [100, 50, 120, 20, 30],
'quantity_sold': [10, 20, 15, 30, 25]
}
# Create a DataFrame from the data
df = pd.DataFrame(data)
# Example of feature engineering: calculating total revenue for each product
df['total_revenue'] = df['price'] * df['quantity_sold']
# Another example: creating a binary feature indicating whether the product is expensive or not
df['is_expensive'] = df['price'].apply(lambda x: 1 if x > 100 else 0)
# Yet another example: creating a feature indicating the length of the product name
df['product_name_length'] = df['product_name'].apply(len)
# Print the updated DataFrame with engineered features
print(df)
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