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
