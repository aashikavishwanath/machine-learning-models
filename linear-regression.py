import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# Importing necessary libraries for data manipulation, numerical operations, plotting, and visualization.

housing = pd.read_csv('USA_Housing.csv')
housing.head()
housing.info()

# Loading the 'USA_Housing.csv' file into a Pandas DataFrame named 'housing'.
# Displaying the first few rows of the DataFrame.
# Providing information about the DataFrame, including column names, data types, and non-null values.

sns.pairplot(housing)

# Creating a pairwise scatter plot matrix using Seaborn's 'pairplot' function.
# Visualizing the relationships between all pairs of numerical variables in the 'housing' DataFrame.

x = housing['Avg. Area Income']
y = housing['Price']
plt.plot(x, y)

# Creating a line plot using matplotlib to visualize the relationship between 'Avg. Area Income' and 'Price'
# in the 'housing' DataFrame.

x = 'Avg. Area Income'
y = 'Price'
sns.lmplot(x, y, housing)

# Creating a regression plot using Seaborn's 'lmplot' function to show the linear relationship
# between 'Avg. Area Income' and 'Price' in the 'housing' DataFrame.

x = 'Avg. Area Number of Rooms'
y = 'Price'
sns.lmplot(x, y, housing)

# Creating a regression plot using Seaborn's 'lmplot' function to visualize the linear relationship
# between 'Avg. Area Number of Rooms' and 'Price' in the 'housing' DataFrame.

x = 'Avg. Area Number of Bedrooms'
y = 'Price'
sns.lmplot(x, y, housing)

# Creating a regression plot using Seaborn's 'lmplot' function to visualize the linear relationship
# between 'Avg. Area Number of Bedrooms' and 'Price' in the 'housing' DataFrame.

sns.distplot(housing['Price'], bins=100)

# Creating a distribution plot using Seaborn's 'distplot' function to visualize the distribution of the 'Price' column in the 'housing' DataFrame.

sns.heatmap(housing.corr(), cmap="YlGnBu")

# Creating a heatmap using Seaborn's 'heatmap' function to visualize the correlation between numerical variables in the 'housing' DataFrame.

housing.columns

# Displaying the column names of the 'housing' DataFrame.

X = housing[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
             'Avg. Area Number of Bedrooms', 'Area Population']]
y = housing['Price']

# Creating feature matrix 'X' with selected columns and target vector 'y' from the 'housing' DataFrame.

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Splitting the data into training and testing sets using the train_test_split function from scikit-learn.
# Using 33% of the data for testing and setting the random state to 42 for reproducibility.

from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, y_train)
predictions = lm.predict(X_test)
plt.scatter(y_test, predictions)

# Training a Linear Regression model on the training data.
# Making predictions on the test data and creating a scatter plot to compare the actual 'Price' values with the predicted values.
