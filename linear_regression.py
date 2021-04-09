# Multiple Linear Regression

# Importing the libraries
import pandas as pd

# Load the data
data = pd.read_csv('50_startups.csv')
x = data.iloc[:,:-1]
y = data.iloc[:, 4]

#Convert the column into categorical columns
states = pd.get_dummies(x['State'], drop_first = True)

# Drop the state coulmn
x = x.drop('State', axis = 1)

# concat the dummy variables
x = pd.concat([x, states], axis = 1)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train, y_train)

# Predicting the Test set results
y_pred = model.predict(x_test)

# Checking the r2_score
from sklearn.metrics import r2_score
score = r2_score(y_test, y_pred)
print(f'Score is : {score}')