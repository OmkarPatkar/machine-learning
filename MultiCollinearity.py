# Import the library
import pandas as pd
import statsmodels.api as st

# Load the data (low collinearity data)
data = pd.read_csv('advertising.csv', index_col = 0)
x = data[['TV', 'radio', 'newspaper']]
y = data['sales']

print(data.head())
print('*' * 50)

# y = B0 + B1x1 + B2x2 + B3x3 + B4x4
# we need to add the B0 value as a constant to use the data in OLS(Ordinary Least Squares)
x = st.add_constant(x)
print(x)
print('*' * 50)

# Now we can use OLS method
model = st.OLS(y, x).fit()
summary = model.summary()
print(summary)
print('*' * 50)

# Plot the data
from matplotlib import pyplot as plt
corr = x.iloc[:,1:].corr()
print(corr)
print('=' * 50)


# Load the data (high collinearity data)
df = pd.read_csv('salary.csv')
print(df.head())
print('*' * 50)

x1 = df[['YearsExperience', 'Age']]
y1 = df['Salary']

## add a constant and fit a OLS model
x1 = st.add_constant(x1)
model1 = st.OLS(y1, x1).fit()
print(model1.summary())
print('*' * 50)

# Plot the data
print(x1.iloc[:, 1:].corr())
print('*' * 50)