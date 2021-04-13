import numpy as np
x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
y = np.array([5, 20, 14, 32, 22, 38])
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x,y)
print(lr.predict([[15]]))
