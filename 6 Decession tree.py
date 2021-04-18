import numpy as np
from sklearn.tree import DecisionTreeClassifier
x = np.arange(10).reshape(-1, 1)
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

model = DecisionTreeClassifier()
model.fit(x, y)
a = model.predict_proba(x)
print(a)