import numpy as np
from sklearn.linear_model import LogisticRegression
x = np.arange(10).reshape(-1, 1)
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

model = LogisticRegression(solver='liblinear', random_state=0)
model.fit(x, y)
a = model.predict_proba(x)
print(a)