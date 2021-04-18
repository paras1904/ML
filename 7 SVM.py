import numpy as np
x = np.arange(10).reshape(-1, 1)
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

from sklearn.svm import SVC
svclassifier = SVC(kernel='linear')
svclassifier.fit(x,y)
a = svclassifier.predict(x)
print(a)