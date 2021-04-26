import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

x = -2 * np.random.rand(200,2)
x0 = 1 + 2 * np.random.rand(100,2)
x[100:200, :] = x0

plt.scatter(x[ : , 0], x[ :, 1], s = 25, color='r')
plt.grid()


Kmean = KMeans(n_clusters=3)
Kmean.fit(x)
Kmean.cluster_centers_

plt.scatter(2.03078996,  2.05446538, s=100, color='green')
plt.show()