import numpy as np
from matplotlib import pyplot as plt


class kNN:
    def __init__(self, k:int=3):
        self.k = k
        self.X = None
        self.Y = None


    def fit(self, X:np.ndarray, Y:np.ndarray):
        self.X = X
        self.Y = Y


    def euclidean_distance(self, X:np.ndarray, x1:np.ndarray):
        return np.sqrt(np.sum((X-x1)**2, axis=1))


    def predict(self, x1:np.ndarray):
        distances = self.euclidean_distance(self.X, x1)
        k_nearest_neighbors = np.argsort(distances)[:self.k]
        k_nearest_neighbors_labels = np.array([self.Y[ind] for ind in k_nearest_neighbors])

        return np.bincount(k_nearest_neighbors_labels).argmax()


X = np.array([[1,2],[2,3],[3,4],[6,7],[7,8],[8,9],[10,11],[11,12],[12,13]])
Y = np.array([0,0,0,1,1,1,0,0,0])

knn = kNN(k=3)

knn.fit(X,Y)

label = knn.predict(np.array([[10,10]]))

plt.scatter(X[:,0], X[:,1], c=Y)
plt.scatter(10,10, c=Y[label], marker='*', s=100)
plt.savefig('./knn.png')





