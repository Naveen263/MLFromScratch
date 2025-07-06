import numpy as np
import matplotlib.pyplot as plt



class KmeansClustering:
    def __init__(self, k:int=3, max_iter:int=200):
        self.k = k
        self.max_iter = max_iter
        self.centroids = None
    
    def euclidean_distance(self, x1:np.ndarray, centroids:np.ndarray):
        return np.sqrt(np.sum((centroids-x1)**2,axis=1))

    def fit(self, X:np.ndarray):

        #Initialize centroids with random samples
        self.centroids = np.random.uniform(np.min(X,axis=0),np.max(X, axis=0), size=(self.k, X.shape[1]))
        n,d = X.shape
        Y = np.array([-1 for i in range(n)])

        for i in range(self.max_iter):
            Y_new = np.array([-1 for i in range(n)])
            
            for j in range(n):
                distances  = self.euclidean_distance(X[j],self.centroids)
                Y_new[j] = np.argmin(distances)
            
            # Check for convergence of centroids
            if np.all(Y == Y_new):
                break

            Y = Y_new
            #Recompute centroids
            for c in range(self.k):
                self.centroids[c] = np.mean(X[Y == c], axis=0)
        
        return Y


X = np.random.randint(0,100, size=(100,2))            
kmeans = KmeansClustering(k=3)
Y = kmeans.fit(X)
plt.scatter(X[:,0], X[:,1], c=Y)
plt.scatter(kmeans.centroids[:,0], kmeans.centroids[:,1], c=range(kmeans.k), marker='*', s=100)
plt.savefig('./kmeans_clustering.png')