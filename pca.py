# Author: Emelie Wästlund
# 2022-01-05
# DD2434

# Run by creating a PCA-object with the data matrix as a parameter and then call the transform method with the desired number of features as a parameter.
# Ex:
# pca = PCA(Y)
# X = pca.transform(k)

import numpy as np

class PCA():
    def __init__(self, Y):
        self.Y = Y
        self.eig_vectors = self.find_eig_vectors(Y)
    def find_eig_vectors(self, Y):
        '''
        Takes in the data matrix Y (of dimension n×d) and returns the matrix of sorted eigen vectors
        '''
        # Centering
        Y_sum = np.sum(Y, 0)
        Y_mean = Y_sum / Y.shape[0]
        Y = (Y - Y_mean)
        # Finding covariance matrix
        cov = np.cov(Y.T)
        # Finding eigenvalue decomposition
        eig_vals, eig_vectors = np.linalg.eig(cov)
        indices = list(reversed(np.argsort(eig_vals)))
        eig_vectors = eig_vectors[:,indices]
        return eig_vectors
    def transform(self, k):
        '''
        Takes in a desired number of features and returns a projected matrix X (of dimension n×k)
        '''
        # Reducing to k dimensions
        E_k = self.eig_vectors[:,:k]
        X = np.matmul(E_k.T, self.Y.T)
        return X.T
