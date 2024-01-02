import numpy as np
from distances import euclidean_distance

def linear_kernel(X, Y):
    K = np.dot(X, Y.T)
    return K

def polynomial_kernel(X, Y, degree):
    K = np.power(np.dot(X, Y.T) + 1, degree)
    return K

def rbf_kernel(X, Y, gamma):
    dists = np.square(euclidean_distance(X[:, np.newaxis], Y, axis=2))
    K = np.exp(-gamma * dists)
    return K

class LinearKernel:

    def transform(self, X, Y):
        return linear_kernel(X, Y)


class PolynomialKernel:
    
    def __init__(self, degree = 2):
        self.degree = degree

    def transform(self, X, Y):
        return polynomial_kernel(X, Y, self.degree)
        
class RBFKernel:

    def __init__(self, gamma = 1.0):
        self.gamma = gamma
        
    def transform(self, X, Y):
        return rbf_kernel(X, Y, self.gamma)
