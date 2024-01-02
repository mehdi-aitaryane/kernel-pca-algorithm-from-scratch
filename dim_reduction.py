import numpy as np
from kernels import RBFKernel

class PCA:

    def __init__(self, n_components = 2):
        self.n_components = n_components
        self.mean = None
        self.components = None

    def fit(self, X):
        # Mean centering
        self.mean = np.mean(X, axis=0)
        centered_X = X - self.mean

        # Compute the covariance matrix
        covariance_matrix = np.cov(centered_X, rowvar=False)

        # Compute eigenvectors and eigenvalues
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

        # Sort eigenvectors by decreasing eigenvalues
        sorted_indices = np.argsort(eigenvalues)[::-1]
        self.components = eigenvectors[:, sorted_indices]

        if self.n_components is not None:
            if self.n_components < 1:
                # Interpret n_components as the desired variance retained
                explained_variance_ratio = eigenvalues[sorted_indices] / np.sum(eigenvalues)
                cumulative_variance = np.cumsum(explained_variance_ratio)
                self.n_components = np.argmax(cumulative_variance >= self.n_components) + 1
                self.components = self.components[:, :self.n_components]
            else:
                # If n_components is a positive integer, use it directly
                self.components = self.components[:, :self.n_components]

    def transform(self, X):
        # Mean centering
        centered_X = X - self.mean

        # Project data onto the new feature space
        transformed_data = np.dot(centered_X, self.components)

        return transformed_data

    def fit_transform(self, X):
        # Call fit
        self.fit(X)

        # Call transform
        return self.transform(X)

    def inverse_transform(self, X):
        # Project data back to the original feature space
        original_data = np.dot(X, self.components.T) + self.mean

        return original_data

class KernelPCA(PCA):

    def __init__(self, n_components, kernel = RBFKernel()):
        super().__init__(n_components = n_components)
        self.kernel = kernel
        self.X = None

    def fit(self, X):
        self.X = X
        K = self.kernel.transform(X, self.X)
        super().fit(K)

    def transform(self, X):
        K = self.kernel.transform(X, self.X)
        return super().transform(K).real

    def inverse_transform(self, X):
        K = self.kernel.transform(X, self.X)
        return super().inverse_transform(K).real