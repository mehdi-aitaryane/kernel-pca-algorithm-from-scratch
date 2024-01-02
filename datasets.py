import numpy as np

# This function prints the shape of the feature and target arrays
# X: a 2D array of features
# y: a 1D array of targets

def print_shape(X, y):
    print("X shape is ", X.shape)
    print("y shape is ", y.shape)

def make_circles(n_samples=100, noise=0.0, random_state=None, n_circles=2, balanced = True, n_features=2):
    np.random.seed(random_state)
    # Create balanced or unbalanced samples
    size = n_samples // n_circles
    rest = n_samples % n_circles
    prob = np.random.uniform(size=(n_circles))
    prob = np.where(balanced == True, [1/n_circles] * n_circles, prob / sum(prob))
    # Generate target values
    Y = np.random.choice(n_circles, n_samples , replace=True, p=prob)
    Y = np.sort(Y)
    class_sizes = np.bincount(Y)
    factors = np.sort(np.random.uniform(size=(n_circles,)))
    X = []
    for i in range(n_circles):
        theta = np.random.rand(class_sizes[i]) * 2 * np.pi
        r = np.random.uniform(low=factors[i], high=factors[i], size=(class_sizes[i]))
        x, y = r * np.cos(theta), r * np.sin(theta)
        X.append(np.vstack((x, y)).T)
    X = np.vstack(X)
    if n_features > 2:
        extra_features = np.random.uniform(size=(n_samples, n_features - 2))
        X = np.hstack((X, extra_features))
    noise = np.random.normal(scale=noise, size=(n_samples, n_features))
    X = X + noise
    return X, Y