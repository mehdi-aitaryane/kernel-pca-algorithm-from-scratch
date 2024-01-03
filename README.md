# Kernel PCA Algorithm From Scratch

## Introduction

This repository contains a Python implementation of the Kernel Principal Component Analysis (Kernel PCA) algorithm from scratch. Kernel PCA is an extension of the classic Principal Component Analysis (PCA) algorithm, allowing nonlinear dimensionality reduction by using kernel functions.

## Mathematics Behind Kernel PCA Algorithm

The Kernel PCA algorithm is based on the mathematical concepts of covariance matrices, eigenvectors, and eigenvalues. It incorporates kernel functions to map input data into a higher-dimensional feature space, making it possible to capture nonlinear relationships. A detailed explination of PCA algorithm is provided here [PCA Algorithm From Scratch]

[PCA Algorithm From Scratch]: https://github.com/scientistlab/PCA-Algorithm-From-Scratch


## Pseudocode of the Algorithm

```
Algorithm: Kernel Principal Component Analysis (Kernel PCA)

Procedure: PCA Initialization
    Input: Number of components (n_components)
    Output: PCA object with initialized parameters

    1. Set n_components
    2. Set mean to None
    3. Set components to None

Procedure: PCA Fitting
    Input: Data matrix (X)
    Output: Fitted PCA model

    1. Compute the mean of the input data (mean)
    2. Center the data by subtracting the mean (centered_X)
    3. Compute the covariance matrix of the centered data (covariance_matrix)
    4. Compute the eigensystem (eigenvalues, eigenvectors) of the covariance matrix
    5. Sort eigenvectors by decreasing eigenvalues
    6. Select the top n_components eigenvectors

Procedure: PCA Transformation
    Input: Data matrix (X)
    Output: Transformed data in reduced dimension

    1. Center the data by subtracting the mean (centered_X)
    2. Project the centered data onto the selected eigenvectors (transformed_data)

Procedure: PCA Fit and Transform
    Input: Data matrix (X)
    Output: Transformed data in reduced dimension

    1. Call PCA Fitting procedure
    2. Call PCA Transformation procedure

Procedure: PCA Inverse Transformation
    Input: Transformed data (X)
    Output: Original data in the original feature space

    1. Project the transformed data back to the original feature space (original_data)

Procedure: Kernel PCA Initialization
    Input: Number of components (n_components), Kernel function (kernel)
    Output: Kernel PCA object with initialized parameters

    1. Call PCA Initialization procedure
    2. Set kernel to RBFKernel() if not provided
    3. Set X to None

Procedure: Kernel PCA Fitting
    Input: Data matrix (X)
    Output: Fitted Kernel PCA model

    1. Transform the data using the kernel function (K)
    2. Call PCA Fitting procedure on the transformed data (K)

Procedure: Kernel PCA Transformation
    Input: Data matrix (X)
    Output: Transformed data in reduced dimension

    1. Transform the data using the kernel function (K)
    2. Call PCA Transformation procedure on the transformed data (K)

Procedure: Kernel PCA Inverse Transformation
    Input: Transformed data (X)
    Output: Original data in the original feature space

    1. Transform the data using the kernel function (K)
    2. Call PCA Inverse Transformation procedure on the transformed data (K)
```

## Kernel PCA vs PCA

Kernel PCA extends the linear PCA by introducing kernel functions that enable the algorithm to capture nonlinear patterns in the data. While PCA is suitable for linear relationships, Kernel PCA excels in handling complex, nonlinear structures.

## Modules

### Module 1: Datasets
make_circles: Generate synthetic circular datasets with control over noise, number of circles, and balanced/unbalanced samples.

### Module 2: Plots
scatter2D: Create a 2D scatter plot.
scatter3D: Create a 3D scatter plot.

### Module 3: Dimension Reduction
PCA: Classic PCA implementation for linear dimensionality reduction.
KernelPCA: Extension of PCA using kernel functions for nonlinear dimensionality reduction.

### Module 4: Kernels
LinearKernel: Linear kernel function.
PolynomialKernel: Polynomial kernel function.
RBFKernel: Radial Basis Function (RBF) kernel function.

### Module 5: Datasets
print_shape: Print the shape of input data and labels.

## Notebooks
Jupyter notebooks demonstrating the usage of the implemented Kernel PCA algorithm.

### Example 1
Using make_circles with three different kernel PCA implementations (linear, polynomial, RBF) and comparing their performance.

### Example 2
Using make_circles with added noise and applying three different kernel PCA implementations, demonstrating the superiority of RBF over linear and polynomial kernels.

## Usage
To use the project, Make sure to install necessary dependencies by running pip install numpy matplotlib before executing the code in the notebook.

## Contributing
The project welcomes contributions from other users. They can open an issue or submit a pull request with their ideas or changes.

## License
The project is licensed under the terms of the MIT license.



