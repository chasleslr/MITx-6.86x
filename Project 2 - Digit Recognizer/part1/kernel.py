import numpy as np

### Functions for you to fill in ###



def polynomial_kernel(X, Y, c, p):
    """
        Compute the polynomial kernel between two matrices X and Y::
            K(x, y) = (<x, y> + c)^p
        for each pair of rows x in X and y in Y.

        Args:
            X - (n, d) NumPy array (n datapoints each with d features)
            Y - (m, d) NumPy array (m datapoints each with d features)
            c - a coefficient to trade off high-order and low-order terms (scalar)
            p - the degree of the polynomial kernel

        Returns:
            kernel_matrix - (n, m) Numpy array containing the kernel matrix
    """
    kernel_matrix = np.power((np.dot(X, Y.T) + c), p)
    return kernel_matrix


def rbf_kernel(X, Y, gamma):
    """
        Compute the Gaussian RBF kernel between two matrices X and Y::
            K(x, y) = exp(-gamma ||x-y||^2)
        for each pair of rows x in X and y in Y.

        Args:
            X - (n, d) NumPy array (n datapoints each with d features)
            Y - (m, d) NumPy array (m datapoints each with d features)
            gamma - the gamma parameter of gaussian function (scalar)

        Returns:
            kernel_matrix - (n, m) Numpy array containing the kernel matrix

            n = X.shape[0]
            m = Y.shape[0]
            for i in range(n):
                for j in range(m):

    """

    # calculating row-wise distance using loop -- less efficient
    """
    n = X.shape[0]
    m = Y.shape[0]
    C = np.zeros((n,m))

    for i in range(n):
        for j in range(m):
            C[i][j] = np.linalg.norm(X[i] - Y[j])
    
    kernel_matrix = np.exp(-gamma * np.power(C, 2))
    """

    # calculating row-wise distance using matrix notation -- more efficient
    C = X[:, np.newaxis] - Y
    kernel_matrix = np.exp(-gamma * np.power(np.linalg.norm(C.T,axis=0).T, 2))
    return kernel_matrix

