# -*- coding: utf-8 -*-
"""
Used to repair non-positive definite correlation matrices.
The eigenvalues of corr matrices must be positive.
Negative eigenvalues result from numerical error (i.e., -1.22e-16); this script helps repair those.

Source: https://pyportfolioopt.readthedocs.io/en/latest/_modules/pypfopt/risk_models.html
"""

import numpy as np
import pandas as pd
import warnings
# -*- coding: utf-8 -*-
"""
Author: Trevor Amestoy
Cornell University
Spring 2022

Contains several basic functions that are used in the generation of synthetic
inflow and demand timeseries.

"""
import pandas as pd
import numpy as np



###############################################################################


def check_positive_definite(A):
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False


###############################################################################


def cholskey_repair_and_decompose(A, 
                                max_iter = 20, 
                                debugging = True,
                                method = 'spectral'):
    """
    Attempts to perform cholskey decomp, and repairs as needed. 

    Method: https://www.mathworks.com/matlabcentral/answers/6057-repair-non-positive-definite-correlation-matrix

    A : matrix to be decomposed
    """
    machine_eps = np.finfo(float).eps

    pass_test = check_positive_definite(A)
    if pass_test:
        return np.linalg.cholesky(A)
    
    elif not pass_test:
        if debugging:
            print('Matrix is not posdef. Attempting repair.')
        i = 0
    
        while not pass_test and (i < max_iter):
            # Get eigenvalues
            L, V = np.linalg.eig(A)
            real_lambdas = np.real(L)

            # Find the minimum eigenvalue
            neg_lambda = min(real_lambdas)
            neg_lambda_index = np.argmin(real_lambdas)
            neg_V = V[:, neg_lambda_index]
            if debugging:
                print(f'Negative lambda ({neg_lambda}) in column {neg_lambda_index}.')

            if method == "spectral":
                # Remove negative eigenvalues
                pos_L = np.where(L > 0, L, 0)
                # Reconstruct matrix
                A = V @ np.diag(pos_L) @ V.T
            
            elif method == 'rank':
                # Add machine epsilon
                shift = np.spacing(neg_lambda) - neg_lambda
                A = A + neg_V* neg_V.T * shift
            
            elif method == 'custom_diag':
                diff = min([neg_lambda, -np.finfo(float).eps])
                shift = diff * np.eye(A.shape[0])
                A = A - shift
        
            elif method == 'simple_diag':    
                A = A + machine_eps * np.eye(len(A))

            else:
                print('Invalid repair method specified.')
                return

            pass_test = check_positive_definite(A)
            
            i += 1

        if pass_test:
            if debugging:
                print(f'Matrix repaired after {i} updates.')
            return np.linalg.cholesky(A)
        elif not pass_test:
            print(f'Matrix could not repaired after {i} updates!')
            return 

def _is_positive_semidefinite(matrix):
    """
    Helper function to check if a given matrix is positive semidefinite.
    Any method that requires inverting the covariance matrix will struggle
    with a non-positive semidefinite matrix

    :param matrix: (covariance) matrix to test
    :type matrix: np.ndarray, pd.DataFrame
    :return: whether matrix is positive semidefinite
    :rtype: bool
    """
    try:
        # Significantly more efficient than checking eigenvalues (stackoverflow.com/questions/16266720)
        np.linalg.cholesky(matrix + 1e-16 * np.eye(len(matrix)))
        return True
    except np.linalg.LinAlgError:
        return False


def fix_nonpositive_semidefinite(matrix, fix_method="spectral"):
    """
    Check if a covariance matrix is positive semidefinite, and if not, fix it
    with the chosen method.

    The ``spectral`` method sets negative eigenvalues to zero then rebuilds the matrix,
    while the ``diag`` method adds a small positive value to the diagonal.

    :param matrix: raw covariance matrix (may not be PSD)
    :type matrix: pd.DataFrame
    :param fix_method: {"spectral", "diag"}, defaults to "spectral"
    :type fix_method: str, optional
    :raises NotImplementedError: if a method is passed that isn't implemented
    :return: positive semidefinite covariance matrix
    :rtype: pd.DataFrame
    """
    if _is_positive_semidefinite(matrix):
        return matrix

    warnings.warn(
        "The covariance matrix is non positive semidefinite. Amending eigenvalues."
    )

    # Eigendecomposition
    q, V = np.linalg.eigh(matrix)

    if fix_method == "spectral":
        # Remove negative eigenvalues
        q = np.where(q > 0, q, 0)
        # Reconstruct matrix
        fixed_matrix = V @ np.diag(q) @ V.T
    elif fix_method == "diag":
        min_eig = np.min(q)
        fixed_matrix = matrix - 1.1 * min_eig * np.eye(len(matrix))
    else:
        raise NotImplementedError("Method {} not implemented".format(fix_method))

    if not _is_positive_semidefinite(fixed_matrix):  # pragma: no cover
        warnings.warn(
            "Could not fix matrix. Please try a different risk model.", UserWarning
        )

    # Rebuild labels if provided
    if isinstance(matrix, pd.DataFrame):
        tickers = matrix.index
        return pd.DataFrame(fixed_matrix, index=tickers, columns=tickers)
    else:
        return fixed_matrix
    
def get_hurst_exponent(time_series, max_lag=20):
	"""Returns the Hurst Exponent of the time series"""
	lags = range(2, max_lag)

	# variances of the lagged differences
	tau = [np.std(np.subtract(time_series[lag:], time_series[:-lag])) for lag in lags]

	# calculate the slope of the log plot -> the Hurst Exponent
	reg = np.polyfit(np.log(lags), np.log(tau), 1)
	return reg[0]



    
def nearestPD(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = la.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(la.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(la.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3


def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = la.cholesky(B)
        return True
    except la.LinAlgError:
        return False
