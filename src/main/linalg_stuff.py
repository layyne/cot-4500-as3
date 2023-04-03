from typing import Tuple
import numpy as np


def gaussian_elimination(A, b):
    n = len(b)
    # Set up augmented matrix
    Ab = np.append(np.array(A), np.array([b]).T, axis=1).astype('float64')

    # Elimination
    for i in range(n - 1):
        # Find pivot and bring to diagonal
        max_row = i
        for j in range(i + 1, n - 1):
            if (abs(Ab[j, i]) > abs(Ab[max_row, i])):
                max_row = j
        Ab[i], Ab[max_row] = Ab[max_row], Ab[i]

        for j in range(i + 1, n):
            m = Ab[j, i] / Ab[i, i]
            Ab[j] -= m * Ab[i]

    # Backward subsitution
    x = np.zeros(n)
    x[-1] = Ab[n-1, n] / Ab[n-1, n-1]
    for i in reversed(range(n - 1)):
        x[i] = (Ab[i, n] - sum(Ab[i, j] * x[j]
                for j in range(i + 1, n))) / Ab[i, i]

    return x


def lu_factor(A):
    A = np.array(A)
    n = len(A)
    L, U = np.zeros((n, n)), np.zeros((n, n))
    # L will have its diagonal be all 1s
    np.fill_diagonal(L, 1)

    U[0, 0] = A[0, 0]
    if U[0, 0] == 0:
        # Factorization impossible
        return 0, None, None

    for j in range(1, n):
        # First row of U
        U[0, j] = A[0, j] / L[0, 0]
        # First column of L
        L[j, 0] = A[j, 0] / U[0, 0]

    for i in range(1, n - 1):
        U[i, i] = A[i, i] - sum(L[i, k] * U[k, i] for k in range(i))
        if U[i, i] == 0:
            # Factorization impossible
            return 0, None, None

        for j in range(i + 1, n):
            # ith row of U
            U[i, j] = (A[i, j] - sum(L[i, k] * U[k, j]
                       for k in range(i))) / L[i, i]

            # ith column of L
            L[j, i] = (A[j, i] - sum(L[j, k] * U[k, i]
                       for k in range(i))) / U[i, i]

    U[-1, -1] = A[-1, -1] - sum(L[-1, k] * U[k, -1] for k in range(n - 1))

    # L and U triangular, use det LU = det L * det U
    detL = np.prod(np.diag(L))
    detU = np.prod(np.diag(U))
    determinant = float(detL * detU)

    return determinant, L, U


def is_diagonally_dominate(A):
    A = np.array(A)
    n = len(A)
    return all(
        abs(A[i, i]) >= sum(abs(A[i, j]) for j in range(n) if j != i)
        for i in range(n)
    )


def is_positive_definite(A):
    eigenvalues, _ = np.linalg.eig(A)
    return all(eigenvalue >= 0 for eigenvalue in eigenvalues)
