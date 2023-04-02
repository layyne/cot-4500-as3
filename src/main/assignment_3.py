import builtins
from typing import Any, Tuple
from initial_value import *
from linalg_stuff import *


# Override default print function for proper float formatting/whitespace
def print(*args, **kwargs):
    args = [f'{arg:.5f}' if type(arg) is float else arg for arg in args]
    kwargs.setdefault('end', '\n\n')
    builtins.print(*args, **kwargs, sep='\n\n')


def q1():
    def dy(t, y):
        return t - y**2

    a, b = 0, 2
    n = 10
    alpha = 1

    return euler(dy, a, b, n, alpha)


def q2():
    def dy(t, y):
        return t - y**2

    a, b = 0, 2
    n = 10
    alpha = 1

    return runge_kutta(dy, a, b, n, alpha)


def q3():
    A = [[2, -1, 1],
         [1, 3, 1],
         [-1, 5, 4]]

    b = [6, 0, -3]

    return gaussian_elimination(A, b)


def q4():
    A = [[1, 1, 0, 3],
         [2, 1, -1, 1],
         [3, -1, -1, 2],
         [-1, 2, 3, -1]]

    return lu_factor(A)


def q5():
    A = [[9, 0, 5, 2, 1],
         [3, 9, 1, 2, 1],
         [0, 1, 7, 2, 3],
         [4, 2, 3, 12, 2],
         [3, 2, 4, 0, 8]]

    return is_diagonally_dominate(A)


def q6():
    A = [[2, 2, 1],
         [2, 3, 0],
         [1, 0, 2]]

    return is_positive_definite(A)


if __name__ == '__main__':
    print(q1())
    print(q2())
    print(q3())
    print(*q4())
    print(q5())
    print(q6(), end='\n')
