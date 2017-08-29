import numpy as np
import sys


def form_matrix(values, n=5, m=5, k=2):
    util_matrix = np.array(np.zeros((n, m)))
    U = np.array(np.ones((n, f)))
    V = np.array(np.ones((f, m)))
    for i in values:
        util_matrix[i[0]-1][i[1]-1] = i[2]
    return util_matrix, U, V


def rmse(matrix, U, V):
    e, m = 0, 0
    r, c = matrix.shape
    for i in range(r):
        for j in range(c):
            if matrix[i, j] != 0:
                e += (matrix[i, j] - np.dot(U[i, :], V[:, j]))**2
                m += 1
    return np.sqrt(e/m)


def find_x(r=0, s=0):

    denominator = 0.0
    numerator = 0.0
    for j in range(m):
        if matrix[r, j] != 0:
            denominator += V[s, j] ** 2
    for j in range(m):
            k_not_s_sum = 0.0
            for d in range(f):
                if d != s:
                    k_not_s_sum += U[r, d]*V[d, j]
            if matrix[r, j] != 0:
                numerator += V[s, j]*(matrix[r, j] - k_not_s_sum)

    return numerator/denominator


def find_y(r=0, s=0):
    denominator = 0.0
    numerator = 0.0
    for i in range(n):
        if matrix[i, s] != 0:
            denominator += U[i, r] ** 2
    for i in range(n):
            k_not_s_sum = 0.0
            for d in range(f):
                if d != r:
                    k_not_s_sum += U[i, d] * V[d, s]
            if matrix[i, s] != 0:
                numerator += U[i, r] * (matrix[i, s] - k_not_s_sum)

    return numerator/denominator


def uv_decomposition():
    for r in range(n):
        for s in range(f):
            U[r, s] = find_x(r, s)
    #print(U)
    for r in range(f):
        for s in range(m):
            V[r,s] = find_y(r,s)
    #print(V)
    print("%.4f"%rmse(matrix,U,V))

file = open(sys.argv[1], 'r+')
file_input = file.readlines()
lines = []
n = int(sys.argv[2])
m = int(sys.argv[3])
f = int(sys.argv[4])
k = int(sys.argv[5])
for line in file_input:
    lines.append([int(y) for y in line.strip('\n').split(',')])
matrix, U, V = form_matrix(lines,n,m,k)


for i in range(k):
    uv_decomposition()
