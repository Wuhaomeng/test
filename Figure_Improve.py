import numpy as np
from numpy.linalg import solve
import time
from scipy.sparse.linalg import gmres, lgmres
from scipy.sparse import csr_matrix

if __name__ == '__main__':
    alpha = 0.8
    vertex = ['A', 'B', 'C', 'a', 'b', 'c', 'd']
    #矩阵是由图的来的 A,B,C,a,b,c,d分别为矩阵的行和列
    M = np.matrix([[0, 0, 0, 0.5, 0, 0.5, 0],
                   [0, 0, 0, 0.25, 0.25, 0.25, 0.25],
                   [0, 0, 0, 0, 0, 0.5, 0.5],
                   [0.5, 0.5, 0, 0, 0, 0, 0],
                   [0, 1.0, 0, 0, 0, 0, 0],
                   [0.333, 0.333, 0.333, 0, 0, 0, 0],
                   [0, 0.5, 0.5, 0, 0, 0, 0]])
    r0 = np.matrix([[0], [0], [0], [0], [1], [0], [0]])  # 从'b'开始游走
    #print(r0.shape)
    n = M.shape[0]
    # print(n) 7
    # 方法一：直接解线性方程法
    A = np.eye(n) - alpha * M.T  #numpy.eye(N,M=None, k=0, dtype=<type 'float'>) 生成对角矩阵  这里生成单位矩阵
    b = (1 - alpha) * r0
    # begin = time.time()
    # r = solve(A, b)  #x = np.linalg.solve(A,b)  解线性方程组
    # end = time.time()
    # print('user time', end - begin)
    # rank = {}
    # for j in range(n):
    #     rank[vertex[j]] = r[j]
    # li = sorted(rank.items(), key=lambda x: x[1], reverse=True)
    # print(li)
    # for ele in li:
    #     print("%s:%.3f,\t" % (ele[0], ele[1]))

    # 方法二：采用CSR法对稀疏矩阵进行压缩存储，然后解线性方程
    data = list() #保存不为0的值
    row_ind = list() #保存不为0的行
    col_ind = list() #保存不为0的列
    for row in range(n):
        for col in range(n):
            if (A[row, col] != 0):
                data.append(A[row, col])
                row_ind.append(row)
                col_ind.append(col)

    AA = csr_matrix((data, (row_ind, col_ind)), shape=(n, n))  #对矩阵进行压缩
    begin = time.time()
    r = gmres(AA, b, tol=1e-08, maxiter=1)[0]  #求解稀疏表示的线性方程
    end = time.time()
    print("user time", end - begin)

    rank = {}
    for j in range(n):
        rank[vertex[j]] = r[j]
    li = sorted(rank.items(), key=lambda x: x[1], reverse=True)
    for ele in li:
        print("%s:%.3f,\t" % (ele[0], ele[1]))