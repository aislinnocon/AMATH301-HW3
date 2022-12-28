

import numpy as np
import scipy.linalg

A = np.genfromtxt("bridge_matrix.csv", delimiter=",")
b = np.array([[0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [2.], [0.], [8.], [0.], [4.]])

A1 = scipy.linalg.solve(A,b)


P, L, U = scipy.linalg.lu(A)
P = P.T

A2 = L.copy()

y = scipy.linalg.solve_triangular(L, P @ b, lower=True)
A3 = y

f = 0
while np.max(np.abs(f)) <= 30:
    b[8] = b[8] + 0.01
    y = scipy.linalg.solve_triangular(L, P @ b, lower=True)
    f = scipy.linalg.solve_triangular(U, y)

A4 = b[8]
index = np.argmax(np.abs(f)) + 1
A5 = index


# Problem 2

A = np.array([[1-(-0.003), -0.05], [0.05, 1-(-0.003)]])

x0 = np.array([[1, -1]])

pos = np.random.rand(2, 1001)
pos[:,0] = x0

P, L, U = scipy.linalg.lu(A)

P = P.T

for i in range(1, 1001):
    y = scipy.linalg.solve_triangular(L, P @ pos[:, i - 1], lower=True)
    pos[:,i] = scipy.linalg.solve_triangular(U, y)

xVal = pos[0,:]
A6 = xVal.reshape(1, 1001)

yVal = pos[1,:]
A7 = yVal.reshape(1, 1001)
print(A6.shape)
print(A7.shape)

def distance(x, y):
    d = np.sqrt(x**2 + y**2)
    return d

dist = np.zeros(1001)

for k in range(1001):
    dist[k] = distance(pos[0, k], pos[1, k])

A8 = dist.reshape(1, 1001)

origin = 0

for r in range(1001):
    origin = distance(pos[0, r], pos[1, r])
    if origin <= 0.05:
        break

A9 = r
print("A9 =", A9)
failA9 = 0.049985320697560925
A10 = origin
print("r =", r)
failr = 789
# Problem 3

def rotation(theta):
    R = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
    return R

A11 = rotation(np.pi/10)

x = np.array([[0.3, 1.4, -2.7]]).T
A12 = rotation(3*np.pi/8) @ x

print("A12 = ", A12)

y = np.array([[1.2, -0.3, 2]]).T
rTheta = rotation(np.pi/7)


P, L, U = scipy.linalg.lu(rTheta)
P = P.T
angle = scipy.linalg.solve_triangular(L, P @ y, lower=True)
x = scipy.linalg.solve_triangular(U, angle)

A13 = x
print(A13)

R = rotation(5*np.pi/7)
A14 = scipy.linalg.inv(R)


inverse = rotation(5* np.pi/7)

A15 = -5* np.pi/7
