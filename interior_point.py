import numpy as np
import matplotlib.pyplot as plt

A = np.array([[1,1,-1,0,0],[3,1,0,-1,0],[1,3,0,0,-1]])
c = np.array([1,1,0,0,0]).T

# # Null(A) projection of c, gives same solution but different objective value :).
# # Used for Sanity checking my answer to problem 6.
# #
# # c = np.array([0.05263158, 0.05263158, 0.10526316, 0.21052632, 0.21052632])

b = np.array([2,3,3]).T

x = np.array([2,2,2,5,5],dtype=np.float64).T
y = np.array([0.1,0.1,0.1],dtype=np.float64).T
s = np.array([0.5,0.5,0.1,0.1,0.1],dtype=np.float64).T
tol = 1e-4
step = 0.95


# A = np.array([[8,5,3,0,1,0,0,0],[6,9,8,0,0,1,0,0],[7,6,8,-4,0,0,-1,0],[5,9,4,-3,0,0,0,-1]])
# c = np.array([0,0,0,-1,0,0,0,0]).T
# b = np.array([100,200,0,0]).T

# x = np.array([1,1,1,1,84,177,17,15]).T
# y = np.array([-0.24,-0.53,0.33,0.34]).T
# s = np.array([1.11,0.97,1.01,1.33,0.23,0.54,0.33,0.34]).T

e = np.ones(len(x))
mu = x.T @ s / 100
itr = 0

while not (0 <= x.T @ s < tol):
    dy = np.linalg.solve((A * x/s) @ A.T, (b - mu*A@(1/s)))
    ds = -A.T @ dy
    dx = (mu*e-x*s-x*ds)/s

    theta = np.min((-x/dx)[np.where(dx<0)])
    alpha = np.min((-s/ds)[np.where(ds<0)])

    # print(rf"""{itr+1} & $\begin{{bmatrix}} {r"\\".join(list(map(str, np.round(y,2))))} \end{{bmatrix}}$ &
    #     $\begin{{bmatrix}} {r"\\".join(list(map(str, np.round(s,2))))} \end{{bmatrix}}$ &
    #     $\begin{{bmatrix}} {r"\\".join(list(map(str, np.round(x,2))))} \end{{bmatrix}}$ &
    #     $\begin{{bmatrix}} {r"\\".join(list(map(str, np.round(dy,2))))} \end{{bmatrix}}$ &
    #     $\begin{{bmatrix}} {r"\\".join(list(map(str, np.round(dx,2))))} \end{{bmatrix}}$ &
    #     $\begin{{bmatrix}} {r"\\".join(list(map(str, np.round(ds,2))))} \end{{bmatrix}}$ &
    #     {round(theta,2)} &
    #     {round(alpha,2)} &
    #     $\begin{{bmatrix}} {r"\\".join(list(map(str, np.round(y+step*theta*dy,2))))} \end{{bmatrix}}$ &
    #     $\begin{{bmatrix}} {r"\\".join(list(map(str, np.round(s+step*theta*ds,2))))} \end{{bmatrix}}$ &
    #     $\begin{{bmatrix}} {r"\\".join(list(map(str, np.round(x+step*theta*dx,2))))} \end{{bmatrix}}$ &
    #     {round((x+step*theta*dx).T @ s / 100,4)}
    #     \\  \hline """)
    x += step*theta*dx
    y += step*alpha*dy
    s += step*alpha*ds
    mu = x.T @ s / 100
    itr += 1

print("Primal: ",np.round(x,3))
print("Objective: ", c.T@x)
print("Dual Gap: ", x.T@s)
print("Iterations: ", itr)

print()
print("Dual: ",y)
print("Slack: ",s)