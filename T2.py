import time
import tracemalloc

import numpy as np
from scipy.optimize import least_squares, minimize, LinearConstraint, NonlinearConstraint
import torch
import numdifftools as nd


def func(w, x):
    return (w[0] * x) ** 2 + np.cos(x) * w[1] ** 2


def test_jacob(w, x):
    jac = np.zeros((len(X), len(w)))
    for i in range(len(X)):
        jac[i] = np.array([2 * w[0] * x[i] ** 2, 2 * w[1] * np.cos(x[i])])
    return jac


def generate_points(n, f, rg=1):
    X = rg * np.random.uniform(0, 1, n)
    y = []
    X_err = X
    for x in X_err:
        y.append(f([5, 12], x))
    return X, np.asarray(y)


X, y = generate_points(100, func)


def residuals(w, x, y):
    return y - (w[0] * x) ** 2 - np.cos(x) * w[1] ** 2


res = least_squares(residuals, [1.0, 1.0], args=(X, y), method='lm')

#print(res.x, mem, time_end - time_start, res)


def f_B(x):
    return 0.5 * x[0] ** 2 + np.sin(x[1]) * 2 * np.cos(x[0] + x[1] * 2)


def f(x):
    return 0.5 * x ** 2 + np.sin(x) * 2 * np.cos(x + x * 2)


def f_t(x):
    return 0.5 * x ** 2 + torch.sin(x) * 2 * torch.cos(x + x * 2)


def f2_t(x):
    return 0.5 * x ** 3 - torch.log2(x * 3 + 1)


def f2(x):
    return 0.5 * x ** 3 - np.log2(x * 3 + 1)


initial_guess = [2.0, 2.0]
time_start = time.time()
tracemalloc.start()
result = minimize(f_B, initial_guess, method='BFGS', tol=1e-7)
time_end = time.time()
mem = tracemalloc.get_tracemalloc_memory()
tracemalloc.stop()
print(result, mem, time_end - time_start)
time_start = time.time()
tracemalloc.start()
resultB = minimize(f_B, initial_guess, method='L-BFGS-B', tol=1e-7)
time_end = time.time()
mem = tracemalloc.get_tracemalloc_memory()
tracemalloc.stop()
print(resultB,  mem, time_end - time_start)


# B
def num_grad(f, x, eps=1e-4):
    return (f(x + eps) - f(x - eps)) / (2 * eps)


def num_dif(f, x):
    df = nd.Gradient(f)
    return df(x)


def torch_grad(f, x):
    x_g = torch.tensor(x, requires_grad=True)
    z = f(x_g)
    z.backward()
    return x_g.grad.item()


def test():
    start_time = time.time()
    tracemalloc.start()
    g1 = num_grad(f, 3)
    end_time = time.time()
    mem = tracemalloc.get_tracemalloc_memory()
    tracemalloc.stop()
    print("num_grad_f1", g1, mem, end_time - start_time)
    start_time = time.time()
    tracemalloc.start()
    g2 = num_grad(f2, 3.0)
    end_time = time.time()
    mem = tracemalloc.get_tracemalloc_memory()
    tracemalloc.stop()
    print("num_grad_f2", g2, mem, end_time - start_time)
    start_time = time.time()
    tracemalloc.start()
    g1 = num_dif(f, 3.0)
    end_time = time.time()
    mem = tracemalloc.get_tracemalloc_memory()
    tracemalloc.stop()
    print("num_dif_f1", g1, mem, end_time - start_time)
    start_time = time.time()
    tracemalloc.start()
    g2 = num_dif(f2, 3.0)
    end_time = time.time()
    mem = tracemalloc.get_tracemalloc_memory()
    tracemalloc.stop()
    print("num_dif_f2", g2, mem, end_time - start_time)
    start_time = time.time()
    tracemalloc.start()
    g1 = torch_grad(f_t, 3.0)
    end_time = time.time()
    mem = tracemalloc.get_tracemalloc_memory()
    tracemalloc.stop()
    print("torch_grad_f1", g1, mem, end_time - start_time)
    start_time = time.time()
    tracemalloc.start()
    g2 = torch_grad(f2_t, 3.0)
    end_time = time.time()
    mem = tracemalloc.get_tracemalloc_memory()
    tracemalloc.stop()
    print("torch_grad_f2", g2, mem, end_time - start_time)


test()


# C
def f(x):
    return 2 * x ** 2 + 0.5 * x ** 3 + 1


def f2(x):
    return 0.5 * x ** 2 + np.sin(x) * 2 * np.cos(x + x * 2)


def test_bound(bound):
    time_start = time.time()
    tracemalloc.start()
    ans = minimize(f, x0=-20, method='L-BFGS-B', bounds=(bound,))
    time_end = time.time()
    mem = tracemalloc.get_tracemalloc_memory()
    tracemalloc.stop()
    print("f1")
    print(ans.x, ans.fun, time_end - time_start, mem)
    time_start = time.time()
    tracemalloc.start()
    ans = minimize(f2, x0=3, method='L-BFGS-B', bounds=(bound,))
    time_end = time.time()
    mem = tracemalloc.get_tracemalloc_memory()
    tracemalloc.stop()
    print("f2")
    print(ans.x, ans.fun, time_end - time_start, mem)


bounds = [[-1, 1], [-10, 10], [-50, 50], [-100, 100]]
for i in bounds:
    test_bound(i)

time_start = time.time()
tracemalloc.start()
ans = minimize(f, x0=-20, method='L-BFGS-B')
time_end = time.time()
mem = tracemalloc.get_tracemalloc_memory()
tracemalloc.stop()
print("f1_unbound")
print(ans.x, ans.fun, time_end - time_start, mem)


# Bonus
def f(x):
    return 2 * x ** 2 + 0.5 * x ** 3 + 1


initial_guess = [2.0]
A = np.array([[1.0]])  # Матрица A
b = np.array([3.0])    # Вектор b


linear_constraint = LinearConstraint(A, -2.5, 3)


#linear_constraint = LinearConstraint(A, -np.inf, b)


def constraint_function(x):
    return 2 * x ** 3 - x ** 2 + x


non_linear_constraint = NonlinearConstraint(constraint_function, lb=-5, ub=20)
time_start = time.time()
tracemalloc.start()
#result = minimize(f, initial_guess, method='SLSQP', constraints=non_linear_constraint)
time_end = time.time()
mem = tracemalloc.get_tracemalloc_memory()
tracemalloc.stop()
result = minimize(f, initial_guess, method='SLSQP', constraints=linear_constraint)

print(mem,  time_end- time_start)
optimized_parameters = result.x


optimal_value = result.fun

print("Оптимизированные параметры:", optimized_parameters)
print("Значение оптимизируемой функции в оптимуме:", optimal_value)

