import tracemalloc
import time

import numpy as np
import torch
import torch.optim as optim

torch.manual_seed(42)
x = torch.randn(100, 1)
y = 2.6 + 3 * x + 0.1 * torch.randn(100, 1)


class LinearRegression(torch.nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


optim_arr = [ 'adam', 'adagrad', 'RMS']


def torch_optim():
    eps = 1e-6

    for elem in optim_arr:
        model = LinearRegression()
        criterion = torch.nn.MSELoss()
        if elem == 'sgd':
            time_start = time.time()
            tracemalloc.start()
            optimizer = optim.SGD(model.parameters(), lr=0.01)
        elif elem == 'moment':
            time_start = time.time()
            tracemalloc.start()
            optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.6)
        elif elem == 'adam':
            time_start = time.time()
            tracemalloc.start()
            optimizer = optim.Adam(model.parameters(), lr=0.9)
        elif elem == 'adagrad':
            time_start = time.time()
            tracemalloc.start()
            optimizer = optim.Adagrad(model.parameters(), lr=0.8)
        elif elem == "RMS":
            time_start = time.time()
            tracemalloc.start()
            optimizer = optim.RMSprop(model.parameters(), lr=0.6)
        prev_loss = None
        num_epochs = 300
        for epoch in range(num_epochs):
            outputs = model(x)
            loss = criterion(outputs, y)

            if prev_loss is not None and abs(prev_loss - loss) < eps:
                break
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            prev_loss = loss
        time_end = time.time()
        mem = tracemalloc.get_tracemalloc_memory()
        tracemalloc.stop()

        print("Обученные параметры:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(param.data.item(), epoch, time_end - time_start,
                      mem, loss.item())


x_array = x.numpy()
y_array = y.numpy()
w_start = np.array([1.0, 1.0])


def mse(w):
    ans = 0
    for i in range(len(x_array)):
        ans += (y_array[i][0] - (w[0] + w[1] * x_array[i][0])) ** 2
    return ans


def grad(x, y, w):
    fst = 0
    scd = 0
    for i in range(len(x)):
        fst += -2 * (y[i][0] - (w[0] + w[1] * x[i][0]))
        scd += -2 * x[i][0] * (y[i][0] - (w[0] + w[1] * x[i][0]))
    return np.array([scd, fst])


def grad_s(x, y, w):
    return [-2 * (y - np.sum(w * np.array([1, x]))), -2 * x * (y - np.sum(w * np.array([1, x])))]


def sgd():
    w = np.array([1.0, 1.0])
    eps = 1e-4
    for epoch in range(500):

        grad = np.array([0., 0.])
        for j in range(100):
            grad += np.array(grad_s(x_array[j][0], y_array[j][0], w))
        grad /= 100
        w -= 0.05 * grad
        if np.linalg.norm(grad) < eps:
            break
    return w, epoch


def momentum():
    w = np.array([1.0, 1.0])
    v = 0
    momentum = 0.8
    lr = 0.08
    eps = 1e-4
    for epoch in range(500):
        grad = np.array([0., 0.])
        for j in range(100):
            grad += np.array(grad_s(x_array[j][0], y_array[j][0], w))
        grad /= 100
        v = momentum * v + lr * grad
        w -= v
        if np.linalg.norm(grad) < eps:
            break
    return w, epoch


def adagrad():
    w = np.array([1.0, 1.0])
    eps = 1e-4
    lr = 0.8
    G = 0
    grad = np.array([0., 0.])
    for epoch in range(500):
        for j in range(100):
            grad += np.array(grad_s(x_array[j][0], y_array[j][0], w))
        grad /= 100
        G += grad * grad.T
        v = lr * grad / np.sqrt(G + eps)
        w -= v
        if np.linalg.norm(grad) < eps:
            break
    return w, epoch


def RSMprop():
    w = np.array([1.0, 1.0])
    eps = 1e-4
    lr = 0.6
    G = 0
    gamma = 0.8
    alpha = 0
    grad = np.array([0., 0.])
    for epoch in range(500):
        for j in range(100):
            grad += np.array(grad_s(x_array[j][0], y_array[j][0], w))
        grad /= 100
        G += grad * grad.T
        alpha = gamma * alpha + (1 - gamma) * G
        w -= lr * grad / (np.sqrt(alpha) + eps)
        if np.linalg.norm(grad) < eps:
            break
    return w, epoch


def adam():
    w = np.array([1.0, 1.0])
    eps = 1e-4
    lr = 0.8
    G = 0
    beta = 0.999
    alpha = 0
    m = 0
    v = 0
    grad = np.array([0., 0.])
    for epoch in range(1, 100):
        for j in range(100):
            grad += np.array(grad_s(x_array[j][0], y_array[j][0], w))
        grad /= 100
        G = grad * grad.T
        m = alpha * m + (1 - alpha) * grad
        v = beta * v + (1 - beta) * G

        vHat = v / (1 - beta ** epoch)
        mHat = m / (1 - alpha ** epoch)
        w -= lr * mHat / (np.sqrt(vHat) + eps)
        if np.linalg.norm(grad) < eps:
            break
    return w, epoch


def start():
    for elem in optim_arr:
        if elem == 'sgd':
            time_start = time.time()
            tracemalloc.start()
            w, iter = sgd()
            time_end = time.time()
            mem = tracemalloc.get_tracemalloc_memory()
            tracemalloc.stop()
            print(w, iter, time_end - time_start, mem, mse(w))
        elif elem == 'moment':
            time_start = time.time()
            tracemalloc.start()
            w, iter = momentum()
            time_end = time.time()
            mem = tracemalloc.get_tracemalloc_memory()
            tracemalloc.stop()
            print(w, iter, time_end - time_start, mem, mse(w))
        elif elem == 'adagrad':
            time_start = time.time()
            tracemalloc.start()
            w, iter = adagrad()
            time_end = time.time()
            mem = tracemalloc.get_tracemalloc_memory()
            tracemalloc.stop()
            print(w, iter, time_end - time_start, mem, mse(w))
        elif elem == "RMS":
            time_start = time.time()
            tracemalloc.start()
            w, iter = RSMprop()
            time_end = time.time()
            mem = tracemalloc.get_tracemalloc_memory()
            tracemalloc.stop()
            print(w, iter, time_end - time_start, mem, mse(w))
        elif elem == 'adam':
            time_start = time.time()
            tracemalloc.start()
            w, iter = adam()
            time_end = time.time()
            mem = tracemalloc.get_tracemalloc_memory()
            tracemalloc.stop()
            print(w, iter, time_end - time_start, mem, mse(w))


start()
torch_optim()
