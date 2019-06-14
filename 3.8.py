from mxnet import autograd, nd
from matplotlib import pyplot as plt


def xyplot(x_vals, y_vals, name):
    plt.rcParams['figure.figsize'] = (5, 2.5)
    plt.plot(x_vals.asnumpy(), y_vals.asnumpy())
    plt.xlabel('x')
    plt.ylabel(name + '(x)')
    plt.show()


x = nd.arange(-8.0, 8.0, 0.1)
x.attach_grad()
with autograd.record():
    # y = x.relu()
    # y = x.sigmoid()
    y = x.tanh()
# xyplot(x, y, 'relu')
# xyplot(x, y, 'sigmoid')
# xyplot(x, y, 'tanh')
y.backward()
# xyplot(x, x.grad, 'grad of sigmoid')
xyplot(x, x.grad, 'grad of tanh')
