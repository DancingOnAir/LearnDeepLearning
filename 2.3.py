from mxnet import nd, autograd

x = nd.arange(4).reshape(4, 1)
print(x)

x.attach_grad()
with autograd.record():
    y = 2 * nd.dot(x.T, x)

y.backward()
# assert(x.grad - 4 * x).norm().asscalar() == 0
print(x.grad)

print(autograd.is_training())
with autograd.record():
    print(autograd.is_training())


def f(a):
    b = a * 2
    while b.norm().asscalar() < 1000:
        b = b * 2
    if b.sum().asscalar() > 0:
        c = b
    else:
        c = 100 * b
    return c


a = nd.random.normal(shape=1)
a.attach_grad()
with autograd.record():
    c = f(a)
c.backward()

print(a.grad == c / a)
